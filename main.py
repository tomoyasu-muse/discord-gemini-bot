"""
Discord + Gemini AI chat bot.

- In a normal channel: when the bot is mentioned, it answers with a
  one-shot reply in that same channel (no thread, no history).
- To have a continuing conversation, use the /thread slash command:
  it creates a thread and answers inside it. Inside such a thread no
  mention is needed and the bot keeps a per-thread Gemini chat session.
- Mentioning the bot inside any thread (re)activates that thread as a
  conversation thread.
- Search: messages starting with 「調べて」 (or the /search command) let
  the model use Tavily tools (web search, plus full-page extraction
  when a snippet is not enough) and answer with cited source URLs.
  If search is unavailable, the bot falls back to a normal answer.
- keep_alive: a small Flask server runs in a background thread so
  Render (and uptime pingers) have an HTTP endpoint to hit.
"""

import asyncio
import contextvars
import datetime
import json
import os
import re
import threading
import urllib.request
import zoneinfo

import discord
import google.generativeai as genai
from discord import app_commands
from discord.ext import commands
from flask import Flask

# ---------------------------------------------------------------------------
# keep_alive (Render free-tier sleep workaround)
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/")
def home():
    return "Bot is alive!", 200


@app.route("/health")
def health():
    return {"status": "ok"}, 200


def run_web_server():
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)


def keep_alive():
    thread = threading.Thread(target=run_web_server, daemon=True)
    thread.start()


# ---------------------------------------------------------------------------
# Gemini setup
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

# Guild used for instant slash-command sync during testing.
# Global sync also runs, so other guilds get /thread as it propagates.
TEST_GUILD_ID = 1570294742137176134

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Per-thread chat sessions: thread_id -> Gemini chat session
chat_sessions: dict[int, object] = {}


def get_chat_session(thread_id: int):
    if thread_id not in chat_sessions:
        model = genai.GenerativeModel(GEMINI_MODEL)
        chat_sessions[thread_id] = model.start_chat(history=[])
    return chat_sessions[thread_id]


async def ask_gemini(thread_id: int, prompt: str) -> str:
    """Ask with the per-thread conversation history."""
    chat = get_chat_session(thread_id)
    # google-generativeai is synchronous; run it off the event loop.
    response = await asyncio.to_thread(chat.send_message, prompt)
    text = (response.text or "").strip()
    return text if text else "（応答が空でした。もう一度お試しください）"


async def ask_gemini_once(prompt: str) -> str:
    """Ask a one-shot question with no conversation history."""
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = await asyncio.to_thread(model.generate_content, prompt)
    text = (response.text or "").strip()
    return text if text else "（応答が空でした。もう一度お試しください）"


# --- Search (Tavily tools driven by Gemini automatic function calling) ---
SEARCH_TRIGGER_RE = re.compile(r"^(調べて|検索して)[\s:：、]*")

SEARCH_SYSTEM_INSTRUCTION = (
    "あなたはWeb検索ツールを使って最新情報を調べるアシスタントです。"
    "回答する前に、必ず search_web ツールで検索してください。"
    "検索結果のスニペットだけでは情報が不十分な場合は、extract_page ツールで"
    "該当ページの全文を読んでから答えてください。"
    "ニュース・今日・最新の話題を聞かれた場合は、search_web で topic='news' と"
    "time_range='day'（取れなければ 'week'）を指定し、古い日付の記事を"
    "「今日のニュース」として扱わないでください。"
    "事実に関する主張には、根拠となったページのURLを引用として必ず含めてください。"
    "URLはツールの結果に実際に含まれるものだけを使い、存在しないURLを創作してはいけません。"
    "回答は日本語で簡潔にまとめてください。"
)

# URLs actually returned by the tools during the current search request:
# list of (title, url). Used to append a guaranteed-real source list.
_current_sources: contextvars.ContextVar = contextvars.ContextVar(
    "current_sources", default=None
)


def _collect_source(title: str, url: str):
    sources = _current_sources.get()
    if sources is None or not url:
        return
    # A search-engine results page is not a source.
    if "google.com/search" in url or "bing.com/search" in url:
        return
    if any(existing_url == url for _, existing_url in sources):
        return
    sources.append((title or url, url))


def _tavily_post(path: str, payload: dict) -> dict:
    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY is not set")
    request = urllib.request.Request(
        f"https://api.tavily.com{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TAVILY_API_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _tavily_search_sync(
    query: str, search_depth: str, topic: str, time_range: str = ""
) -> list[dict]:
    payload = {
        "query": query,
        "max_results": 5,
        "search_depth": search_depth,
        "topic": topic,
    }
    if time_range:
        payload["time_range"] = time_range
    data = _tavily_post("/search", payload)
    return data.get("results", [])


def search_web(
    query: str,
    search_depth: str = "basic",
    topic: str = "general",
    time_range: str = "",
) -> str:
    """Search the web for current information.

    Args:
        query: What to search for.
        search_depth: "basic" (default, 1 credit) or "advanced"
            (2 credits, deeper search; use only when basic detail is
            unlikely to be enough).
        topic: "general" (default) or "news" (for recent news).
        time_range: Optional recency filter: "day", "week", "month"
            or "year". For questions about today or the latest news,
            use topic="news" with time_range="day" (or "week").

    Returns:
        Search results as text: title, URL and a snippet per result.
    """
    if search_depth not in ("basic", "advanced"):
        search_depth = "basic"
    if topic not in ("general", "news"):
        topic = "general"
    if time_range not in ("", "day", "week", "month", "year"):
        time_range = ""
    results = _tavily_search_sync(query, search_depth, topic, time_range)
    if not results:
        return "検索結果は見つかりませんでした。クエリを変えて再検索できます。"
    parts = []
    for index, result in enumerate(results, start=1):
        title = result.get("title") or "(no title)"
        url = result.get("url") or ""
        content = (result.get("content") or "")[:600]
        _collect_source(title, url)
        parts.append(f"[{index}] {title}\n{url}\n{content}")
    return "\n\n".join(parts)


def extract_page(url: str) -> str:
    """Read the full text of one web page (for example from search_web).

    Use this when a search snippet is not enough to answer accurately.

    Args:
        url: The page URL to read.

    Returns:
        The page content as Markdown text (truncated if very long).
    """
    data = _tavily_post("/extract", {"urls": [url], "extract_depth": "basic"})
    results = data.get("results", [])
    if not results:
        return "ページの内容を取得できませんでした。"
    first = results[0]
    _collect_source(first.get("title") or url, url)
    text = first.get("raw_content") or first.get("content") or ""
    return text[:4000] if text else "ページの内容を取得できませんでした。"


async def ask_gemini_search(prompt: str) -> str:
    """Answer with web search: the model drives the Tavily tools itself,
    then writes an answer citing the URLs it actually used."""
    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY is not set")
    sources: list = []
    token = _current_sources.set(sources)
    try:
        model = genai.GenerativeModel(
            GEMINI_MODEL,
            system_instruction=SEARCH_SYSTEM_INSTRUCTION,
            tools=[search_web, extract_page],
        )
        chat = model.start_chat(history=[], enable_automatic_function_calling=True)
        today = datetime.datetime.now(zoneinfo.ZoneInfo("Asia/Tokyo")).date()
        dated_prompt = f"（今日は {today.isoformat()} です）\n{prompt}"
        response = await asyncio.to_thread(chat.send_message, dated_prompt)
    finally:
        _current_sources.reset(token)
    try:
        text = (response.text or "").strip()
    except Exception:  # noqa: BLE001
        text = ""
    if not text:
        text = "（応答が空でした。もう一度お試しください）"
    if sources:
        text += "\n\n出典:\n" + "\n".join(
            f"- {title} {url}" for title, url in sources[:5]
        )
    return text


async def answer_with_optional_search(prompt: str, normal_ask):
    """Route 「調べて…」 prompts to search; fall back to a normal answer.

    normal_ask is an async callable taking the prompt, used both for
    non-search prompts and as the fallback when search fails (for
    example when the search model's daily quota is exhausted).
    """
    match = SEARCH_TRIGGER_RE.match(prompt)
    if not match:
        return await normal_ask(prompt)
    query = prompt[match.end():].strip() or prompt
    try:
        return await ask_gemini_search(query)
    except Exception as exc:  # noqa: BLE001
        print(f"Search failed, falling back: {exc}")
        base = await normal_ask(prompt)
        return "（検索できませんでした。通常の回答です）\n" + base


def split_message(text: str, limit: int = 2000) -> list[str]:
    """Split a long reply into Discord-sized chunks."""
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        chunks.append(text[:limit])
        text = text[limit:]
    return chunks


# ---------------------------------------------------------------------------
# Discord bot
# ---------------------------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True  # Requires "Message Content Intent" in the portal

bot = commands.Bot(command_prefix="!", intents=intents)

# Threads this bot is actively conversing in
active_threads: set[int] = set()

_commands_synced = False


@bot.event
async def on_ready():
    global _commands_synced
    print(f"Logged in as {bot.user} (id={bot.user.id})")
    if _commands_synced:
        return
    _commands_synced = True
    try:
        test_guild = discord.Object(id=TEST_GUILD_ID)
        bot.tree.copy_global_to(guild=test_guild)
        await bot.tree.sync(guild=test_guild)
        await bot.tree.sync()
        print("Slash commands synced (test guild + global)")
    except Exception as exc:  # noqa: BLE001
        print(f"Slash command sync failed: {exc}")


@bot.tree.command(name="thread", description="会話用のスレッドを作って質問します")
@app_commands.describe(question="質問・相談の内容")
async def thread_command(interaction: discord.Interaction, question: str):
    channel = interaction.channel
    if isinstance(channel, discord.Thread):
        await interaction.response.send_message(
            "ここはすでにスレッドの中です。そのまま話しかければ会話が続きます。",
            ephemeral=True,
        )
        return
    if channel is None:
        await interaction.response.send_message(
            "チャンネルで実行してください。", ephemeral=True
        )
        return

    await interaction.response.defer(thinking=True)

    starter = await channel.send(
        f"🧵 {interaction.user.display_name} のスレッド: {question[:80]}"
    )
    try:
        thread = await starter.create_thread(
            name=f"{interaction.user.display_name} のスレッド"[:100]
        )
    except discord.HTTPException as exc:
        print(f"Thread creation failed: {exc}")
        await interaction.followup.send(
            "スレッドの作成に失敗しました。ボットにスレッド作成権限があるか確認してください。"
        )
        return

    active_threads.add(thread.id)

    async with thread.typing():
        try:
            reply = await ask_gemini(thread.id, question)
        except Exception as exc:  # noqa: BLE001
            print(f"Gemini error: {exc}")
            await thread.send("エラーが発生しました。しばらくしてからもう一度お試しください。")
            await interaction.followup.send(f"スレッドを作りました: {thread.mention}")
            return

    chunks = split_message(reply)
    await thread.send(f"{interaction.user.mention} さん\n{chunks[0]}")
    for chunk in chunks[1:]:
        await thread.send(chunk)
    await interaction.followup.send(f"スレッドを作りました: {thread.mention}")


@bot.tree.command(name="search", description="Web検索して最新情報で答えます")
@app_commands.describe(question="調べたいこと")
async def search_command(interaction: discord.Interaction, question: str):
    await interaction.response.defer(thinking=True)
    try:
        reply = await ask_gemini_search(question)
    except Exception as exc:  # noqa: BLE001
        print(f"Search failed, falling back: {exc}")
        try:
            base = await ask_gemini_once(question)
            reply = "（検索できませんでした。通常の回答です）\n" + base
        except Exception as exc2:  # noqa: BLE001
            print(f"Gemini error: {exc2}")
            await interaction.followup.send(
                "エラーが発生しました。しばらくしてからもう一度お試しください。"
            )
            return
    chunks = split_message(reply)
    await interaction.followup.send(chunks[0])
    for chunk in chunks[1:]:
        await interaction.channel.send(chunk)


@bot.event
async def on_message(message: discord.Message):
    # Ignore our own messages and other bots
    if message.author.bot:
        return

    in_thread = isinstance(message.channel, discord.Thread)
    mentioned = bot.user is not None and bot.user in message.mentions

    # Strip the mention itself from the text
    prompt = message.content
    if bot.user is not None:
        for mention in (f"<@{bot.user.id}>", f"<@!{bot.user.id}>"):
            prompt = prompt.replace(mention, "")
    prompt = prompt.strip()

    # --- Inside an active thread: continue the conversation ---
    if in_thread and message.channel.id in active_threads:
        if not prompt:
            return
        async with message.channel.typing():
            try:
                reply = await answer_with_optional_search(
                    prompt, lambda p: ask_gemini(message.channel.id, p)
                )
            except Exception as exc:  # noqa: BLE001
                print(f"Gemini error: {exc}")
                await message.channel.send("エラーが発生しました。しばらくしてからもう一度お試しください。")
                return
        for chunk in split_message(reply):
            await message.channel.send(chunk)
        return

    # Everything below needs a mention
    if not mentioned:
        return

    if not prompt:
        prompt = "こんにちは"

    # --- Mentioned inside a thread: (re)activate it as a conversation ---
    if in_thread:
        active_threads.add(message.channel.id)
        async with message.channel.typing():
            try:
                reply = await answer_with_optional_search(
                    prompt, lambda p: ask_gemini(message.channel.id, p)
                )
            except Exception as exc:  # noqa: BLE001
                print(f"Gemini error: {exc}")
                await message.channel.send("エラーが発生しました。しばらくしてからもう一度お試しください。")
                return
        for chunk in split_message(reply):
            await message.channel.send(chunk)
        return

    # --- In a normal channel: one-shot answer in the same channel ---
    async with message.channel.typing():
        try:
            reply = await answer_with_optional_search(prompt, ask_gemini_once)
        except Exception as exc:  # noqa: BLE001
            print(f"Gemini error: {exc}")
            await message.channel.send("エラーが発生しました。しばらくしてからもう一度お試しください。")
            return
    chunks = split_message(reply)
    await message.reply(chunks[0], mention_author=False)
    for chunk in chunks[1:]:
        await message.channel.send(chunk)


def main():
    if not DISCORD_TOKEN:
        raise RuntimeError("DISCORD_TOKEN environment variable is not set")
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")
    keep_alive()
    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
