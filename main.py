"""
Discord + Gemini AI chat bot.

- In a normal channel: when the bot is mentioned, it answers with a
  one-shot reply in that same channel (no thread, no history).
- To have a continuing conversation, use the /thread slash command:
  it creates a thread and answers inside it. Inside such a thread no
  mention is needed and the bot keeps a per-thread Gemini chat session.
- Mentioning the bot inside any thread (re)activates that thread as a
  conversation thread.
- Search: messages starting with 「調べて」 (or the /search command) are
  answered by a separate search model with Google Search grounding,
  with sources appended. Search has its own small daily quota.
- keep_alive: a small Flask server runs in a background thread so
  Render (and uptime pingers) have an HTTP endpoint to hit.
"""

import asyncio
import os
import re
import threading

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
SEARCH_MODEL = os.environ.get("SEARCH_MODEL", "gemini-2.5-flash-lite")

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


# --- Search (Google Search grounding on a separate, quota-limited model) ---
SEARCH_TRIGGER_RE = re.compile(r"^(調べて|検索して)[\s:：、]*")


def extract_sources(response) -> list[str]:
    """Pull up to 3 source links out of a grounded response."""
    sources: list[str] = []
    seen: set[str] = set()
    try:
        candidates = response.candidates or []
    except Exception:  # noqa: BLE001
        return sources
    for candidate in candidates[:1]:
        metadata = getattr(candidate, "grounding_metadata", None)
        chunks = getattr(metadata, "grounding_chunks", None) or []
        for chunk in chunks:
            web = getattr(chunk, "web", None)
            uri = getattr(web, "uri", None)
            if not uri or uri in seen:
                continue
            seen.add(uri)
            title = getattr(web, "title", None)
            sources.append(f"{title} {uri}" if title else uri)
            if len(sources) >= 3:
                return sources
    return sources


async def ask_gemini_search(prompt: str) -> str:
    """Answer using the search model with Google Search grounding."""
    model = genai.GenerativeModel(SEARCH_MODEL, tools="google_search_retrieval")
    response = await asyncio.to_thread(model.generate_content, prompt)
    try:
        text = (response.text or "").strip()
    except Exception:  # noqa: BLE001
        text = ""
    if not text:
        text = "（応答が空でした。もう一度お試しください）"
    sources = extract_sources(response)
    if sources:
        text += "\n\n出典:\n" + "\n".join(f"- {s}" for s in sources)
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


@bot.tree.command(name="search", description="Web検索して最新情報で答えます（検索は1日20回まで）")
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
