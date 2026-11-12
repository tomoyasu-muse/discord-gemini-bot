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
  the model use Tavily tools (web search, a dedicated news search,
  plus full-page extraction when a snippet is not enough) and answer
  with cited source URLs. Questions that clearly need real-time data
  (weather, FX, stock prices, news…) are auto-detected by a keyword
  regex and answered through a faster pre-fetch route instead: Python
  searches Tavily once, then asks Gemini once with the results.
  If search is unavailable, the bot falls back to a normal answer
  (except on rate limits, where it says so).
- Gemini is called through the google-genai SDK's async client
  (client.aio) and Tavily through AsyncTavilyClient, so no call ever
  blocks the Discord event loop.
- keep_alive: a small Flask server runs in a background thread so
  Render (and uptime pingers) have an HTTP endpoint to hit.
"""

import contextvars
import datetime
import os
import re
import threading
import zoneinfo

import discord
from discord import app_commands
from discord.ext import commands
from flask import Flask
from google import genai
from google.genai import types
from tavily import AsyncTavilyClient

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
# Gemini / Tavily setup (async clients: the event loop is never blocked)
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

# Guild used for instant slash-command sync during testing.
# Global sync also runs, so other guilds get the commands as they propagate.
TEST_GUILD_ID = 1570294742137176134

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
tavily_client = AsyncTavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

RATE_LIMIT_MESSAGE = (
    "⚠️ いま少し混み合っています（Geminiの無料枠の上限に達しました）。"
    "1分ほど待ってからもう一度試してください。"
)
GENERIC_ERROR_MESSAGE = "エラーが発生しました。しばらくしてからもう一度お試しください。"


def is_rate_limit(exc: Exception) -> bool:
    return getattr(exc, "code", None) == 429


def user_error_message(exc: Exception) -> str:
    return RATE_LIMIT_MESSAGE if is_rate_limit(exc) else GENERIC_ERROR_MESSAGE


_WEEKDAYS_JA = ["月", "火", "水", "木", "金", "土", "日"]


def now_str_jst() -> str:
    now = datetime.datetime.now(zoneinfo.ZoneInfo("Asia/Tokyo"))
    return f"{now:%Y年%m月%d日}（{_WEEKDAYS_JA[now.weekday()]}）{now:%H:%M}"


def build_chat_system_instruction() -> str:
    """Light system instruction for normal (non-search) answers."""
    return (
        f"現在の日時は {now_str_jst()}（日本時間）です。"
        "日付や時刻が関係する話題では、この日時を前提に答えてください。"
        "回答は日本語で、簡潔に答えてください。"
    )


# Per-thread chat sessions: thread_id -> AsyncChat
chat_sessions: dict[int, object] = {}


def get_chat_session(thread_id: int):
    if thread_id not in chat_sessions:
        chat_sessions[thread_id] = gemini_client.aio.chats.create(
            model=GEMINI_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=build_chat_system_instruction()
            ),
            history=[],
        )
    return chat_sessions[thread_id]


def _clean_text(response) -> str:
    try:
        text = (response.text or "").strip()
    except Exception:  # noqa: BLE001 - e.g. blocked response without text
        text = ""
    return text if text else "（応答が空でした。もう一度お試しください）"


async def ask_gemini(thread_id: int, prompt: str) -> str:
    """Ask with the per-thread conversation history."""
    chat = get_chat_session(thread_id)
    response = await chat.send_message(prompt)
    return _clean_text(response)


async def ask_gemini_once(prompt: str) -> str:
    """Ask a one-shot question with no conversation history."""
    response = await gemini_client.aio.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=build_chat_system_instruction()
        ),
    )
    return _clean_text(response)


# --- Search (Tavily tools driven by Gemini automatic function calling) ---
SEARCH_TRIGGER_RE = re.compile(r"^(調べて|検索して)[\s:：、]*")

# Auto pre-fetch routing: questions that clearly need real-time data are
# searched by Python directly (one Tavily call + one Gemini call), which
# is faster and cheaper than the function-calling route. The keywords are
# deliberately topic words only — bare time words (今日, いま, 今年…) are
# NOT triggers, so everyday chat is never slowed down and the monthly
# Tavily credit budget is not burned on small talk.
REALTIME_PATTERN = re.compile(
    r"(天気|気温|降水確率|雨が降|雨降る|雪が降|雪降る|台風|地震|津波|"
    r"ドル円|円相場|円安|円高|為替|ビットコイン|仮想通貨|"
    r"株価|日経平均|ダウ平均|ナスダック|決算発表|"
    r"ニュース|速報|"
    r"試合結果|試合の結果|順位表|"
    r"運休|運転見合わせ|渋滞情報|"
    r"ググって|調べて|検索して)"
)
NEWS_PATTERN = re.compile(r"(ニュース|速報|事件|事故|トレンド)")
FINANCE_PATTERN = re.compile(
    r"(ドル円|円相場|円安|円高|為替|ビットコイン|仮想通貨|"
    r"株価|日経平均|ダウ平均|ナスダック|決算発表)"
)
WEATHER_PATTERN = re.compile(r"(天気|気温|降水確率|台風)")
# Weather answers are only as good as the forecast pages behind them;
# generic results mix in past-weather records and unrelated pages.
WEATHER_DOMAINS = ["weather.yahoo.co.jp", "tenki.jp", "weathernews.jp"]


def build_search_system_instruction() -> str:
    """System instruction for search answers, with the current JST
    date/time baked in so the model builds recency-aware queries."""
    return (
        f"現在の日時は {now_str_jst()}（日本時間）です。"
        "あなたはWeb検索ツールを使って最新情報を調べるアシスタントです。"
        "回答する前に、必ず検索ツールで検索してください。"
        "「最新」「今日」「直近」などの質問では、現在日時を基準に検索し、"
        "古い情報を最新として扱わないでください。"
        "ニュースや時事の話題には search_news ツール（ニュース専用・新しい順）を、"
        "それ以外の調べものには search_web ツールを使ってください。"
        "単純な質問では検索は1回で終え、情報が足りない場合だけ追加検索や "
        "extract_page ツール（ページ全文の取得）を使ってください。"
        "記事の公開日を確認し、古い日付の記事は最新の話題として扱わないでください。"
        "事実に関する主張には、根拠となったページへのリンクを "
        "[タイトル](<URL>) の形式で必ず含めてください。"
        "URLはツールの結果に実際に含まれるものだけを使い、存在しないURLを創作してはいけません。"
        "回答は日本語で簡潔にまとめてください。"
    )


def build_prefetch_system_instruction() -> str:
    """System instruction for the pre-fetch route, where Python has
    already searched and the model just answers from the results."""
    return (
        f"現在の日時は {now_str_jst()}（日本時間）です。"
        "ユーザーの質問と、それに対するWeb検索結果が渡されます。"
        "あなたには検索結果が渡されているので、「リアルタイムデータにアクセスできない」"
        "とは言わず、検索結果に基づいて答えてください。"
        "天気・為替・株価などのリアルタイムな情報は、検索結果の内容を優先して答えてください。"
        "もし検索結果に質問と関係ない情報しかない場合は、検索結果を無視して"
        "一般的な知識で普通に答えてください。"
        "天気予報では、過去の実況データと今後の予報を混同しないでください。"
        "天気の質問で場所が指定されていない場合は、特定の都市の予報を断定せず、"
        "全国の概況を簡潔に伝えたうえで、詳しく知りたい場合は場所を教えてほしいと添えてください。"
        "為替・株価では、検索結果に具体的な現在値があれば必ずその数値を答え、"
        "なければ無いと正直に伝えてください。"
        "出典を示すときは [タイトル](<URL>) の形式で、検索結果に含まれるURLだけを使ってください。"
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


def _format_results(results: list[dict]) -> str:
    """Format Tavily results for the model, including publish dates
    when available so it can judge freshness itself."""
    parts = []
    for index, result in enumerate(results, start=1):
        title = result.get("title") or "(no title)"
        url = result.get("url") or ""
        content = (result.get("content") or "")[:600]
        _collect_source(title, url)
        published = result.get("published_date")
        date_line = f"公開日: {published}\n" if published else ""
        parts.append(f"[{index}] {title}\n{date_line}{url}\n{content}")
    return "\n\n".join(parts)


async def search_web(
    query: str,
    search_depth: str = "basic",
    topic: str = "general",
    time_range: str = "",
    include_domains: list | None = None,
) -> str:
    """Search the web for current information.

    Args:
        query: What to search for.
        search_depth: "basic" (default, 1 credit) or "advanced"
            (2 credits, deeper search; use only when basic detail is
            unlikely to be enough).
        topic: "general" (default), "news" (for recent news) or
            "finance" (for market prices and financial data).
        time_range: Optional recency filter: "day", "week", "month"
            or "year".
        include_domains: Optional list of domains to restrict the
            search to.

    Returns:
        Search results as text: title, published date, URL and a
        snippet per result.
    """
    if tavily_client is None:
        raise RuntimeError("TAVILY_API_KEY is not set")
    if search_depth not in ("basic", "advanced"):
        search_depth = "basic"
    if topic not in ("general", "news", "finance"):
        topic = "general"
    if time_range not in ("day", "week", "month", "year"):
        time_range = None
    data = await tavily_client.search(
        query=query,
        search_depth=search_depth,
        topic=topic,
        time_range=time_range,
        max_results=5,
        include_domains=include_domains or None,
    )
    results = data.get("results", [])
    if not results:
        return "検索結果は見つかりませんでした。クエリを変えて再検索できます。"
    return _format_results(results)


async def search_news(query: str, days: int = 3) -> str:
    """Search recent news (news sources only, newest first).

    Args:
        query: What to search for.
        days: Only include articles from the last this many days
            (default 3). Widen it if nothing is found.

    Returns:
        News results as text: title, published date, URL and a
        snippet per result.
    """
    if tavily_client is None:
        raise RuntimeError("TAVILY_API_KEY is not set")
    try:
        days = int(days)
    except (TypeError, ValueError):
        days = 3
    days = max(1, min(days, 31))
    data = await tavily_client.search(
        query=query,
        search_depth="basic",
        topic="news",
        days=days,
        max_results=5,
    )
    results = data.get("results", [])
    if not results:
        return "該当する最新ニュースが見つかりませんでした。daysを広げて再検索できます。"
    return _format_results(results)


async def extract_page(url: str) -> str:
    """Read the full text of one web page (for example from search_web).

    Use this when a search snippet is not enough to answer accurately.

    Args:
        url: The page URL to read.

    Returns:
        The page content as Markdown text (truncated if very long).
    """
    if tavily_client is None:
        raise RuntimeError("TAVILY_API_KEY is not set")
    data = await tavily_client.extract(
        urls=[url], extract_depth="basic", format="markdown"
    )
    results = data.get("results", [])
    if not results:
        return "ページの内容を取得できませんでした。"
    first = results[0]
    _collect_source(first.get("title") or url, url)
    text = first.get("raw_content") or ""
    return text[:4000] if text else "ページの内容を取得できませんでした。"


async def ask_gemini_search(prompt: str) -> str:
    """Answer with web search: the model drives the Tavily tools itself
    (automatic function calling), then writes an answer citing sources."""
    if tavily_client is None:
        raise RuntimeError("TAVILY_API_KEY is not set")
    sources: list = []
    token = _current_sources.set(sources)
    try:
        # AFC runs through AsyncChat: the SDK explicitly recommends chat
        # send_message over models.generate_content for async AFC.
        chat = gemini_client.aio.chats.create(
            model=GEMINI_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=build_search_system_instruction(),
                tools=[search_web, search_news, extract_page],
                max_output_tokens=2048,
            ),
            history=[],
        )
        response = await chat.send_message(prompt)
    finally:
        _current_sources.reset(token)
    text = _clean_text(response)
    # Append the collected real sources only when the answer itself
    # contains no URL; if the model already cited its sources inline,
    # a second, possibly different list would just confuse readers.
    if sources and "http" not in text:
        text += "\n\n出典:\n" + "\n".join(
            f"- [{title}](<{url}>)" for title, url in sources[:5]
        )
    return text


async def ask_gemini_prefetch(prompt: str) -> str:
    """Pre-fetch route for real-time questions: Python searches Tavily
    once, then asks Gemini once with the results attached. No function
    calling, so this is the fastest search path (1 Tavily credit +
    1 Gemini call)."""
    if tavily_client is None:
        raise RuntimeError("TAVILY_API_KEY is not set")
    sources: list = []
    token = _current_sources.set(sources)
    try:
        today = datetime.datetime.now(
            zoneinfo.ZoneInfo("Asia/Tokyo")
        ).strftime("%Y年%m月%d日")
        query = f"{prompt}（{today} 現在）"
        if NEWS_PATTERN.search(prompt):
            results_text = await search_news(prompt, days=3)
        elif FINANCE_PATTERN.search(prompt):
            # Tavily's finance topic is tuned for financial queries and
            # can return zero results for natural-language questions
            # (observed live: 「今のドル円は？」 returned nothing and the
            # model answered as if it had no data). Retry with general
            # search before giving up.
            results_text = await search_web(prompt, topic="finance")
            if results_text.startswith("検索結果は見つかりませんでした"):
                results_text = await search_web(query)
        elif WEATHER_PATTERN.search(prompt):
            # Restrict to forecast sites so past-weather records and
            # unrelated pages do not pollute the answer.
            results_text = await search_web(query, include_domains=WEATHER_DOMAINS)
        else:
            results_text = await search_web(query)
    finally:
        _current_sources.reset(token)
    contents = (
        "以下のWeb検索結果をもとに、ユーザーの質問に回答してください。\n\n"
        f"【ユーザーの質問】\n{prompt}\n\n"
        f"【Web検索結果】\n{results_text}"
    )
    response = await gemini_client.aio.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=build_prefetch_system_instruction(),
            max_output_tokens=2048,
        ),
    )
    text = _clean_text(response)
    if sources and "http" not in text:
        text += "\n\n出典:\n" + "\n".join(
            f"- [{title}](<{url}>)" for title, url in sources[:5]
        )
    return text


async def answer_with_optional_search(prompt: str, normal_ask):
    """Three-way routing (decided by regex in well under a millisecond,
    so normal chat pays no latency):

    - Starts with 「調べて」 -> the agentic function-calling search
      (deepest research; falls back to a normal answer on failure).
    - Clearly real-time topic (REALTIME_PATTERN: weather, FX, stocks,
      news…) -> the pre-fetch search (fast: 1 Tavily + 1 Gemini call;
      silently falls back to a normal answer on failure).
    - Anything else -> the normal answer as-is.

    normal_ask is an async callable taking the prompt. A Gemini rate
    limit (429) is re-raised on every route, because a fallback call
    would hit the same limit.
    """
    match = SEARCH_TRIGGER_RE.match(prompt)
    if match:
        query = prompt[match.end():].strip() or prompt
        try:
            return await ask_gemini_search(query)
        except Exception as exc:  # noqa: BLE001
            if is_rate_limit(exc):
                raise
            print(f"Search failed, falling back: {exc}")
            base = await normal_ask(prompt)
            return "（検索できませんでした。通常の回答です）\n" + base
    if REALTIME_PATTERN.search(prompt):
        try:
            return await ask_gemini_prefetch(prompt)
        except Exception as exc:  # noqa: BLE001
            if is_rate_limit(exc):
                raise
            print(f"Prefetch failed, falling back: {exc}")
            return await normal_ask(prompt)
    return await normal_ask(prompt)


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
            await thread.send(user_error_message(exc))
            await interaction.followup.send(f"スレッドを作りました: {thread.mention}")
            return

    chunks = split_message(reply)
    await thread.send(f"{interaction.user.mention} さん\n{chunks[0]}", suppress_embeds=True)
    for chunk in chunks[1:]:
        await thread.send(chunk, suppress_embeds=True)
    await interaction.followup.send(f"スレッドを作りました: {thread.mention}")


@bot.tree.command(name="search", description="Web検索して最新情報で答えます")
@app_commands.describe(question="調べたいこと")
async def search_command(interaction: discord.Interaction, question: str):
    await interaction.response.defer(thinking=True)
    try:
        reply = await ask_gemini_search(question)
    except Exception as exc:  # noqa: BLE001
        if is_rate_limit(exc):
            print(f"Search rate limited: {exc}")
            await interaction.followup.send(RATE_LIMIT_MESSAGE)
            return
        print(f"Search failed, falling back: {exc}")
        try:
            base = await ask_gemini_once(question)
            reply = "（検索できませんでした。通常の回答です）\n" + base
        except Exception as exc2:  # noqa: BLE001
            print(f"Gemini error: {exc2}")
            await interaction.followup.send(user_error_message(exc2))
            return
    chunks = split_message(reply)
    await interaction.followup.send(chunks[0], suppress_embeds=True)
    for chunk in chunks[1:]:
        await interaction.channel.send(chunk, suppress_embeds=True)


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
                await message.channel.send(user_error_message(exc))
                return
        for chunk in split_message(reply):
            await message.channel.send(chunk, suppress_embeds=True)
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
                await message.channel.send(user_error_message(exc))
                return
        for chunk in split_message(reply):
            await message.channel.send(chunk, suppress_embeds=True)
        return

    # --- In a normal channel: one-shot answer in the same channel ---
    async with message.channel.typing():
        try:
            reply = await answer_with_optional_search(prompt, ask_gemini_once)
        except Exception as exc:  # noqa: BLE001
            print(f"Gemini error: {exc}")
            await message.channel.send(user_error_message(exc))
            return
    chunks = split_message(reply)
    await message.reply(chunks[0], mention_author=False, suppress_embeds=True)
    for chunk in chunks[1:]:
        await message.channel.send(chunk, suppress_embeds=True)


def main():
    if not DISCORD_TOKEN:
        raise RuntimeError("DISCORD_TOKEN environment variable is not set")
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")
    keep_alive()
    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
