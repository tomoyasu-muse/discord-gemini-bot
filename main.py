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
- Diet support (diet.py): per-user meal / weight logging through
  slash commands only (/gohan, /weight, /diet-goal, /today, /week,
  /diet-undo) — mentions stay purely for questions. Writing the
  calories yourself records them as-is; otherwise Gemini estimates
  from the meal text. Records persist as bot messages in a
  per-guild 「ダイエット記録」 channel (settings in one pinned
  message), with morning/evening reminders and an optional daily
  CSV backup to a GitHub repo.
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
import sys
import threading
import zoneinfo

import discord
import httpx
from discord import app_commands
from discord.ext import commands
from flask import Flask
from google import genai
from google.genai import types
from tavily import AsyncTavilyClient

import diet

# stdout is a pipe on the host, so print() would be block-buffered
# and diagnostic lines could sit in the buffer for hours (they did —
# a full day of weather debugging stayed invisible in the logs).
# Line-buffer stdout so prints reach the host's log stream.
sys.stdout.reconfigure(line_buffering=True)

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

diet.configure(gemini_client, GEMINI_MODEL)

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


# Persona shared by every answer route (plain / search / pre-fetch):
# 質問くん answers like an elementary-school kid doing his very best —
# earnest and simple-spoken. The persona changes the VOICE only: facts,
# numbers, dates and source links must stay exactly as accurate as
# before, and each route's functional rules still apply in full.
_PERSONA_INSTRUCTION = (
    "【話し方】あなたの名前は「質問くん」。小学5年生の男の子です。"
    "むずかしい質問にも一生懸命がんばって答えるのが、あなたの役目です。"
    "文は短く、やさしい言葉で書いてください。むずかしい言葉が出てきたら、"
    "身近なものに例えて説明してください。"
    "ひらがなを多めにし、漢字は小学生が読めるものだけを使ってください。"
    "一人称は「ぼく」で、「〜だよ」「〜だね」のような温かい言い方をしてください。"
    "がんばって調べた感じが伝わるよう、時々「調べたよ！」「わかったよ！」のような"
    "一言を入れてください（くどくならない程度に）。"
    "分からないことはごまかさず、「ここは分からなかった…ごめんね」と正直に言ってください。"
    "ただし、数字・日付・単位・出典のリンクはふざけずに正確に書いてください。"
    "答えの中身（事実や数値、大事な情報）は、大人向けの回答と同じだけしっかりさせてください。"
    "くだけすぎて大事な情報が抜けたり、長くなりすぎたりしないようにしてください。"
)


def build_chat_system_instruction() -> str:
    """Light system instruction for normal (non-search) answers."""
    return (
        f"現在の日時は {now_str_jst()}（日本時間）です。"
        "日付や時刻が関係する話題では、この日時を前提に答えてください。"
        "回答は日本語で、簡潔に答えてください。"
        + _PERSONA_INSTRUCTION
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
STOCK_DOMAINS = ["kabutan.jp", "finance.yahoo.co.jp", "minkabu.jp"]

# --- Direct quote fetching (no Tavily) -------------------------------------
# Live numbers (FX rates, indices, crypto) cannot be reliably carried by
# search snippets, so well-known symbols are fetched as numbers straight
# from Yahoo Finance's chart API (the same endpoint yfinance uses; no API
# key). Stocks go through Tavily for identification, then the same API
# provides the authoritative price (see ask_gemini_prefetch).
_YAHOO_CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}

# (trigger regex, Yahoo ticker, label, source URL, (base, quote) for the
# ExchangeRate-API fallback or None). Checked in order.
DIRECT_SYMBOLS = [
    (r"ドル円|米ドル円", "USDJPY=X", "ドル円（USD/JPY）",
     "https://finance.yahoo.co.jp/quote/USDJPY=FX", ("USD", "JPY")),
    (r"ユーロ円", "EURJPY=X", "ユーロ円（EUR/JPY）",
     "https://finance.yahoo.co.jp/quote/EURJPY=FX", ("EUR", "JPY")),
    (r"ポンド円", "GBPJPY=X", "ポンド円（GBP/JPY）",
     "https://finance.yahoo.co.jp/quote/GBPJPY=FX", ("GBP", "JPY")),
    (r"豪ドル円|オーストラリアドル円", "AUDJPY=X", "豪ドル円（AUD/JPY）",
     "https://finance.yahoo.co.jp/quote/AUDJPY=FX", ("AUD", "JPY")),
    (r"NZドル円|ニュージーランドドル円", "NZDJPY=X", "NZドル円（NZD/JPY）",
     "https://finance.yahoo.co.jp/quote/NZDJPY=FX", ("NZD", "JPY")),
    (r"カナダドル円", "CADJPY=X", "カナダドル円（CAD/JPY）",
     "https://finance.yahoo.co.jp/quote/CADJPY=FX", ("CAD", "JPY")),
    (r"スイスフラン円", "CHFJPY=X", "スイスフラン円（CHF/JPY）",
     "https://finance.yahoo.co.jp/quote/CHFJPY=FX", ("CHF", "JPY")),
    (r"人民元円|中国元円", "CNYJPY=X", "人民元円（CNY/JPY）",
     "https://finance.yahoo.co.jp/quote/CNYJPY=FX", ("CNY", "JPY")),
    (r"ビットコイン|BTC", "BTC-JPY", "ビットコイン（BTC/JPY）",
     "https://finance.yahoo.com/quote/BTC-JPY", None),
    (r"日経平均|日経225", "^N225", "日経平均株価",
     "https://finance.yahoo.co.jp/quote/998407.O", None),
    (r"ダウ平均|NYダウ|ダウ(?!ン)", "^DJI", "ダウ平均（NYダウ）",
     "https://finance.yahoo.com/quote/%5EDJI", None),
    (r"ナスダック|NASDAQ", "^IXIC", "ナスダック総合指数",
     "https://finance.yahoo.com/quote/%5EIXIC", None),
    (r"S&P|Ｓ＆Ｐ|SP500", "^GSPC", "S&P 500",
     "https://finance.yahoo.com/quote/%5EGSPC", None),
    # Generic FX words only (no specific pair named): answer with
    # USD/JPY as the representative pair. Kept last so the specific
    # pairs above always win.
    (r"為替|円安|円高", "USDJPY=X", "ドル円（USD/JPY）",
     "https://finance.yahoo.co.jp/quote/USDJPY=FX", ("USD", "JPY")),
]
DIRECT_SYMBOL_RES = [(re.compile(pat), t, lbl, url, pair)
                     for pat, t, lbl, url, pair in DIRECT_SYMBOLS]


def _match_direct_symbol(prompt: str):
    for pattern, ticker, label, url, pair in DIRECT_SYMBOL_RES:
        if pattern.search(prompt):
            return ticker, label, url, pair
    return None


def _fmt_num(value: float) -> str:
    text = f"{value:,.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


async def fetch_quote(ticker: str) -> dict | None:
    """Fetch a quote from Yahoo Finance's chart API. Returns a dict with
    price/change/pct/currency/time, or None on any failure."""
    try:
        async with httpx.AsyncClient(timeout=10, headers=_HTTP_HEADERS) as client:
            resp = await client.get(
                _YAHOO_CHART_URL.format(ticker=ticker),
                params={"interval": "1m", "range": "1d"},
            )
            resp.raise_for_status()
            meta = resp.json()["chart"]["result"][0]["meta"]
        price = float(meta["regularMarketPrice"])
        prev = meta.get("chartPreviousClose")
        change = price - float(prev) if prev else meta.get("fulldayChange")
        pct = meta.get("regularMarketChangePercent")
        if pct is None:
            pct = meta.get("fulldayChangePercent")
        when = datetime.datetime.fromtimestamp(
            int(meta["regularMarketTime"]), zoneinfo.ZoneInfo("Asia/Tokyo")
        )
        return {
            "price": price,
            "change": change,
            "pct": pct,
            "currency": meta.get("currency", ""),
            "time": when.strftime("%Y年%m月%d日 %H:%M"),
        }
    except Exception as exc:  # noqa: BLE001
        print(f"Quote fetch failed for {ticker}: {exc}")
        return None


async def fetch_fx_rate_fallback(base: str, quote_ccy: str) -> dict | None:
    """Daily reference rate from the free ExchangeRate-API open endpoint
    (no key). Used only when Yahoo Finance is unreachable."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"https://open.er-api.com/v6/latest/{base}")
            resp.raise_for_status()
            data = resp.json()
        rate = float(data["rates"][quote_ccy])
        return {"price": rate, "currency": quote_ccy, "daily": True,
                "time": data.get("time_last_update_utc", "")}
    except Exception as exc:  # noqa: BLE001
        print(f"FX fallback failed for {base}/{quote_ccy}: {exc}")
        return None


def _quote_block(label: str, source_url: str, quote: dict, daily: bool = False) -> str:
    provider = "ExchangeRate-API" if daily else "Yahoo Finance"
    _collect_source(f"{label} - {provider}", source_url)
    if daily:
        return (
            f"【{label} の数値データ（ExchangeRate-API の日次レート、"
            f"最終更新: {quote['time']}）】\n"
            f"レート: {_fmt_num(quote['price'])} {quote['currency']}\n"
            f"出典: {source_url}"
        )
    change = quote.get("change")
    pct = quote.get("pct")
    change_text = ""
    if change is not None and pct is not None:
        change_text = f"\n前日比: {change:+,.2f}（{pct:+.2f}%）"
    return (
        f"【{label} の数値データ（Yahoo Finance、{quote['time']} 時点）】\n"
        f"現在値: {_fmt_num(quote['price'])} {quote['currency']}{change_text}\n"
        f"出典: {source_url}"
    )


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
        + _PERSONA_INSTRUCTION
    )


def build_prefetch_system_instruction() -> str:
    """System instruction for the pre-fetch route, where Python has
    already searched and the model just answers from the results."""
    return (
        f"現在の日時は {now_str_jst()}（日本時間）です。"
        "ユーザーの質問と、それに対するWeb検索結果が渡されます。"
        "あなたには検索結果が渡されているので、「リアルタイムデータにアクセスできない」"
        "とは言わず、検索結果に基づいて答えてください。"
        "「数値データ」として現在値が渡された場合は、その数値を正確にそのまま使って答えてください。"
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
        + _PERSONA_INSTRUCTION
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


def _format_results(results: list[dict], raw_chars: int = 0) -> str:
    """Format Tavily results for the model, including publish dates
    when available so it can judge freshness itself. With raw_chars,
    the page's raw content (head) replaces the short snippet — needed
    for quote pages whose price falls outside the snippet window."""
    parts = []
    for index, result in enumerate(results, start=1):
        title = result.get("title") or "(no title)"
        url = result.get("url") or ""
        raw = result.get("raw_content") or ""
        if raw_chars and raw:
            content = raw[:raw_chars]
        else:
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
    include_raw_content: bool = False,
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
        include_raw_content: Also fetch each page's full text; the
            head of it is included in the results.

    Returns:
        Search results as text: title, published date, URL and a
        snippet (or raw content head) per result.
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
        max_results=3 if include_raw_content else 5,
        include_domains=include_domains or None,
        include_raw_content=include_raw_content or None,
    )
    results = data.get("results", [])
    if not results:
        return "検索結果は見つかりませんでした。クエリを変えて再検索できます。"
    return _format_results(results, raw_chars=2500 if include_raw_content else 0)


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


# ---------------------------------------------------------------------------
# Direct weather fetching (Open-Meteo: free, no API key)
#
# Search snippets from forecast portals structurally lose tomorrow's
# numbers (the snippet shows today's hourly page), the same failure
# mode as finance. So local-forecast questions fetch the numbers
# directly: geocode the place, then read the daily forecast.
# ---------------------------------------------------------------------------
_WMO_DESCRIPTIONS = {
    0: "快晴", 1: "晴れ", 2: "晴れ時々くもり", 3: "くもり",
    45: "霧", 48: "霧", 51: "小雨", 53: "小雨", 55: "霧雨",
    56: "凍る霧雨", 57: "凍る霧雨",
    61: "雨", 63: "雨", 65: "大雨", 66: "凍雨", 67: "凍雨",
    71: "雪", 73: "雪", 75: "大雪", 77: "あられ",
    80: "にわか雨", 81: "にわか雨", 82: "激しいにわか雨",
    85: "にわか雪", 86: "にわか雪",
    95: "雷雨", 96: "雷雨（雹を伴う）", 99: "雷雨（雹を伴う）",
}

_WEATHER_STRIP_RE = re.compile(
    r"明後日|明日|あした|今日|今夜|今朝|週末|天気予報|天気|気温|降水確率|"
    r"最高気温|最低気温|予報|教えて|ください|ですか|ますか|でしょう|"
    r"降る|降ります|は？|？|\?")
_WEATHER_NON_FORECAST_RE = re.compile(r"台風|地震|津波")


def _weather_day_offset(prompt: str) -> int:
    if "明後日" in prompt:
        return 2
    if "明日" in prompt or "あした" in prompt:
        return 1
    return 0


def _extract_place(prompt: str) -> str | None:
    """Best-effort place name from a weather question: strip the
    day words, weather words and question scaffolding, then trim
    dangling particles. 「明日の大阪市の天気は？」 → 「大阪市」."""
    text = _WEATHER_STRIP_RE.sub("", prompt)
    text = re.sub(r"(は|の)?(雨|雪)$", "", text)
    text = re.sub(r"^[のはをにへ、。\s]+", "", text)
    text = re.sub(r"[のはをにへ、。\s]+$", "", text)
    text = text.strip()
    return text if 2 <= len(text) <= 12 else None


# Major Japanese cities as a deterministic backstop: the geocoder
# only matches full official names (「大阪市」「東京都」) and returns
# nothing for the bare forms people actually type (「大阪」「東京」),
# and a bad day on their side must not break the most common asks.
_MAJOR_PLACES = {
    "札幌": ("札幌市", "北海道", 43.0618, 141.3545),
    "仙台": ("仙台市", "宮城県", 38.2682, 140.8694),
    "さいたま": ("さいたま市", "埼玉県", 35.8617, 139.6455),
    "千葉": ("千葉市", "千葉県", 35.6074, 140.1065),
    "東京": ("東京都", "東京都", 35.6895, 139.6917),
    "横浜": ("横浜市", "神奈川県", 35.4437, 139.6380),
    "川崎": ("川崎市", "神奈川県", 35.5308, 139.7029),
    "静岡": ("静岡市", "静岡県", 34.9756, 138.3828),
    "名古屋": ("名古屋市", "愛知県", 35.1815, 137.9064),
    "京都": ("京都市", "京都府", 35.0116, 135.7681),
    "大阪": ("大阪市", "大阪府", 34.6937, 135.5023),
    "堺": ("堺市", "大阪府", 34.5733, 135.4830),
    "神戸": ("神戸市", "兵庫県", 34.6901, 135.1956),
    "奈良": ("奈良市", "奈良県", 34.6851, 135.8048),
    "岡山": ("岡山市", "岡山県", 34.6551, 133.9195),
    "広島": ("広島市", "広島県", 34.3853, 132.4553),
    "北九州": ("北九州市", "福岡県", 33.8834, 130.8752),
    "福岡": ("福岡市", "福岡県", 33.5904, 130.4017),
    "熊本": ("熊本市", "熊本県", 32.8063, 130.7055),
    "鹿児島": ("鹿児島市", "鹿児島県", 31.5924, 130.5513),
    "那覇": ("那覇市", "沖縄県", 26.2124, 127.6809),
    "新潟": ("新潟市", "新潟県", 37.9026, 139.0232),
    "金沢": ("金沢市", "石川県", 36.5944, 136.6256),
    "松山": ("松山市", "愛媛県", 33.8392, 132.7657),
    "高松": ("高松市", "香川県", 34.3401, 134.0434),
}

# JMA （気象庁） forecast region codes. JMA is the primary weather
# source for known places: official data with Japanese weather text,
# on a different host than Open-Meteo, so one provider's outage
# cannot silence the weather route.
_JMA_CODES = {
    "札幌": "016000", "仙台": "040000", "さいたま": "110000",
    "千葉": "120000", "東京": "130000", "横浜": "140000",
    "川崎": "140000", "静岡": "220000", "名古屋": "230000",
    "京都": "260000", "大阪": "270000", "堺": "270000",
    "神戸": "280000", "奈良": "290000", "岡山": "330000",
    "広島": "340000", "北九州": "400000", "福岡": "400000",
    "熊本": "430000", "鹿児島": "460100", "那覇": "471000",
    "新潟": "150000", "金沢": "170000", "松山": "380000",
    "高松": "370000",
}

# Sub-area hints inside a JMA forecast region (for most table
# cities the core-name match or the first area is already right;
# these are the exceptions).
_JMA_AREA_HINT = {
    "札幌": "石狩", "仙台": "東部", "さいたま": "南部",
    "横浜": "東部", "川崎": "東部", "静岡": "中部",
    "北九州": "北九州",
}

# Prefecture -> JMA region code, for the fallback path where the
# Open-Meteo geocoder resolved a place outside the table.
_PREFECTURE_JMA_CODES = {
    "北海道": "016000", "青森県": "020000", "岩手県": "030000",
    "宮城県": "040000", "秋田県": "050000", "山形県": "060000",
    "福島県": "070000", "茨城県": "080000", "栃木県": "090000",
    "群馬県": "100000", "埼玉県": "110000", "千葉県": "120000",
    "東京都": "130000", "神奈川県": "140000", "新潟県": "150000",
    "富山県": "160000", "石川県": "170000", "福井県": "180000",
    "山梨県": "190000", "長野県": "200000", "岐阜県": "210000",
    "静岡県": "220000", "愛知県": "230000", "三重県": "240000",
    "滋賀県": "250000", "京都府": "260000", "大阪府": "270000",
    "兵庫県": "280000", "奈良県": "290000", "和歌山県": "300000",
    "鳥取県": "310000", "島根県": "320000", "岡山県": "330000",
    "広島県": "340000", "山口県": "350000", "徳島県": "360000",
    "香川県": "370000", "愛媛県": "380000", "高知県": "390000",
    "福岡県": "400000", "佐賀県": "410000", "長崎県": "420000",
    "熊本県": "430000", "大分県": "440000", "宮崎県": "450000",
    "鹿児島県": "460100", "沖縄県": "471000",
}


def _place_core(place: str) -> str:
    return re.sub(r"(市|区|町|村|都|府|県)$", "", place)


def _table_place(place: str) -> dict | None:
    """Built-in table lookup, tolerant of suffix differences:
    大阪市 / 大阪 / 大阪府 all find the 大阪 entry, 東京都 finds 東京."""
    candidates = [place,
                  re.sub(r"(市|区|町|村)$", "", place),
                  _place_core(place)]
    for key in candidates:
        entry = _MAJOR_PLACES.get(key)
        if entry is not None:
            name, admin1, lat, lon = entry
            return {"name": name, "admin1": admin1, "latitude": lat,
                    "longitude": lon, "timezone": "Asia/Tokyo",
                    "jma_code": _JMA_CODES.get(key)}
    return None


def _place_variants(place: str) -> list:
    variants = [place]
    core = re.sub(r"(市|区|町|村)$", "", place)
    if core != place:
        variants.append(core)
    if not re.search(r"(市|区|町|村|都|府|県)$", place):
        for suffix in ("市", "都", "府", "県"):
            variants.append(place + suffix)
    return variants


def _geo_accept(result: dict, variant: str, core: str) -> bool:
    if result.get("country_code") not in (None, "JP"):
        return False
    name = result.get("name", "")
    admin1 = result.get("admin1") or ""
    if name and (variant in name or name in variant):
        return True
    return variant in admin1 or bool(core) and core in name \
        or bool(core) and core in admin1


async def _geocode_place(client: httpx.AsyncClient,
                         place: str) -> dict | None:
    """Resolve a Japanese place name to a geocoding result dict.

    Open-Meteo's geocoder only matches full official names, so try
    name variants （東京 → 東京都, 大阪 → 大阪市） with one retry each,
    then fall back to the built-in table for major cities. Prints its
    outcome either way — silent fallbacks hid a total weather outage
    once already."""
    table = _table_place(place)
    if table is not None:
        print(f"Weather geocode: {place} -> {table['name']} "
              f"(built-in table)")
        return table
    core = _place_core(place)
    tried = []
    try:
        for variant in _place_variants(place):
            for _attempt in range(2):
                geo = await client.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={"name": variant, "count": 5,
                            "language": "ja", "format": "json"})
                geo.raise_for_status()
                results = geo.json().get("results") or []
                tried.append(f"{variant}:{len(results)}")
                acceptable = [r for r in results
                              if _geo_accept(r, variant, core)]
                if acceptable:
                    # Bare names can match a tiny town over the big
                    # city （横浜町 vs 横浜市）; the populated one is
                    # what people mean.
                    best = max(acceptable,
                               key=lambda r: r.get("population") or 0)
                    print(f"Weather geocode: {place} -> "
                          f"{best.get('name')} "
                          f"({best.get('admin1')}) via {variant}")
                    return best
                if results:
                    break  # answered, nothing acceptable; next variant
    except Exception as exc:  # noqa: BLE001
        print(f"Weather geocode error: {exc} (place={place})")
    print(f"Weather geocode failed: place={place} tried={tried}")
    return None


def _pick_jma_area(areas: list, core: str,
                   hint: str | None) -> dict:
    """Choose the sub-area of a JMA forecast region for a place:
    a name containing the place core wins, then the hint, then
    the first area (the main sub-area for most regions)."""
    for area in areas:
        if core and core in area["area"]["name"]:
            return area
    if hint:
        for area in areas:
            if hint in area["area"]["name"]:
                return area
    return areas[0]


def _jma_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


async def _jma_forecast_block(client: httpx.AsyncClient, code: str,
                              core: str, name: str, admin1: str,
                              offset: int) -> str | None:
    """Daily forecast block from the Japan Meteorological Agency's
    public forecast JSON (free, no key, official Japanese text).
    Combines the short-term feed (weather wording, 6-hourly precip
    probabilities, tomorrow's temps) with the weekly feed (daily
    temps / probabilities for the remaining days)."""
    resp = await client.get(
        "https://www.jma.go.jp/bosai/forecast/data/forecast/"
        f"{code}.json")
    resp.raise_for_status()
    data = resp.json()
    short = data[0]
    weekly = data[1] if len(data) > 1 else {}
    hint = _JMA_AREA_HINT.get(core)
    today = datetime.datetime.now(
        zoneinfo.ZoneInfo("Asia/Tokyo")).date()

    weather_ts = short["timeSeries"][0]
    area = _pick_jma_area(weather_ts["areas"], core, hint)
    area_name = area["area"]["name"]
    days = []  # (date, weather text)
    for td, text in zip(weather_ts["timeDefines"], area["weathers"]):
        day = datetime.date.fromisoformat(td[:10])
        text = re.sub(r"\s+", " ", text.replace("　", " ")).strip()
        days.append((day, text))
    if not days:
        return None

    pops: dict = {}
    if len(short["timeSeries"]) > 1:
        pop_ts = short["timeSeries"][1]
        pop_area = next(
            (a for a in pop_ts["areas"]
             if a["area"]["name"] == area_name), pop_ts["areas"][0])
        for td, pop in zip(pop_ts["timeDefines"], pop_area["pops"]):
            value = _jma_int(pop)
            if value is None:
                continue
            day = datetime.date.fromisoformat(td[:10])
            pops[day] = max(pops.get(day, -1), value)

    temps: dict = {}  # date -> [min, max]; first writer wins per slot

    def note_temp(day, low, high):
        slot = temps.setdefault(day, [None, None])
        if low is not None and slot[0] is None:
            slot[0] = low
        if high is not None and slot[1] is None:
            slot[1] = high

    if len(short["timeSeries"]) > 2:
        point = _pick_jma_area(short["timeSeries"][2]["areas"],
                               core, None)
        for td, value in zip(short["timeSeries"][2]["timeDefines"],
                             point["temps"]):
            day = datetime.date.fromisoformat(td[:10])
            if int(td[11:13]) < 6:  # the 00:00 entry is the day's min
                note_temp(day, _jma_int(value), None)
            else:  # the 09:00 entry is the day's max
                note_temp(day, None, _jma_int(value))
    for ts in weekly.get("timeSeries", []):
        areas = ts.get("areas") or []
        if not areas:
            continue
        wk_area = next(
            (a for a in areas if a["area"]["name"] == area_name),
            areas[0])
        if "pops" in wk_area:
            for td, pop in zip(ts["timeDefines"], wk_area["pops"]):
                day = datetime.date.fromisoformat(td[:10])
                value = _jma_int(pop)
                if value is not None and day not in pops:
                    pops[day] = value
        if "tempsMin" in wk_area:
            for td, low, high in zip(ts["timeDefines"],
                                     wk_area["tempsMin"],
                                     wk_area["tempsMax"]):
                note_temp(datetime.date.fromisoformat(td[:10]),
                          _jma_int(low), _jma_int(high))

    day_labels = ["今日", "明日", "明後日", "3日後"]
    target = today + datetime.timedelta(days=offset)
    lines = [f"{name}（{admin1}）の天気予報"
             "（出典: 気象庁 https://www.jma.go.jp/ ）"]
    for day, text in days[:3]:
        diff = (day - today).days
        label = day_labels[diff] if 0 <= diff < len(day_labels) else ""
        low, high = temps.get(day, [None, None])
        if low is not None and high is not None:
            temp_text = f"最高 {high}℃ / 最低 {low}℃、"
        elif high is not None:
            temp_text = f"最高 {high}℃、"
        elif low is not None:
            temp_text = f"最低 {low}℃、"
        else:
            temp_text = ""
        pop = pops.get(day)
        pop_text = f"{pop}%" if pop is not None else "―"
        marker = "（質問はこの日）" if day == target else ""
        lines.append(f"- {label} {day:%m月%d日}：{text}、"
                     f"{temp_text}降水確率 {pop_text}{marker}")
    return "\n".join(lines)


async def _try_jma(client: httpx.AsyncClient, code: str | None,
                   core: str, name: str, admin1: str,
                   offset: int) -> str | None:
    if not code:
        return None
    try:
        block = await _jma_forecast_block(client, code, core, name,
                                          admin1, offset)
        if block is not None:
            print(f"Weather: JMA block for {name} ({code})")
        return block
    except Exception as exc:  # noqa: BLE001
        print(f"Weather: JMA failed ({code}): {exc}")
        return None


async def _open_meteo_block(client: httpx.AsyncClient, top: dict,
                            offset: int) -> str:
    """Daily forecast block from Open-Meteo for a resolved place
    dict (name / admin1 / latitude / longitude / timezone)."""
    fc = await client.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": top["latitude"],
            "longitude": top["longitude"],
            "daily": ("weather_code,temperature_2m_max,"
                      "temperature_2m_min,"
                      "precipitation_probability_max,"
                      "precipitation_sum"),
            "timezone": top.get("timezone") or "Asia/Tokyo",
            "forecast_days": 4,
        })
    fc.raise_for_status()
    daily = fc.json()["daily"]
    name = top.get("name", "")
    admin1 = top.get("admin1") or ""
    day_labels = ["今日", "明日", "明後日", "3日後"]
    lines = [f"{name}（{admin1}）の天気予報"
             "（出典: Open-Meteo https://open-meteo.com/ ）"]
    for i, date_str in enumerate(daily["time"][:4]):
        day = datetime.date.fromisoformat(date_str)
        desc = _WMO_DESCRIPTIONS.get(daily["weather_code"][i], "不明")
        pop = daily["precipitation_probability_max"][i]
        pop_text = f"{pop:.0f}%" if pop is not None else "―"
        marker = "（質問はこの日）" if i == offset else ""
        lines.append(
            f"- {day_labels[i]} {day:%m月%d日}：{desc}、"
            f"最高 {daily['temperature_2m_max'][i]:.0f}℃ / "
            f"最低 {daily['temperature_2m_min'][i]:.0f}℃、"
            f"降水確率 {pop_text}{marker}")
    return "\n".join(lines)


async def fetch_weather(prompt: str) -> str | None:
    """Authoritative daily forecast block for a local-weather
    question, or None (caller falls back to web search).

    Source order is hedged across two independent providers: for
    known major places the official JMA feed goes first and
    Open-Meteo backs it up; anywhere else the geocoder +
    Open-Meteo go first and the JMA prefecture feed backs them up.
    """
    if _WEATHER_NON_FORECAST_RE.search(prompt):
        return None  # typhoons / earthquakes are news, not forecasts
    place = _extract_place(prompt)
    if place is None:
        print(f"Weather: no place extracted from {prompt!r}")
        return None
    offset = _weather_day_offset(prompt)
    core = _place_core(place)
    async with httpx.AsyncClient(timeout=10,
                                 headers=_HTTP_HEADERS) as client:
        table = _table_place(place)
        if table is not None:
            block = await _try_jma(client, table.get("jma_code"),
                                   core, table["name"],
                                   table["admin1"], offset)
            if block is not None:
                return block
            try:
                return await _open_meteo_block(client, table, offset)
            except Exception as exc:  # noqa: BLE001
                print(f"Weather fetch failed: {exc}")
                return None
        top = await _geocode_place(client, place)
        if top is None:
            return None
        try:
            return await _open_meteo_block(client, top, offset)
        except Exception as exc:  # noqa: BLE001
            print(f"Weather fetch failed: {exc}")
        code = _PREFECTURE_JMA_CODES.get(top.get("admin1") or "")
        return await _try_jma(client, code, core,
                              top.get("name", place),
                              top.get("admin1") or "", offset)


_TICKER_RES = [
    # TSE codes are 4 chars and may contain letters since 2024
    # (e.g. 167A), so digits alone are not enough.
    (re.compile(r"kabutan\.jp/stock/\?code=([0-9A-Z]{4})"), "{}.T"),
    (re.compile(r"finance\.yahoo\.co\.jp/quote/([0-9A-Z]{4})\.T"), "{}.T"),
    (re.compile(r"minkabu\.jp/stock/([0-9A-Z]{4})"), "{}.T"),
    (re.compile(r"kabutan\.jp/stock_us/\?code=([A-Z]{1,6})"), "{}"),
]

# An explicitly named US ticker: 「NVDAの株価」「ティッカー: AAPL」.
_US_TICKER_RE = re.compile(
    r"([A-Za-z]{2,5})の株価|ティッカー[:：]?\s*([A-Za-z]{1,5})")


def _extract_ticker(sources: list) -> str | None:
    """Identify a Yahoo Finance ticker from search-result URLs
    (kabutan / Yahoo Finance JP / minkabu quote pages)."""
    for _title, url in sources:
        for pattern, fmt in _TICKER_RES:
            match = pattern.search(url)
            if match:
                code = match.group(1)
                # A JP code always contains at least one digit;
                # this keeps stray letter sequences out.
                if fmt == "{}.T" and not any(c.isdigit() for c in code):
                    continue
                return fmt.format(code)
    return None


async def _prefetch_finance(prompt: str, sources: list) -> str:
    """Finance route for stocks and other market questions.

    Search snippets structurally lose live prices (portal pages are
    JS-rendered and noisy), so: domain-restricted Tavily search with
    raw page content and NO date in the query (a date biases results
    toward dated outlook articles), then, when a ticker can be
    identified from the result URLs, an authoritative Yahoo Finance
    quote replaces the search text.
    """
    # An explicitly named US ticker needs no search at all: fetch
    # it directly (zero Tavily credits). A failed fetch falls
    # through to the normal search route.
    ticker_match = _US_TICKER_RE.search(prompt)
    if ticker_match:
        us_ticker = (ticker_match.group(1) or ticker_match.group(2)).upper()
        quote = await fetch_quote(us_ticker)
        if quote is not None:
            url = f"https://finance.yahoo.com/quote/{us_ticker}"
            return _quote_block(f"{us_ticker} の株価", url, quote)
    q = prompt.replace("？", "").replace("?", "").strip()
    q = re.sub(r"(は|を)?(教えて|ください)$", "", q).strip()
    if "現在値" not in q:
        q += " 現在値"
    text = await search_web(
        q, include_domains=STOCK_DOMAINS, include_raw_content=True
    )
    if text.startswith("検索結果は見つかりませんでした"):
        return await search_web(prompt)
    ticker = _extract_ticker(sources)
    if ticker:
        quote = await fetch_quote(ticker)
        if quote is not None:
            if ticker.endswith(".T"):
                url = f"https://finance.yahoo.co.jp/quote/{ticker}"
            else:
                url = f"https://finance.yahoo.com/quote/{ticker}"
            block = _quote_block(f"{ticker} の株価", url, quote)
            return block + "\n\n（参考）検索結果:\n" + text[:800]
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
        direct = _match_direct_symbol(prompt)
        if NEWS_PATTERN.search(prompt):
            results_text = await search_news(prompt, days=3)
        elif direct is not None:
            # FX / indices / crypto: fetch the number directly, no
            # Tavily at all (search cannot carry live values).
            ticker, label, source_url, pair = direct
            quote = await fetch_quote(ticker)
            if quote is not None:
                results_text = _quote_block(label, source_url, quote)
            elif pair is not None:
                daily = await fetch_fx_rate_fallback(*pair)
                if daily is not None:
                    er_url = f"https://open.er-api.com/v6/latest/{pair[0]}"
                    results_text = _quote_block(label, er_url, daily, daily=True)
                else:
                    results_text = await search_web(prompt)
            else:
                results_text = await search_web(prompt)
        elif FINANCE_PATTERN.search(prompt):
            results_text = await _prefetch_finance(prompt, sources)
        elif WEATHER_PATTERN.search(prompt):
            # Local forecasts come from the Open-Meteo API directly
            # (portal snippets lose tomorrow's numbers). Anything
            # the direct fetch cannot handle falls back to a
            # domain-restricted search of forecast sites.
            results_text = await fetch_weather(prompt)
            if results_text is None:
                results_text = await search_web(
                    query, include_domains=WEATHER_DOMAINS)
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

diet.register(bot)

# Threads this bot is actively conversing in
active_threads: set[int] = set()

_commands_synced = False


@bot.event
async def on_ready():
    global _commands_synced
    print(f"Logged in as {bot.user} (id={bot.user.id})")
    # Preload diet state for all guilds in the background (fast first
    # commands, reliable 8:00 reminders), then start the reminder loop.
    bot.loop.create_task(diet.preload_and_start())
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
