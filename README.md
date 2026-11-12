# discord-gemini-bot

Discord + Gemini API chat bot.

## Behaviour
- Mention the bot in a normal channel -> it answers with a one-shot reply
  in that same channel. No thread is created and no history is kept.
- Use the `/thread` slash command with a question -> it creates a thread
  and answers inside it. Inside that thread, no mention is needed; the
  conversation continues with per-thread Gemini chat history.
- Mentioning the bot inside any thread (re)activates that thread as a
  conversation thread.
- Search: start a message with 「調べて」 (or use the `/search` command)
  and the model itself drives Tavily tools through Gemini automatic
  function calling: `search_web` (basic search = 1 credit by default),
  `search_news` (news only, recency-limited) and `extract_page` (reads
  a full page when a snippet is not enough). The system instruction
  embeds the current JST date/time so queries stay recency-aware, and
  the model must cite only URLs that actually came back from the
  tools; when the answer contains no URL, the bot appends the real
  source list it collected. Questions that clearly need real-time data
  (weather, FX rates, stock prices, news…) are auto-detected by a
  keyword regex — no 「調べて」 needed — and use a faster pre-fetch
  route (Python searches Tavily once, then one Gemini call). For FX
  rates, major indices and Bitcoin, no search is used at all: the
  number is fetched directly from Yahoo Finance's chart API (with a
  daily ExchangeRate-API fallback for currency pairs). Stock questions
  use a domain-restricted Tavily search (kabutan.jp,
  finance.yahoo.co.jp, minkabu.jp) with raw page content, and when a
  ticker code can be identified from the result URLs, the same chart
  API provides the authoritative price. Tavily's free plan allows 1,000 credits/month and needs no credit card. If a
  search fails (or no API key is configured), the bot falls back to a
  normal answer; a Gemini rate limit (429) is reported as such instead.
- Diet support (`diet.py`): run `/diet-setup` once per server; it
  creates a 「ダイエット記録」 channel. Meal/weight records are
  stored as bot messages (`DIETLOG|1|...`) tagged with their user,
  while settings (config + goals) live in one pinned `DIETSET`
  message edited in place, so they never fall out of the history
  window. State is preloaded on startup; logs survive redeploys
  with no database and stay separate per user. `/diet-undo` deletes
  the caller's most recent meal/weight record (message included).
  When `GITHUB_TOKEN` and `DIET_BACKUP_REPO` are set, all records
  are merged into `diet_log.csv` in that repo once a day at 21:00
  as a backup against channel deletion.
  Diet actions are slash commands only (mentions stay purely for
  questions). With `/gohan`, write the food and calories
  (「納豆ご飯 300kcal」) and the numbers are logged as-is; the
  parser also accepts several items at once (「マグロ200g 230kcal
  パックご飯180kcal」) and a combined total in parentheses
  (「ご飯、納豆、オクラ（400kcal）」). Meals are totalled per
  calendar day (00:00–24:00 JST) and each log reply reports the
  day's intake against that user's daily goal. Without calories,
  the same inputs let Gemini estimate calories and PFC instead.
  `/weight` logs weight. `/diet-goal` sets a per-user target weight
  and daily calorie goal, `/today` and `/week` summarise, and
  reminders post at 08:00 (weight) and 21:00 (missing logs; weekly
  summary on Sundays), switchable with `/diet-remind`.
- Gemini is called via the google-genai SDK's async client
  (`client.aio`) and Tavily via `AsyncTavilyClient`, so API calls
  never block the Discord event loop.
- A Flask keep-alive server listens on port 8080 (`/` and `/health`).

## Environment variables
- `DISCORD_TOKEN` — Discord bot token (Message Content Intent must be ON)
- `GEMINI_API_KEY` — Gemini API key from Google AI Studio
- `GEMINI_MODEL` — optional, defaults to `gemini-3.5-flash-lite`
- `TAVILY_API_KEY` — Tavily API key (free plan), needed for search
- `GITHUB_TOKEN` / `DIET_BACKUP_REPO` — optional; a fine-grained
  token (Contents read/write on the backup repo only) and the repo
  as `owner/repo`, enabling the daily diet-log CSV backup
- `PORT` — optional, defaults to `8080` (Render sets this automatically)

## Run on Render
- Build command: `pip install -r requirements.txt`
- Start command: `python main.py`

## Run locally
```bash
pip install -r requirements.txt
DISCORD_TOKEN=... GEMINI_API_KEY=... python main.py
```
