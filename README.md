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
  route (Python searches Tavily once, then one Gemini call). Tavily's free plan allows 1,000 credits/month and needs no credit card. If a
  search fails (or no API key is configured), the bot falls back to a
  normal answer; a Gemini rate limit (429) is reported as such instead.
- Gemini is called via the google-genai SDK's async client
  (`client.aio`) and Tavily via `AsyncTavilyClient`, so API calls
  never block the Discord event loop.
- A Flask keep-alive server listens on port 8080 (`/` and `/health`).

## Environment variables
- `DISCORD_TOKEN` — Discord bot token (Message Content Intent must be ON)
- `GEMINI_API_KEY` — Gemini API key from Google AI Studio
- `GEMINI_MODEL` — optional, defaults to `gemini-3.5-flash-lite`
- `TAVILY_API_KEY` — Tavily API key (free plan), needed for search
- `PORT` — optional, defaults to `8080` (Render sets this automatically)

## Run on Render
- Build command: `pip install -r requirements.txt`
- Start command: `python main.py`

## Run locally
```bash
pip install -r requirements.txt
DISCORD_TOKEN=... GEMINI_API_KEY=... python main.py
```
