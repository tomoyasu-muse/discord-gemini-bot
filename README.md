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
  and the question is answered by a separate search model
  (`gemini-2.5-flash-lite`) with Google Search grounding; sources are
  appended to the answer. The search model has its own small quota
  (20 requests/day on the current project), and if a search fails the
  bot falls back to a normal answer.
- A Flask keep-alive server listens on port 8080 (`/` and `/health`).

## Environment variables
- `DISCORD_TOKEN` — Discord bot token (Message Content Intent must be ON)
- `GEMINI_API_KEY` — Gemini API key from Google AI Studio
- `GEMINI_MODEL` — optional, defaults to `gemini-3.5-flash-lite`
- `SEARCH_MODEL` — optional, defaults to `gemini-2.5-flash-lite`
- `PORT` — optional, defaults to `8080` (Render sets this automatically)

## Run on Render
- Build command: `pip install -r requirements.txt`
- Start command: `python main.py`

## Run locally
```bash
pip install -r requirements.txt
DISCORD_TOKEN=... GEMINI_API_KEY=... python main.py
```
