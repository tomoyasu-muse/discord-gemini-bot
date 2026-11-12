# discord-gemini-bot

Discord + Gemini API chat bot.

## Behaviour
- Mention the bot in a normal channel -> it creates a new thread named after
  the sender and answers there.
- Inside that thread, no mention is needed; the conversation continues with
  per-thread Gemini chat history.
- A Flask keep-alive server listens on port 8080 (`/` and `/health`).

## Environment variables
- `DISCORD_TOKEN` — Discord bot token (Message Content Intent must be ON)
- `GEMINI_API_KEY` — Gemini API key from Google AI Studio
- `GEMINI_MODEL` — optional, defaults to `gemini-2.0-flash`
- `PORT` — optional, defaults to `8080` (Render sets this automatically)

## Run on Render
- Build command: `pip install -r requirements.txt`
- Start command: `python main.py`

## Run locally
```bash
pip install -r requirements.txt
DISCORD_TOKEN=... GEMINI_API_KEY=... python main.py
```
