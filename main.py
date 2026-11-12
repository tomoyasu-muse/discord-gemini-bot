"""
Discord + Gemini AI chat bot.

- In a normal channel: when the bot is mentioned, a new thread is created
  (named after the sender) and the bot answers inside that thread.
- Inside a thread owned by this flow: no mention is needed, the bot keeps
  a continuous conversation (per-thread Gemini chat session).
- keep_alive: a small Flask server on port 8080 runs in a background
  thread so Render (and uptime pingers) have an HTTP endpoint to hit.
"""

import os
import threading

import discord
import google.generativeai as genai
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
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

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
    chat = get_chat_session(thread_id)
    # google-generativeai is synchronous; run it off the event loop.
    import asyncio

    response = await asyncio.to_thread(chat.send_message, prompt)
    text = (response.text or "").strip()
    return text if text else "（応答が空でした。もう一度お試しください）"


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

client = discord.Client(intents=intents)

# Threads this bot is actively conversing in
active_threads: set[int] = set()


@client.event
async def on_ready():
    print(f"Logged in as {client.user} (id={client.user.id})")


@client.event
async def on_message(message: discord.Message):
    # Ignore our own messages and other bots
    if message.author.bot:
        return

    in_thread = isinstance(message.channel, discord.Thread)

    # --- Inside a thread: continue the conversation without a mention ---
    if in_thread and message.channel.id in active_threads:
        prompt = message.content.strip()
        if not prompt:
            return
        async with message.channel.typing():
            try:
                reply = await ask_gemini(message.channel.id, prompt)
            except Exception as exc:  # noqa: BLE001
                print(f"Gemini error: {exc}")
                await message.channel.send("エラーが発生しました。しばらくしてからもう一度お試しください。")
                return
        for chunk in split_message(reply):
            await message.channel.send(chunk)
        return

    # --- In a normal channel: react only when mentioned ---
    if client.user is None or client.user not in message.mentions:
        return

    # Strip the mention itself from the prompt
    prompt = message.content
    for mention in (f"<@{client.user.id}>", f"<@!{client.user.id}>"):
        prompt = prompt.replace(mention, "")
    prompt = prompt.strip()
    if not prompt:
        prompt = "こんにちは"

    # Create a new thread named after the sender
    thread_name = f"{message.author.display_name} のスレッド"
    try:
        thread = await message.create_thread(name=thread_name[:100])
    except discord.HTTPException as exc:
        print(f"Thread creation failed: {exc}")
        await message.channel.send("スレッドの作成に失敗しました。ボットにスレッド作成権限があるか確認してください。")
        return

    active_threads.add(thread.id)

    async with thread.typing():
        try:
            reply = await ask_gemini(thread.id, prompt)
        except Exception as exc:  # noqa: BLE001
            print(f"Gemini error: {exc}")
            await thread.send("エラーが発生しました。しばらくしてからもう一度お試しください。")
            return

    for chunk in split_message(reply):
        await thread.send(chunk)


def main():
    if not DISCORD_TOKEN:
        raise RuntimeError("DISCORD_TOKEN environment variable is not set")
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")
    keep_alive()
    client.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
