"""
Diet support for the bot: meal / weight logging with Gemini-estimated
calories and PFC, goals, daily & weekly summaries, and reminders.

Storage: there is no database. Each guild gets a text channel named
「ダイエット記録」; every record is one message posted by the bot in a
machine-readable one-line format (DIETLOG|1|...). On first use the bot
re-reads that channel's history and rebuilds its in-memory state, so
logs survive redeploys and stay visible (and editable) in Discord
itself. Reminder and summary posts in the same channel do not carry
the prefix, so they never pollute the records.

Single-user-per-guild assumption: the person who ran /diet-setup is
the dieter; reminders mention them.
"""

import datetime
import json
import re
import zoneinfo

import discord
from discord import app_commands
from discord.ext import commands, tasks
from google.genai import types

JST = zoneinfo.ZoneInfo("Asia/Tokyo")
LOG_CHANNEL_NAME = "ダイエット記録"
RECORD_PREFIX = "DIETLOG|1|"

_client = None
_model = None
_bot = None


def configure(gemini_client, model: str):
    global _client, _model
    _client = gemini_client
    _model = model


# ---------------------------------------------------------------------------
# Record encoding / decoding
# ---------------------------------------------------------------------------
def _clean_field(text: str) -> str:
    return re.sub(r"[|\n\r]+", " ", str(text)).strip()


def encode_record(rtype: str, ts: datetime.datetime, fields: dict,
                  text: str = "") -> str:
    kv = ";".join(
        f"{key}={_clean_field(value)}" for key, value in fields.items()
    )
    return f"{RECORD_PREFIX}{rtype}|{ts.isoformat()}|{kv}|{_clean_field(text)}"


def decode_record(content: str) -> dict | None:
    if not content.startswith(RECORD_PREFIX):
        return None
    parts = content.split("|", 5)
    if len(parts) < 6:
        return None
    _tag, _ver, rtype, ts_raw, kv_raw, text = parts
    try:
        ts = datetime.datetime.fromisoformat(ts_raw)
    except ValueError:
        return None
    fields = {}
    for pair in kv_raw.split(";"):
        if "=" in pair:
            key, value = pair.split("=", 1)
            fields[key.strip()] = value.strip()
    return {"type": rtype, "ts": ts, "fields": fields, "text": text}


def _f(fields: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(fields.get(key, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Per-guild state
# ---------------------------------------------------------------------------
class GuildDiet:
    def __init__(self, guild_id: int):
        self.guild_id = guild_id
        self.channel_id: int | None = None
        self.loaded = False
        self.records: list[dict] = []

    # -- derived views -------------------------------------------------
    def goal(self) -> dict | None:
        goals = [r for r in self.records if r["type"] == "goal"]
        return goals[-1] if goals else None

    def config(self) -> dict:
        merged: dict = {}
        for record in self.records:
            if record["type"] == "config":
                merged.update(record["fields"])
        return merged

    def setup_user_id(self) -> int | None:
        raw = self.config().get("setup_by")
        try:
            return int(raw) if raw else None
        except ValueError:
            return None

    def remind_on(self) -> bool:
        return self.config().get("remind", "on") != "off"

    def meals_on(self, day: datetime.date) -> list[dict]:
        return [
            r for r in self.records
            if r["type"] == "meal" and r["ts"].astimezone(JST).date() == day
        ]

    def weights(self) -> list[dict]:
        return [r for r in self.records if r["type"] == "weight"]

    def latest_weight(self) -> dict | None:
        ws = self.weights()
        return ws[-1] if ws else None


_states: dict[int, GuildDiet] = {}


def _state(guild_id: int) -> GuildDiet:
    if guild_id not in _states:
        _states[guild_id] = GuildDiet(guild_id)
    return _states[guild_id]


def _find_log_channel(guild: discord.Guild):
    for channel in guild.text_channels:
        if channel.name == LOG_CHANNEL_NAME:
            return channel
    return None


async def _ensure_loaded(guild: discord.Guild) -> GuildDiet:
    """Locate the log channel and (once) rebuild state from history."""
    state = _state(guild.id)
    if state.loaded:
        return state
    channel = _find_log_channel(guild)
    if channel is None:
        state.loaded = True  # no setup yet; nothing to load
        return state
    state.channel_id = channel.id
    records = []
    async for msg in channel.history(limit=1000, oldest_first=True):
        if _bot is not None and msg.author.id == _bot.user.id:
            record = decode_record(msg.content)
            if record is not None:
                records.append(record)
    state.records = records
    state.loaded = True
    return state


async def _append_record(guild: discord.Guild, state: GuildDiet,
                         rtype: str, fields: dict, text: str = "") -> dict:
    channel = guild.get_channel(state.channel_id)
    if channel is None:
        raise RuntimeError("log channel not found")
    now = datetime.datetime.now(JST)
    content = encode_record(rtype, now, fields, text)
    await channel.send(content)
    record = decode_record(content)
    state.records.append(record)
    return record


# ---------------------------------------------------------------------------
# Meal estimation (Gemini, JSON out; totals are summed in Python)
# ---------------------------------------------------------------------------
_ESTIMATE_INSTRUCTION = (
    "あなたは管理栄養士です。渡された食事の内容（テキストまたは写真）について、"
    "一般的な1人前の量を前提に、カロリーとPFC（たんぱく質・脂質・炭水化物）を推定してください。"
    "出力は次のJSON形式のみとし、説明文やコードフェンスは付けないでください。\n"
    '{"meal_name": "料理名の要約", "items": [{"name": "品目名", "kcal": 123, '
    '"protein_g": 1.0, "fat_g": 1.0, "carbs_g": 1.0}]}\n'
    "食事ではない場合や判別できない場合は items を空配列にしてください。"
)


def _parse_estimate(text: str) -> dict | None:
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except (ValueError, TypeError):
        return None
    items = []
    for item in data.get("items") or []:
        try:
            items.append({
                "name": str(item.get("name", "品目")),
                "kcal": float(item.get("kcal", 0)),
                "p": float(item.get("protein_g", 0)),
                "f": float(item.get("fat_g", 0)),
                "c": float(item.get("carbs_g", 0)),
            })
        except (TypeError, ValueError):
            continue
    if not items:
        return None
    return {"meal_name": str(data.get("meal_name", "")), "items": items}


def _totals(items: list[dict]) -> dict:
    return {
        "kcal": sum(i["kcal"] for i in items),
        "p": sum(i["p"] for i in items),
        "f": sum(i["f"] for i in items),
        "c": sum(i["c"] for i in items),
    }


async def _generate_estimate(contents) -> dict | None:
    response = await _client.aio.models.generate_content(
        model=_model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=_ESTIMATE_INSTRUCTION,
            max_output_tokens=1024,
        ),
    )
    try:
        text = response.text or ""
    except Exception:  # noqa: BLE001
        text = ""
    return _parse_estimate(text)


async def estimate_meal_text(description: str) -> dict | None:
    return await _generate_estimate(f"食事の内容: {description}")


async def estimate_meal_image(data: bytes, mime_type: str,
                              note: str = "") -> dict | None:
    parts = [types.Part.from_bytes(data=data, mime_type=mime_type)]
    parts.append(f"この写真の食事を推定してください。補足: {note}" if note
                 else "この写真の食事を推定してください。")
    return await _generate_estimate(parts)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def _fmt_pfc(t: dict) -> str:
    return (f"{t['kcal']:.0f} kcal"
            f"（P {t['p']:.0f}g / F {t['f']:.0f}g / C {t['c']:.0f}g）")


def today_summary(state: GuildDiet) -> str:
    today = datetime.datetime.now(JST).date()
    meals = state.meals_on(today)
    lines = [f"📊 今日（{today:%m/%d}）の記録"]
    total = {"kcal": 0.0, "p": 0.0, "f": 0.0, "c": 0.0}
    if meals:
        for meal in meals:
            t = {k: _f(meal["fields"], k) for k in ("kcal", "p", "f", "c")}
            for key in total:
                total[key] += t[key]
            lines.append(
                f"- {meal['ts'].astimezone(JST):%H:%M} {meal['text']}："
                f"{t['kcal']:.0f} kcal"
            )
        lines.append(f"合計: {_fmt_pfc(total)}")
    else:
        lines.append("食事の記録はまだありません。")
    goal = state.goal()
    if goal and _f(goal["fields"], "daily_kcal") > 0:
        remain = _f(goal["fields"], "daily_kcal") - total["kcal"]
        lines.append(
            f"1日の目標 {_f(goal['fields'], 'daily_kcal'):.0f} kcal まで"
            f"あと {remain:.0f} kcal"
        )
    latest = state.latest_weight()
    if latest:
        kg = _f(latest["fields"], "kg")
        line = f"最新の体重: {kg:.1f} kg"
        if goal:
            diff = kg - _f(goal["fields"], "target_kg")
            line += f"（目標 {_f(goal['fields'], 'target_kg'):.1f} kg まで あと {diff:.1f} kg）"
        lines.append(line)
    return "\n".join(lines)


def week_summary(state: GuildDiet) -> str:
    today = datetime.datetime.now(JST).date()
    lines = [f"📈 直近7日間（{today - datetime.timedelta(days=6):%m/%d}〜{today:%m/%d}）"]
    day_totals = []
    for offset in range(7):
        day = today - datetime.timedelta(days=offset)
        meals = state.meals_on(day)
        if meals:
            kcal = sum(_f(m["fields"], "kcal") for m in meals)
            day_totals.append((day, kcal, len(meals)))
    if day_totals:
        for day, kcal, count in reversed(day_totals):
            lines.append(f"- {day:%m/%d}: {kcal:.0f} kcal（{count}食）")
        avg = sum(k for _, k, _ in day_totals) / len(day_totals)
        lines.append(f"記録日の平均: {avg:.0f} kcal")
    else:
        lines.append("食事の記録はありません。")
    week_ago = datetime.datetime.now(JST) - datetime.timedelta(days=7)
    recent = [w for w in state.weights() if w["ts"] >= week_ago]
    if len(recent) >= 2:
        first = _f(recent[0]["fields"], "kg")
        last = _f(recent[-1]["fields"], "kg")
        lines.append(f"体重: {first:.1f} kg → {last:.1f} kg（{last - first:+.1f} kg）")
    elif state.latest_weight():
        kg = _f(state.latest_weight()["fields"], "kg")
        lines.append(f"最新の体重: {kg:.1f} kg")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Message routing (called from main.on_message before the chat routes)
# ---------------------------------------------------------------------------
WEIGHT_RE = re.compile(r"体重\s*(?:は|が)?\s*(\d{2,3}(?:\.\d+)?)\s*(?:kg|キロ)?")
MEAL_PREFIX_RE = re.compile(r"^(食事|食べた|食べました|朝食|昼食|夕食|夜食|間食|朝ごはん|昼ごはん|夜ごはん|晩ごはん)")

RATE_LIMIT_MESSAGE = (
    "⚠️ いま少し混み合っています（Geminiの無料枠の上限に達しました）。"
    "1分ほど待ってからもう一度試してください。"
)
SETUP_HINT = (
    "ダイエット記録はまだセットアップされていません。"
    "先に /diet-setup を実行してください（記録専用チャンネルを作ります）。"
)


def detect_intent(prompt: str, has_image: bool) -> str | None:
    """'weight' | 'meal' | None — decided locally, no API cost."""
    if WEIGHT_RE.search(prompt):
        return "weight"
    if has_image:
        return "meal"
    if MEAL_PREFIX_RE.match(prompt):
        return "meal"
    return None


async def try_handle(message: discord.Message, prompt: str) -> str | None:
    """Return a reply string if the message is a diet action, else None."""
    guild = message.guild
    if guild is None or _client is None:
        return None
    image_atts = [
        a for a in message.attachments
        if (a.content_type or "").startswith("image/")
    ]
    intent = detect_intent(prompt, bool(image_atts))
    if intent is None:
        return None

    state = await _ensure_loaded(guild)
    if state.channel_id is None:
        return SETUP_HINT

    try:
        if intent == "weight":
            kg = float(WEIGHT_RE.search(prompt).group(1))
            if not (20.0 <= kg <= 300.0):
                return "体重の数値が範囲外のようです。もう一度確認してください。"
            prev = state.latest_weight()
            await _append_record(guild, state, "weight", {"kg": f"{kg:.1f}"})
            line = f"⚖️ 体重を記録しました： {kg:.1f} kg"
            if prev:
                diff = kg - _f(prev["fields"], "kg")
                line += f"（前回 {_f(prev['fields'], 'kg'):.1f} kg から {diff:+.1f} kg）"
            goal = state.goal()
            if goal:
                remain = kg - _f(goal["fields"], "target_kg")
                line += f"\n目標 {_f(goal['fields'], 'target_kg'):.1f} kg まで あと {remain:.1f} kg"
            return line

        # meal
        if image_atts:
            att = image_atts[0]
            data = await att.read()
            est = await estimate_meal_image(data, att.content_type, prompt)
            fallback_name = prompt or "食事（写真）"
        else:
            desc = MEAL_PREFIX_RE.sub("", prompt).strip(" ：:、") or prompt
            est = await estimate_meal_text(desc)
            fallback_name = desc
        if est is None:
            return ("食事の内容を読み取れませんでした。"
                    "写真の場合は食事が写っているか、テキストの場合は"
                    "メニューを具体的に書いてもう一度試してください。")
        totals = _totals(est["items"])
        name = est["meal_name"] or fallback_name
        await _append_record(
            guild, state, "meal",
            {"kcal": f"{totals['kcal']:.0f}", "p": f"{totals['p']:.1f}",
             "f": f"{totals['f']:.1f}", "c": f"{totals['c']:.1f}"},
            name,
        )
        lines = [f"🍽️ 記録しました： {name}"]
        for item in est["items"]:
            lines.append(
                f"- {item['name']}： {item['kcal']:.0f} kcal"
                f"（P {item['p']:.0f} / F {item['f']:.0f} / C {item['c']:.0f}）"
            )
        lines.append(f"推定合計: {_fmt_pfc(totals)}")
        today = datetime.datetime.now(JST).date()
        day_total = sum(
            _f(m["fields"], "kcal") for m in state.meals_on(today)
        )
        goal = state.goal()
        extra = f"今日の合計は {day_total:.0f} kcal です"
        if goal and _f(goal["fields"], "daily_kcal") > 0:
            extra += f"（目標まで あと {_f(goal['fields'], 'daily_kcal') - day_total:.0f} kcal）"
        lines.append(extra + "。")
        lines.append("※カロリーはAIの推定です")
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        print(f"Diet handling failed: {exc}")
        if getattr(exc, "code", None) == 429:
            return RATE_LIMIT_MESSAGE
        return "記録中にエラーが発生しました。しばらくしてからもう一度お試しください。"


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------
def register(bot: commands.Bot):
    global _bot
    _bot = bot
    tree = bot.tree

    @tree.command(name="diet-setup",
                  description="ダイエット記録のセットアップ（記録専用チャンネルを作成）")
    async def diet_setup(interaction: discord.Interaction):
        guild = interaction.guild
        if guild is None:
            await interaction.response.send_message(
                "サーバーの中で実行してください。", ephemeral=True)
            return
        await interaction.response.defer(thinking=True)
        state = await _ensure_loaded(guild)
        if state.channel_id is not None:
            channel = guild.get_channel(state.channel_id)
            await interaction.followup.send(
                f"セットアップ済みです。記録チャンネル: {channel.mention if channel else LOG_CHANNEL_NAME}\n"
                "使い方は /diet-goal で目標設定、食事の写真をメンション付きで送るか、"
                "「食べた：〜」と送れば記録します。「体重 68.4」でも記録できます。"
            )
            return
        overwrites = {
            guild.default_role: discord.PermissionOverwrite(read_messages=False),
            guild.me: discord.PermissionOverwrite(
                read_messages=True, send_messages=True, read_message_history=True
            ),
        }
        try:
            channel = await guild.create_text_channel(
                LOG_CHANNEL_NAME, overwrites=overwrites,
                topic="ダイエット記録（ボットが記録を保存します。消さないでください）",
            )
        except discord.HTTPException as exc:
            print(f"Diet channel creation failed: {exc}")
            await interaction.followup.send(
                "チャンネルの作成に失敗しました。ボットにチャンネル管理権限があるか確認してください。"
            )
            return
        state.channel_id = channel.id
        state.loaded = True
        await _append_record(
            guild, state, "config",
            {"setup_by": str(interaction.user.id), "remind": "on"},
        )
        await channel.send(
            "🥗 ダイエット記録をはじめます。ここに記録が保存されます。\n"
            "- 食事： 写真をメンション付きで送る、または「食べた：親子丼」のように送る\n"
            "- 体重： 「体重 68.4」のように送る\n"
            "- 目標： /diet-goal で設定\n"
            "- 集計： /today（今日）・/week（直近7日）\n"
            "- リマインド： 朝8時に体重、夜21時に記録もれの声かけ（/diet-remind で切替）"
        )
        await interaction.followup.send(
            f"記録チャンネル {channel.mention} を作りました。"
            "次に /diet-goal で目標を設定すると、残りカロリーや目標までの差も出せます。"
        )

    @tree.command(name="diet-goal", description="目標体重と1日の目標カロリーを設定")
    @app_commands.describe(target_kg="目標体重（kg）",
                           daily_kcal="1日の目標摂取カロリー（任意）")
    async def diet_goal(interaction: discord.Interaction,
                        target_kg: float, daily_kcal: int = 0):
        guild = interaction.guild
        if guild is None:
            await interaction.response.send_message(
                "サーバーの中で実行してください。", ephemeral=True)
            return
        state = await _ensure_loaded(guild)
        if state.channel_id is None:
            await interaction.response.send_message(
                SETUP_HINT, ephemeral=True)
            return
        fields = {"target_kg": f"{target_kg:.1f}",
                  "daily_kcal": str(int(daily_kcal))}
        latest = state.latest_weight()
        if latest:
            fields["start_kg"] = f"{_f(latest['fields'], 'kg'):.1f}"
        await _append_record(guild, state, "goal", fields)
        msg = f"🎯 目標を設定しました： {target_kg:.1f} kg"
        if daily_kcal:
            msg += f" / 1日 {int(daily_kcal)} kcal"
        if latest:
            msg += f"（現在 {_f(latest['fields'], 'kg'):.1f} kg、あと {(_f(latest['fields'], 'kg') - target_kg):.1f} kg）"
        await interaction.response.send_message(msg)

    @tree.command(name="meal", description="食事をテキストで記録（カロリー推定つき）")
    @app_commands.describe(text="食べたもの（例：親子丼と味噌汁）")
    async def meal_command(interaction: discord.Interaction, text: str):
        guild = interaction.guild
        if guild is None:
            await interaction.response.send_message(
                "サーバーの中で実行してください。", ephemeral=True)
            return
        state = await _ensure_loaded(guild)
        if state.channel_id is None:
            await interaction.response.send_message(
                SETUP_HINT, ephemeral=True)
            return
        await interaction.response.defer(thinking=True)
        try:
            est = await estimate_meal_text(text)
        except Exception as exc:  # noqa: BLE001
            print(f"Meal estimate failed: {exc}")
            await interaction.followup.send(
                RATE_LIMIT_MESSAGE if getattr(exc, "code", None) == 429
                else "推定中にエラーが発生しました。")
            return
        if est is None:
            await interaction.followup.send(
                "食事の内容を読み取れませんでした。メニューを具体的に書いてください。")
            return
        totals = _totals(est["items"])
        name = est["meal_name"] or text
        await _append_record(
            guild, state, "meal",
            {"kcal": f"{totals['kcal']:.0f}", "p": f"{totals['p']:.1f}",
             "f": f"{totals['f']:.1f}", "c": f"{totals['c']:.1f}"},
            name,
        )
        await interaction.followup.send(
            f"🍽️ 記録しました： {name}\n推定合計: {_fmt_pfc(totals)}\n"
            "※カロリーはAIの推定です")

    @tree.command(name="weight", description="体重を記録")
    @app_commands.describe(kg="体重（kg）")
    async def weight_command(interaction: discord.Interaction, kg: float):
        guild = interaction.guild
        if guild is None:
            await interaction.response.send_message(
                "サーバーの中で実行してください。", ephemeral=True)
            return
        state = await _ensure_loaded(guild)
        if state.channel_id is None:
            await interaction.response.send_message(
                SETUP_HINT, ephemeral=True)
            return
        if not (20.0 <= kg <= 300.0):
            await interaction.response.send_message(
                "体重の数値が範囲外のようです。", ephemeral=True)
            return
        prev = state.latest_weight()
        await _append_record(guild, state, "weight", {"kg": f"{kg:.1f}"})
        line = f"⚖️ 体重を記録しました： {kg:.1f} kg"
        if prev:
            line += f"（前回 {_f(prev['fields'], 'kg'):.1f} kg から {kg - _f(prev['fields'], 'kg'):+.1f} kg）"
        await interaction.response.send_message(line)

    @tree.command(name="today", description="今日の食事・体重のまとめ")
    async def today_command(interaction: discord.Interaction):
        guild = interaction.guild
        if guild is None:
            await interaction.response.send_message(
                "サーバーの中で実行してください。", ephemeral=True)
            return
        state = await _ensure_loaded(guild)
        if state.channel_id is None:
            await interaction.response.send_message(
                SETUP_HINT, ephemeral=True)
            return
        await interaction.response.send_message(today_summary(state))

    @tree.command(name="week", description="直近7日間のまとめ")
    async def week_command(interaction: discord.Interaction):
        guild = interaction.guild
        if guild is None:
            await interaction.response.send_message(
                "サーバーの中で実行してください。", ephemeral=True)
            return
        state = await _ensure_loaded(guild)
        if state.channel_id is None:
            await interaction.response.send_message(
                SETUP_HINT, ephemeral=True)
            return
        await interaction.response.send_message(week_summary(state))

    @tree.command(name="diet-remind", description="朝・夜のリマインドのオン/オフ")
    @app_commands.describe(on_off="on または off")
    @app_commands.choices(on_off=[
        app_commands.Choice(name="on", value="on"),
        app_commands.Choice(name="off", value="off"),
    ])
    async def diet_remind(interaction: discord.Interaction, on_off: str):
        guild = interaction.guild
        if guild is None:
            await interaction.response.send_message(
                "サーバーの中で実行してください。", ephemeral=True)
            return
        state = await _ensure_loaded(guild)
        if state.channel_id is None:
            await interaction.response.send_message(
                SETUP_HINT, ephemeral=True)
            return
        await _append_record(guild, state, "config", {"remind": on_off})
        await interaction.response.send_message(
            f"リマインドを {on_off} にしました。")


# ---------------------------------------------------------------------------
# Reminders (started from main.on_ready)
# ---------------------------------------------------------------------------
_reminded: set[tuple[int, str, str]] = set()
_loops_started = False


@tasks.loop(minutes=1)
async def _reminder_loop():
    now = datetime.datetime.now(JST)
    kind = None
    if (now.hour, now.minute) == (8, 0):
        kind = "morning"
    elif (now.hour, now.minute) == (21, 0):
        kind = "sunday-week" if now.weekday() == 6 else "evening"
    if kind is None or _bot is None:
        return
    today_key = now.date().isoformat()
    for guild in list(_bot.guilds):
        key = (guild.id, today_key, kind)
        if key in _reminded:
            continue
        state = await _ensure_loaded(guild)
        if state.channel_id is None or not state.remind_on():
            continue
        channel = guild.get_channel(state.channel_id)
        if channel is None:
            continue
        _reminded.add(key)
        user_id = state.setup_user_id()
        mention = f"<@{user_id}> " if user_id else ""
        try:
            if kind == "morning":
                latest = state.latest_weight()
                extra = (f"前回の記録は {_f(latest['fields'], 'kg'):.1f} kg でした。"
                         if latest else "")
                await channel.send(
                    f"{mention}おはようございます☀️ 体重を測ったら"
                    "「体重 68.4」のように教えてください。" + extra)
            elif kind == "evening":
                await channel.send(
                    f"{mention}今日の記録もれはないですか？\n"
                    + today_summary(state))
            else:
                await channel.send(
                    f"{mention}1週間お疲れさまでした。\n"
                    + week_summary(state))
        except Exception as exc:  # noqa: BLE001
            print(f"Diet reminder failed for guild {guild.id}: {exc}")


def start_reminders():
    global _loops_started
    if _loops_started:
        return
    _loops_started = True
    _reminder_loop.start()
