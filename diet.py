"""
Diet support for the bot: per-user meal / weight logging, daily
calorie goals, daily & weekly summaries, and reminders.

Logging a meal is deliberately low-effort: write the food and its
calories — 「納豆ご飯 300kcal」 — in a mention or the /gohan
command, and the numbers are recorded as-is. The parser accepts
several items at once (「マグロ200g 230kcal パックご飯180kcal」),
a combined total in parentheses (「ご飯、納豆、オクラ（400kcal）」),
and half-/full-width spacing. Meals are totalled per calendar day
(00:00–24:00 JST) and every log reply reports the day's intake so
far. If the calories are unknown, the same inputs without numbers
let Gemini estimate calories and PFC instead.

Storage: there is no database. Each guild gets a text channel named
「ダイエット記録」; every record is one message posted by the bot in a
machine-readable one-line format (DIETLOG|1|...), tagged with the
user it belongs to. On first use the bot re-reads that channel's
history and rebuilds its in-memory state, so logs survive redeploys
and stay visible (and editable) in Discord itself. Reminder and
summary posts in the same channel do not carry the prefix, so they
never pollute the records.
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
# Per-guild state (records are filtered per user at read time)
# ---------------------------------------------------------------------------
class GuildDiet:
    def __init__(self, guild_id: int):
        self.guild_id = guild_id
        self.channel_id: int | None = None
        self.loaded = False
        self.records: list[dict] = []

    # -- helpers --------------------------------------------------------
    @staticmethod
    def _mine(records: list[dict], user_id: int) -> list[dict]:
        return [r for r in records
                if r["fields"].get("user") == str(user_id)]

    def users(self) -> list[int]:
        """User ids that have any diet record in this guild (in order
        of first appearance), falling back to the setup user."""
        seen: list[int] = []
        for record in self.records:
            raw = record["fields"].get("user")
            if raw:
                try:
                    uid = int(raw)
                except ValueError:
                    continue
                if uid not in seen:
                    seen.append(uid)
        if not seen:
            setup = self.setup_user_id()
            if setup:
                seen.append(setup)
        return seen

    # -- derived views --------------------------------------------------
    def goal(self, user_id: int) -> dict | None:
        goals = self._mine(
            [r for r in self.records if r["type"] == "goal"], user_id)
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

    def meals_on(self, day: datetime.date, user_id: int) -> list[dict]:
        return self._mine([
            r for r in self.records
            if r["type"] == "meal" and r["ts"].astimezone(JST).date() == day
        ], user_id)

    def weights(self, user_id: int) -> list[dict]:
        return self._mine(
            [r for r in self.records if r["type"] == "weight"], user_id)

    def latest_weight(self, user_id: int) -> dict | None:
        ws = self.weights(user_id)
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
                         rtype: str, fields: dict, text: str = "",
                         user_id: int | None = None) -> dict:
    channel = guild.get_channel(state.channel_id)
    if channel is None:
        raise RuntimeError("log channel not found")
    if user_id is not None:
        fields = {"user": str(user_id), **fields}
    now = datetime.datetime.now(JST)
    content = encode_record(rtype, now, fields, text)
    await channel.send(content)
    record = decode_record(content)
    state.records.append(record)
    return record


# ---------------------------------------------------------------------------
# Meal estimation (Gemini, JSON out; totals are summed in Python)
# Used only when the user does not write the calories themselves.
# ---------------------------------------------------------------------------
_ESTIMATE_INSTRUCTION = (
    "あなたは管理栄養士です。渡された食事の内容について、"
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


async def estimate_meal_text(description: str) -> dict | None:
    response = await _client.aio.models.generate_content(
        model=_model,
        contents=f"食事の内容: {description}",
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


# ---------------------------------------------------------------------------
# Summaries (per user; days are JST calendar days, 00:00–24:00)
# ---------------------------------------------------------------------------
def _fmt_pfc(t: dict) -> str:
    return (f"{t['kcal']:.0f} kcal"
            f"（P {t['p']:.0f}g / F {t['f']:.0f}g / C {t['c']:.0f}g）")


def _today_line(state: GuildDiet, user_id: int) -> str:
    """The standing status line appended after every log entry."""
    today = datetime.datetime.now(JST).date()
    total = sum(_f(m["fields"], "kcal")
                for m in state.meals_on(today, user_id))
    line = f"今日の摂取カロリーは合計 {total:.0f} kcal です"
    goal = state.goal(user_id)
    if goal and _f(goal["fields"], "daily_kcal") > 0:
        daily = _f(goal["fields"], "daily_kcal")
        remain = daily - total
        if remain >= 0:
            line += f"（1日の目標 {daily:.0f} kcal まで あと {remain:.0f} kcal）"
        else:
            line += f"（1日の目標 {daily:.0f} kcal を {-remain:.0f} kcal 超えています）"
    return line


def today_summary(state: GuildDiet, user_id: int) -> str:
    today = datetime.datetime.now(JST).date()
    meals = state.meals_on(today, user_id)
    lines = [f"📊 今日（{today:%m/%d}）の記録"]
    if meals:
        total_kcal = 0.0
        est_total = {"kcal": 0.0, "p": 0.0, "f": 0.0, "c": 0.0}
        has_manual = False
        for meal in meals:
            kcal = _f(meal["fields"], "kcal")
            total_kcal += kcal
            manual = meal["fields"].get("manual") == "1"
            has_manual = has_manual or manual
            if not manual:
                for key in est_total:
                    est_total[key] += _f(meal["fields"], key)
            suffix = "（手入力）" if manual else ""
            lines.append(
                f"- {meal['ts'].astimezone(JST):%H:%M} {meal['text']}："
                f"{kcal:.0f} kcal{suffix}"
            )
        lines.append(_today_line(state, user_id))
        if est_total["kcal"] > 0:
            note = "（推定分のみ）" if has_manual else ""
            lines.append(
                f"PFC{note}: P {est_total['p']:.0f}g / "
                f"F {est_total['f']:.0f}g / C {est_total['c']:.0f}g")
    else:
        lines.append("食事の記録はまだありません。")
    latest = state.latest_weight(user_id)
    if latest:
        kg = _f(latest["fields"], "kg")
        line = f"最新の体重: {kg:.1f} kg"
        goal = state.goal(user_id)
        if goal:
            diff = kg - _f(goal["fields"], "target_kg")
            line += f"（目標 {_f(goal['fields'], 'target_kg'):.1f} kg まで あと {diff:.1f} kg）"
        lines.append(line)
    return "\n".join(lines)


def week_summary(state: GuildDiet, user_id: int) -> str:
    today = datetime.datetime.now(JST).date()
    lines = [f"📈 直近7日間（{today - datetime.timedelta(days=6):%m/%d}〜{today:%m/%d}）"]
    day_totals = []
    for offset in range(7):
        day = today - datetime.timedelta(days=offset)
        meals = state.meals_on(day, user_id)
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
    recent = [w for w in state.weights(user_id) if w["ts"] >= week_ago]
    if len(recent) >= 2:
        first = _f(recent[0]["fields"], "kg")
        last = _f(recent[-1]["fields"], "kg")
        lines.append(f"体重: {first:.1f} kg → {last:.1f} kg（{last - first:+.1f} kg）")
    elif state.latest_weight(user_id):
        kg = _f(state.latest_weight(user_id)["fields"], "kg")
        lines.append(f"最新の体重: {kg:.1f} kg")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Message routing (called from main.on_message before the chat routes)
# ---------------------------------------------------------------------------
# Manual meal input is parsed, not just regex-matched, because the
# writing varies: several items with their own calories
# (「マグロ200g 230kcal パックご飯180kcal」), one combined total in
# parentheses (「ご飯、納豆、オクラ（400kcal）」), half- and
# full-width spaces. parse_meal_input returns None for anything it
# cannot fully account for (questions like 「…は何kcal？」 have no
# digits before kcal, and trailing non-filler text rejects the
# parse), so such messages fall through to normal chat.
_KCAL_TOKEN_RE = re.compile(r"(\d{2,4})\s*(?:kcal|カロリー)")
_PAREN_TOTAL_RE = re.compile(r"[（(]\s*(\d{2,4})\s*(?:kcal|カロリー)\s*[）)]\s*$")
_FILLER_SUFFIX_RE = re.compile(r"(を食べた|食べた|でした|です|だよ|だね|のみ|だけ)$")
_ITEM_SEPARATORS = " 　、・,，"


def _clean_item_name(text: str) -> str:
    return text.strip(_ITEM_SEPARATORS)


def parse_meal_input(text: str) -> dict | None:
    """Parse hand-written meal input into {name, total, items}.

    items is a list of (item name, kcal or None). Returns None when
    the text carries no usable calories or has unexplained residue.
    """
    if not text:
        return None
    t = text.strip()
    if t.endswith(("？", "?")):
        return None
    paren = _PAREN_TOTAL_RE.search(t)
    if paren:
        total = int(paren.group(1))
        name = _clean_item_name(t[:paren.start()])
        if not name:
            return None
        parts = [p.strip() for p in re.split(r"[、・,，]", name) if p.strip()]
        return {"name": name, "total": total,
                "items": [(part, None) for part in parts]}
    tokens = list(_KCAL_TOKEN_RE.finditer(t))
    if not tokens:
        return None
    items = []
    pos = 0
    for token in tokens:
        name = _clean_item_name(t[pos:token.start()])
        if not name:
            return None
        items.append((name, int(token.group(1))))
        pos = token.end()
    residue = _FILLER_SUFFIX_RE.sub("", t[pos:].strip(_ITEM_SEPARATORS)).strip()
    if residue:
        return None
    return {"name": "、".join(name for name, _ in items),
            "total": sum(kcal for _, kcal in items),
            "items": items}


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


def detect_intent(prompt: str) -> str | None:
    """'manual' | 'weight' | 'meal' | None — decided locally, no API cost."""
    # Messages starting with a meal prefix go to the meal route,
    # which strips the prefix and then tries the manual parser on
    # the rest (「食べた：納豆ご飯 300kcal」).
    if MEAL_PREFIX_RE.match(prompt):
        return "meal"
    if parse_meal_input(prompt):
        return "manual"
    if WEIGHT_RE.search(prompt):
        return "weight"
    return None


async def _record_manual(guild, state, user_id: int, parsed: dict) -> str:
    await _append_record(
        guild, state, "meal",
        {"kcal": str(int(parsed["total"])), "manual": "1"},
        parsed["name"], user_id=user_id)
    lines = ["🍽️ 記録しました"]
    for name, kcal in parsed["items"]:
        lines.append(f"- {name}：{kcal} kcal" if kcal is not None
                     else f"- {name}")
    if len(parsed["items"]) > 1 or parsed["items"][0][1] is None:
        lines.append(f"合計 {parsed['total']} kcal")
    lines.append(_today_line(state, user_id))
    return "\n".join(lines)


async def try_handle(message: discord.Message, prompt: str) -> str | None:
    """Return a reply string if the message is a diet action, else None."""
    guild = message.guild
    if guild is None or _client is None:
        return None
    intent = detect_intent(prompt)
    if intent is None:
        return None

    state = await _ensure_loaded(guild)
    if state.channel_id is None:
        return SETUP_HINT
    user_id = message.author.id

    try:
        if intent == "manual":
            return await _record_manual(
                guild, state, user_id, parse_meal_input(prompt))

        if intent == "weight":
            kg = float(WEIGHT_RE.search(prompt).group(1))
            if not (20.0 <= kg <= 300.0):
                return "体重の数値が範囲外のようです。もう一度確認してください。"
            prev = state.latest_weight(user_id)
            await _append_record(guild, state, "weight",
                                 {"kg": f"{kg:.1f}"}, user_id=user_id)
            line = f"⚖️ 体重を記録しました： {kg:.1f} kg"
            if prev:
                diff = kg - _f(prev["fields"], "kg")
                line += f"（前回 {_f(prev['fields'], 'kg'):.1f} kg から {diff:+.1f} kg）"
            goal = state.goal(user_id)
            if goal:
                remain = kg - _f(goal["fields"], "target_kg")
                line += f"\n目標 {_f(goal['fields'], 'target_kg'):.1f} kg まで あと {remain:.1f} kg"
            return line

        # meal: calories may still be written out after the prefix
        # (「食べた：納豆ご飯 300kcal」) — record those as-is too.
        desc = MEAL_PREFIX_RE.sub("", prompt).strip(" ：:、") or prompt
        parsed = parse_meal_input(desc)
        if parsed:
            return await _record_manual(guild, state, user_id, parsed)
        est = await estimate_meal_text(desc)
        if est is None:
            return ("食事の内容を読み取れませんでした。"
                    "カロリーが分かれば「納豆ご飯 300kcal」のように書くと"
                    "そのまま記録できます。")
        totals = _totals(est["items"])
        name = est["meal_name"] or desc
        await _append_record(
            guild, state, "meal",
            {"kcal": f"{totals['kcal']:.0f}", "p": f"{totals['p']:.1f}",
             "f": f"{totals['f']:.1f}", "c": f"{totals['c']:.1f}"},
            name, user_id=user_id,
        )
        lines = [f"🍽️ 記録しました： {name}"]
        for item in est["items"]:
            lines.append(
                f"- {item['name']}： {item['kcal']:.0f} kcal"
                f"（P {item['p']:.0f} / F {item['f']:.0f} / C {item['c']:.0f}）"
            )
        lines.append(f"推定合計: {_fmt_pfc(totals)}")
        lines.append(_today_line(state, user_id))
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
                "使い方は /diet-goal で目標設定、/gohan で食事記録、"
                "「体重 68.4」でも記録できます。"
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
            "🥗 ダイエット記録をはじめます。ここに記録が保存されます（記録はユーザーごとです）。\n"
            "- 食事： /gohan が基本です。「マグロ200g 230kcal パックご飯180kcal」のように\n"
            "　複数まとめて書けます。「ご飯、納豆、オクラ（400kcal）」と合計だけでもOK\n"
            "　メンションで同じように書いても記録されます\n"
            "　カロリーが分からなければ品名だけで、AIが推定します\n"
            "- 体重： 「体重 68.4」のように送る\n"
            "- 目標： /diet-goal で目標体重と1日の目標カロリーを設定\n"
            "- 集計： /today（今日）・/week（直近7日）。集計は0時〜24時単位です\n"
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
        latest = state.latest_weight(interaction.user.id)
        if latest:
            fields["start_kg"] = f"{_f(latest['fields'], 'kg'):.1f}"
        await _append_record(guild, state, "goal", fields,
                             user_id=interaction.user.id)
        msg = f"🎯 目標を設定しました： {target_kg:.1f} kg"
        if daily_kcal:
            msg += f" / 1日 {int(daily_kcal)} kcal"
        if latest:
            msg += f"（現在 {_f(latest['fields'], 'kg'):.1f} kg、あと {(_f(latest['fields'], 'kg') - target_kg):.1f} kg）"
        await interaction.response.send_message(msg)

    @tree.command(name="gohan", description="食事を記録（カロリーを書けばそのまま、なければAI推定）")
    @app_commands.describe(
        text="食べたもの（例：マグロ200g 230kcal パックご飯180kcal／ご飯、納豆、オクラ（400kcal）／親子丼と味噌汁）")
    async def gohan_command(interaction: discord.Interaction, text: str):
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
        parsed = parse_meal_input(text)
        if parsed:
            reply = await _record_manual(
                guild, state, interaction.user.id, parsed)
            await interaction.followup.send(reply)
            return
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
                "食事の内容を読み取れませんでした。"
                "カロリーが分かれば「納豆ご飯 300kcal」の形式で記録できます。")
            return
        totals = _totals(est["items"])
        name = est["meal_name"] or text
        await _append_record(
            guild, state, "meal",
            {"kcal": f"{totals['kcal']:.0f}", "p": f"{totals['p']:.1f}",
             "f": f"{totals['f']:.1f}", "c": f"{totals['c']:.1f}"},
            name, user_id=interaction.user.id,
        )
        await interaction.followup.send(
            f"🍽️ 記録しました： {name}\n推定合計: {_fmt_pfc(totals)}\n"
            + _today_line(state, interaction.user.id)
            + "\n※カロリーはAIの推定です")

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
        prev = state.latest_weight(interaction.user.id)
        await _append_record(guild, state, "weight", {"kg": f"{kg:.1f}"},
                             user_id=interaction.user.id)
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
        await interaction.response.send_message(
            today_summary(state, interaction.user.id))

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
        await interaction.response.send_message(
            week_summary(state, interaction.user.id))

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
# Reminders (started from main.on_ready; one post per diet user)
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
        for user_id in state.users():
            try:
                if kind == "morning":
                    latest = state.latest_weight(user_id)
                    extra = (f"前回の記録は {_f(latest['fields'], 'kg'):.1f} kg でした。"
                             if latest else "")
                    await channel.send(
                        f"<@{user_id}> おはようございます☀️ 体重を測ったら"
                        "「体重 68.4」のように教えてください。" + extra)
                elif kind == "evening":
                    await channel.send(
                        f"<@{user_id}> 今日の記録もれはないですか？\n"
                        + today_summary(state, user_id))
                else:
                    await channel.send(
                        f"<@{user_id}> 1週間お疲れさまでした。\n"
                        + week_summary(state, user_id))
            except Exception as exc:  # noqa: BLE001
                print(f"Diet reminder failed for guild {guild.id}: {exc}")


def start_reminders():
    global _loops_started
    if _loops_started:
        return
    _loops_started = True
    _reminder_loop.start()
