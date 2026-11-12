"""
Diet support for the bot: per-user meal / weight logging, daily
calorie goals, daily & weekly summaries, reminders, and undo.

Logging a meal is deliberately low-effort: with the /gohan command,
write the food and its calories — 「納豆ご飯 300kcal」 — and the
numbers are recorded as-is. (Mentions are for questions, so diet
actions live only in slash commands.) Input is NFKC-normalised
first, and the parser accepts several items at once
(「マグロ200g 230kcal パックご飯180kcal」), a combined total in
parentheses (「ご飯、納豆、オクラ（400kcal）」), mixed known/unknown
items (known calories are taken as-is, the rest is estimated), and
half-/full-width variants. Meals are totalled per logical day
(see below) and every log reply reports the day's intake so
far. If the calories are unknown, Gemini estimates calories and
PFC instead (totals are always summed in Python).

Days run 04:00–04:00 JST, not midnight to midnight: anything
logged between 00:00 and 03:59 counts as the previous day, so a
late-night snack lands on the day it was eaten, not the next
morning.

Storage: there is no database. Each guild gets a text channel named
「ダイエット記録」. Meal / weight records are one bot message each:
a machine-readable first line (DIETLOG|1|...), tagged with the
user they belong to, plus a human-readable second line
(「🍽️ 12:30 名前： マグロ200g 230kcal・パックご飯 180kcal → 合計
410kcal」) so the channel reads as a shared multi-user log. The
loader only ever parses the first line. Settings (config + each user's goal) live
in ONE pinned message (DIETSET|1 + DIETLOG lines) that is edited in
place, so they never fall out of the history window. On startup the
bot preloads every guild's channel (pins + recent history) and
rebuilds its in-memory state, so logs survive redeploys and stay
visible (and editable) in Discord itself.

Backup: when GITHUB_TOKEN and DIET_BACKUP_REPO are configured, all
records are merged into a single diet_log.csv in that (private)
repo once a day at 21:00 — a safety net in case the Discord
channel is ever deleted. Rows are keyed by Discord message_id;
the window covered by in-memory records is rewritten from memory
(so /diet-undo deletions propagate) and older rows are kept.

Access: the log channel starts hidden from @everyone. Each user
who logs (or sets a goal) is granted read-only access to it, so
reminder mentions there actually reach them and the shared log
stays readable by its users. Reminder on/off is per user
(/diet-remind), stored as a 'pref' line in the settings message.
"""

import base64
import csv
import datetime
import io
import json
import os
import re
import unicodedata
import zoneinfo

import discord
import httpx
from discord import app_commands
from discord.ext import commands, tasks
from google.genai import types

JST = zoneinfo.ZoneInfo("Asia/Tokyo")
# A diet "day" starts at 04:00 JST; 00:00–03:59 belongs to the
# previous day (late-night snacks, pre-dawn weigh-ins).
DAY_BOUNDARY_HOUR = 4


def _logical_date(ts: datetime.datetime) -> datetime.date:
    return (ts - datetime.timedelta(hours=DAY_BOUNDARY_HOUR)).date()


def _today_logical() -> datetime.date:
    return _logical_date(datetime.datetime.now(JST))


LOG_CHANNEL_NAME = "ダイエット記録"
RECORD_PREFIX = "DIETLOG|1|"
SETTINGS_HEADER = "DIETSET|1"

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
BACKUP_REPO = os.environ.get("DIET_BACKUP_REPO")  # "owner/repo"
BACKUP_PATH = "diet_log.csv"
BACKUP_FIELDS = [
    "timestamp", "user_id", "type", "name", "kcal", "protein_g",
    "fat_g", "carbs_g", "kg", "target_kg", "daily_kcal", "manual",
    "guild_id", "message_id",
]

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
    # The record text is the first line only; anything after a
    # newline is the human-readable display line.
    text = text.split("\n", 1)[0]
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
        self.settings_msg_id: int | None = None
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

    def pref_record(self, user_id: int) -> dict | None:
        prefs = self._mine(
            [r for r in self.records if r["type"] == "pref"], user_id)
        return prefs[-1] if prefs else None

    def remind_pref(self, user_id: int) -> bool:
        """Per-user reminder preference (default on). Stored as a
        'pref' record in the settings message; the latest wins."""
        pref = self.pref_record(user_id)
        if pref is None:
            return True
        return pref["fields"].get("remind", "on") != "off"

    def meals_on(self, day: datetime.date, user_id: int) -> list[dict]:
        return self._mine([
            r for r in self.records
            if r["type"] == "meal" and _logical_date(r["ts"]) == day
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


def _parse_settings(content: str) -> list[dict]:
    records = []
    for line in content.splitlines()[1:]:
        record = decode_record(line.strip())
        if record is not None:
            records.append(record)
    return records


_REMINDER_MARKERS = (
    ("おはようございます☀️", "morning"),
    ("今日の記録もれはないですか", "evening"),
    ("今日はまだ食事の記録がありません", "evening"),
    ("1週間お疲れさまでした", "sunday-week"),
)


def _seed_reminded(guild_id: int, msg):
    """If a reminder post from earlier today is found in the log
    channel (e.g. after a restart minutes later), mark that kind as
    already sent so it is not posted twice."""
    created = getattr(msg, "created_at", None)
    if created is None:
        return
    jst = created.astimezone(JST)
    if jst.date() != datetime.datetime.now(JST).date():
        return
    for marker, kind in _REMINDER_MARKERS:
        if marker in msg.content:
            if kind == "evening" and jst.weekday() == 6:
                kind = "sunday-week"
            _reminded.add((guild_id, jst.date().isoformat(), kind))
            return


async def _ensure_loaded(guild: discord.Guild) -> GuildDiet:
    """Locate the log channel and (once) rebuild state: settings
    from the pinned DIETSET message, meal/weight records from the
    recent channel history."""
    state = _state(guild.id)
    if state.loaded:
        return state
    channel = _find_log_channel(guild)
    if channel is None:
        state.loaded = True  # no setup yet; nothing to load
        return state
    state.channel_id = channel.id
    records: list[dict] = []
    settings_found = False
    bot_id = _bot.user.id if _bot is not None and _bot.user else None

    def _apply_settings(msg) -> bool:
        nonlocal settings_found
        parsed = _parse_settings(msg.content)
        records.extend(parsed)
        state.settings_msg_id = msg.id
        settings_found = True
        return True

    if bot_id is not None:
        try:
            pins = await channel.pins()
        except Exception:  # noqa: BLE001
            pins = []
        for msg in pins:
            if msg.author.id == bot_id and msg.content.startswith(SETTINGS_HEADER):
                _apply_settings(msg)
                break
    async for msg in channel.history(limit=1000, oldest_first=True):
        if bot_id is None or msg.author.id != bot_id:
            continue
        _seed_reminded(guild.id, msg)
        if msg.content.startswith(SETTINGS_HEADER):
            if not settings_found:
                _apply_settings(msg)
            continue
        record = decode_record(msg.content)
        if record is None:
            continue
        if record["type"] in ("meal", "weight"):
            record["msg_id"] = msg.id
            records.append(record)
        elif not settings_found:
            # Legacy goal/config posted as plain messages (pre-pin
            # format): accept them until settings are saved once.
            record["msg_id"] = msg.id
            records.append(record)
    records.sort(key=lambda r: r["ts"])
    state.records = records
    state.loaded = True
    return state


def _push_local(state: GuildDiet, rtype: str, fields: dict,
                user_id: int | None = None) -> dict:
    """Add a record to in-memory state only (used for goal/config,
    whose persistence is the settings message, not a chat post)."""
    if user_id is not None:
        fields = {"user": str(user_id), **fields}
    now = datetime.datetime.now(JST)
    record = decode_record(encode_record(rtype, now, fields))
    state.records.append(record)
    return record


async def _save_settings(guild: discord.Guild, state: GuildDiet):
    """Rewrite the pinned settings message from current state:
    one config line (merged) + each user's latest goal line and
    reminder-preference line."""
    channel = guild.get_channel(state.channel_id)
    if channel is None:
        return
    now = datetime.datetime.now(JST)
    lines = [SETTINGS_HEADER]
    config = state.config()
    if config:
        lines.append(encode_record("config", now, config))
    for user_id in state.users():
        goal = state.goal(user_id)
        if goal is not None:
            lines.append(encode_record(
                "goal", goal["ts"], goal["fields"], goal["text"]))
        pref = state.pref_record(user_id)
        if pref is not None:
            lines.append(encode_record(
                "pref", pref["ts"], pref["fields"], pref["text"]))
    content = "\n".join(lines)
    if state.settings_msg_id is not None:
        try:
            message = channel.get_partial_message(state.settings_msg_id)
            await message.edit(content=content)
            return
        except Exception as exc:  # noqa: BLE001
            print(f"Settings edit failed, recreating: {exc}")
            state.settings_msg_id = None
    message = await channel.send(content)
    state.settings_msg_id = message.id
    try:
        await message.pin()
    except Exception as exc:  # noqa: BLE001
        print(f"Settings pin failed (message still in history): {exc}")


async def _append_record(guild: discord.Guild, state: GuildDiet,
                         rtype: str, fields: dict, text: str = "",
                         user_id: int | None = None,
                         display: str | None = None) -> dict:
    """Post a meal/weight record as its own message in the log
    channel (one message per record, items joined in the text).
    `display`, when given, is appended as a human-readable second
    line so the channel reads as a shared log; the loader ignores
    everything after the first line."""
    channel = guild.get_channel(state.channel_id)
    if channel is None:
        raise RuntimeError("log channel not found")
    if user_id is not None:
        fields = {"user": str(user_id), **fields}
    now = datetime.datetime.now(JST)
    content = encode_record(rtype, now, fields, text)
    if display:
        content += "\n" + _clean_field(display)
    message = await channel.send(content)
    record = decode_record(content)
    record["msg_id"] = message.id
    state.records.append(record)
    return record


# ---------------------------------------------------------------------------
# Meal estimation (Gemini, JSON out; totals are summed in Python)
# Used only for items whose calories the user did not write.
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
# Summaries (per user; days are logical days, 04:00–04:00 JST)
# ---------------------------------------------------------------------------
def _fmt_pfc(t: dict) -> str:
    return (f"{t['kcal']:.0f} kcal"
            f"（P {t['p']:.0f}g / F {t['f']:.0f}g / C {t['c']:.0f}g）")


def _today_line(state: GuildDiet, user_id: int) -> str:
    """The standing status line appended after every log entry."""
    today = _today_logical()
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
    today = _today_logical()
    meals = state.meals_on(today, user_id)
    lines = [f"📊 今日（{today:%m/%d}）の記録"]
    if meals:
        est_total = {"kcal": 0.0, "p": 0.0, "f": 0.0, "c": 0.0}
        has_manual = False
        for meal in meals:
            kcal = _f(meal["fields"], "kcal")
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
    today = _today_logical()
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
# Manual meal input parsing (used by the /gohan command)
#
# Input is NFKC-normalised first, so full-width digits / spaces /
# parentheses and the squared-kcal character (㎉) all become their
# standard half-width forms before parsing.
# ---------------------------------------------------------------------------
_UNIT_RE = r"(?:kcal|キロカロリー|cal)"
_KCAL_TOKEN_RE = re.compile(
    r"(\d{1,4}(?:\.\d+)?)\s*" + _UNIT_RE, re.IGNORECASE)
_PAREN_TOTAL_RE = re.compile(
    r"[(]\s*(\d{1,4}(?:\.\d+)?)\s*" + _UNIT_RE + r"\s*[)]\s*$",
    re.IGNORECASE)
_FILLER_SUFFIX_RE = re.compile(r"(を食べた|食べた|でした|です|だよ|だね|のみ|だけ)$")
_ITEM_SEPARATORS = " 　、・,，"


def _normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    # Newlines act as item separators, like 、.
    return t.replace("\r\n", "、").replace("\n", "、").replace("\r", "、")


def _clean_item_name(text: str) -> str:
    return text.strip(_ITEM_SEPARATORS)


_AMOUNT_NUMBER = r"(?:\d+(?:\.\d+)?|\d+/\d+)"
_AMOUNT_UNITS = (r"(?:g|kg|ml|l|個|枚|本|切れ|パック|人前|食分|杯|皿|"
                 r"粒|袋|缶|束|玉|丁|尾|匹|かけ|片)")
# A chunk that is only an amount: 「100g」「2個」「大さじ1」「小さじ2」
# 「少々」... Spoon measures count as amounts of the previous food.
_AMOUNT_CHUNK_RE = re.compile(
    r"^(?:" + _AMOUNT_NUMBER + r")\s*" + _AMOUNT_UNITS + r"$"
    r"|^(?:大さじ|小さじ|カップ)(?:" + _AMOUNT_NUMBER + r")$"
    r"|^(?:少々|適量|ひとつまみ)$", re.IGNORECASE)
# A chunk that ends with an amount glued to a name: 「アジ100g」.
_GLUED_AMOUNT_RE = re.compile(
    r"^(?P<name>.+?)(?P<amount>" + _AMOUNT_NUMBER + _AMOUNT_UNITS +
    r"|(?:大さじ|小さじ|カップ)" + _AMOUNT_NUMBER + r")$",
    re.IGNORECASE)
_SPOON_WORDS = ("大さじ", "小さじ", "カップ")
_MODIFIER_CHUNKS = ("大盛り", "特盛", "小盛り", "並盛", "半分",
                    "少なめ", "多め")


def _split_items(segment: str) -> list[str]:
    """Split a space-separated run into food items.

    An item is a name plus its optional amount. An amount chunk
    (「100g」「大さじ1」「2個」) closes the item it follows; a
    chunk with the amount glued on (「アジ100g」) is a complete
    item by itself; anything else continues the current name.
    「マグロ 100g アジ 100g パックご飯」 →
    ['マグロ100g', 'アジ100g', 'パックご飯'], while
    「ご飯 納豆 オクラ」 (no amounts) stays one item.
    """
    parts = [p for p in re.split(r"\s+", (segment or "").strip()) if p]
    items: list[str] = []
    current = ""
    pending_spoon = False
    for part in parts:
        if pending_spoon and re.fullmatch(_AMOUNT_NUMBER, part):
            current += part
            items.append(current)
            current = ""
            pending_spoon = False
            continue
        pending_spoon = False
        if part in _SPOON_WORDS:
            current += part
            pending_spoon = True
            continue
        if _AMOUNT_CHUNK_RE.match(part):
            current += part
            items.append(current)
            current = ""
            continue
        if part in _MODIFIER_CHUNKS:
            current += part
            continue
        if _GLUED_AMOUNT_RE.match(part):
            if current:
                items.append(current)
            items.append(part)
            current = ""
            continue
        current = (current + " " + part) if current else part
    if current:
        items.append(current)
    return items


def _scan_items(t: str):
    """Scan kcal tokens in normalised text.

    Returns (items, residue): items is a list of (name, kcal) where
    the name is the text since the previous token. A token with an
    empty name segment (e.g. the leading 0kcal of 「0kcalゼリー」)
    is part of a name, not a calorie value, and is skipped.
    """
    items = []
    name_start = 0
    for token in _KCAL_TOKEN_RE.finditer(t):
        name = _clean_item_name(t[name_start:token.start()])
        if not name:
            continue
        items.append((name, int(round(float(token.group(1))))))
        name_start = token.end()
    residue = _FILLER_SUFFIX_RE.sub(
        "", t[name_start:].strip(_ITEM_SEPARATORS)).strip()
    return items, residue


def parse_meal_input(text: str) -> dict | None:
    """Parse fully-specified meal input into {name, total, items}.

    items is a list of (item name, kcal or None). Returns None when
    the text carries no usable calories or has unexplained residue
    (such messages are questions or mixed input, not pure logs).
    """
    if not text:
        return None
    t = _normalize(text).strip()
    if t.endswith(("？", "?")):
        return None
    paren = _PAREN_TOTAL_RE.search(t)
    if paren:
        body = t[:paren.start()]
        # The parenthesised total only counts when the body itself
        # carries no kcal values (otherwise it is ambiguous with
        # per-item values or flavour notes like （ハーブ）).
        if body.strip() and not _KCAL_TOKEN_RE.search(body):
            total = int(round(float(paren.group(1))))
            name = _clean_item_name(body)
            parts = [p.strip() for p in re.split(r"[、・,，]", name)
                     if p.strip()]
            return {"name": name, "total": total,
                    "items": [(part, None) for part in parts]}
    items, residue = _scan_items(t)
    if not items or residue:
        return None
    # A segment that itemises into several foods (「マグロ 100g
    # アジ 100g パックご飯」) means some foods carry no value of
    # their own — that is mixed input, not a pure manual log.
    for name, _kcal in items:
        sub = []
        for part in re.split(r"[、・,，]", name):
            sub.extend(_split_items(part))
        if len(sub) > 1:
            return None
    return {"name": "、".join(name for name, _ in items),
            "total": sum(kcal for _, kcal in items),
            "items": items}


def parse_partial_input(text: str) -> dict | None:
    """Parse mixed input: some items carry calories, others don't.

    Returns {manual_items, estimate_text} or None. Within a
    、-separated run before a kcal value, only the last part is the
    valued item's name; the earlier parts (and any tail after the
    last value) are handed to the estimator.
    """
    if not text:
        return None
    t = _normalize(text).strip()
    if t.endswith(("？", "?")) or _PAREN_TOTAL_RE.search(t):
        return None
    manual_items = []
    estimate_parts = []
    name_start = 0
    for token in _KCAL_TOKEN_RE.finditer(t):
        segment = _clean_item_name(t[name_start:token.start()])
        if not segment:
            continue
        parts = [p.strip() for p in re.split(r"[、・,，]", segment)
                 if p.strip()]
        # Within the last run, itemise by amounts: only the final
        # item owns the kcal value; earlier items go to estimation.
        last_items = _split_items(parts[-1]) if parts else [segment]
        if len(parts) > 1:
            for part in parts[:-1]:
                estimate_parts.extend(_split_items(part))
        estimate_parts.extend(last_items[:-1])
        manual_items.append(
            (last_items[-1], int(round(float(token.group(1))))))
        name_start = token.end()
    tail = _FILLER_SUFFIX_RE.sub(
        "", t[name_start:].strip(_ITEM_SEPARATORS)).strip()
    if tail:
        estimate_parts.append(tail)
    if not manual_items or not estimate_parts:
        return None
    return {"manual_items": manual_items,
            "estimate_text": "、".join(estimate_parts)}


RATE_LIMIT_MESSAGE = (
    "⚠️ いま少し混み合っています（Geminiの無料枠の上限に達しました）。"
    "1分ほど待ってからもう一度試してください。"
)
SETUP_HINT = (
    "ダイエット記録はまだセットアップされていません。"
    "先に /diet-setup を実行してください（記録専用チャンネルを作ります）。"
)


async def _record_manual(guild, state, user_id: int, parsed: dict,
                         user_name: str = "") -> str:
    now = datetime.datetime.now(JST)
    who = f"{user_name}：" if user_name else ""
    item_text = "・".join(
        f"{name} {kcal}kcal" if kcal is not None else name
        for name, kcal in parsed["items"])
    display = f"🍽️ {now:%H:%M} {who}{item_text}"
    if len(parsed["items"]) > 1 or parsed["items"][0][1] is None:
        display += f" → 合計 {parsed['total']}kcal"
    await _append_record(
        guild, state, "meal",
        {"kcal": str(int(parsed["total"])), "manual": "1"},
        parsed["name"], user_id=user_id, display=display)
    lines = ["🍽️ 記録しました"]
    for name, kcal in parsed["items"]:
        lines.append(f"- {name}：{kcal} kcal" if kcal is not None
                     else f"- {name}")
    if len(parsed["items"]) > 1 or parsed["items"][0][1] is None:
        lines.append(f"合計 {parsed['total']} kcal")
    lines.append(_today_line(state, user_id))
    return "\n".join(lines)


async def _record_estimated(guild, state, user_id: int, est: dict,
                            manual_items: list | None = None,
                            user_name: str = "") -> str:
    """Record a meal whose calories were (partly) estimated.
    manual_items, when given, are (name, kcal) pairs already known;
    they are added on top of the estimate."""
    totals = _totals(est["items"])
    manual_total = sum(kcal for _, kcal in (manual_items or []))
    total_kcal = totals["kcal"] + manual_total
    names = [name for name, _ in (manual_items or [])]
    if est["meal_name"]:
        names.append(est["meal_name"])
    name = "、".join(names) if names else "食事"
    now = datetime.datetime.now(JST)
    who = f"{user_name}：" if user_name else ""
    suffix = "（一部推定）" if manual_items else "（推定）"
    display = f"🍽️ {now:%H:%M} {who}{name} {total_kcal:.0f}kcal{suffix}"
    await _append_record(
        guild, state, "meal",
        {"kcal": f"{total_kcal:.0f}", "p": f"{totals['p']:.1f}",
         "f": f"{totals['f']:.1f}", "c": f"{totals['c']:.1f}"},
        name, user_id=user_id, display=display,
    )
    lines = ["🍽️ 記録しました"]
    for item_name, kcal in (manual_items or []):
        lines.append(f"- {item_name}：{kcal} kcal（確定）")
    for item in est["items"]:
        lines.append(
            f"- {item['name']}： {item['kcal']:.0f} kcal（推定）"
            f"（P {item['p']:.0f} / F {item['f']:.0f} / C {item['c']:.0f}）"
        )
    lines.append(f"合計: {_fmt_pfc({'kcal': total_kcal, 'p': totals['p'], 'f': totals['f'], 'c': totals['c']})}")
    lines.append(_today_line(state, user_id))
    lines.append("※推定分のカロリーはAIの推定です")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Daily backup to a GitHub repo (Contents API, merge by record key)
# ---------------------------------------------------------------------------
def _record_row(record: dict, guild_id: int) -> list:
    fields = record["fields"]
    return [
        record["ts"].isoformat(), fields.get("user", ""), record["type"],
        record["text"], fields.get("kcal", ""), fields.get("p", ""),
        fields.get("f", ""), fields.get("c", ""), fields.get("kg", ""),
        fields.get("target_kg", ""), fields.get("daily_kcal", ""),
        fields.get("manual", ""), str(guild_id),
        str(record.get("msg_id") or ""),
    ]


def _parse_csv_rows(text: str) -> list[list]:
    """Data rows of an existing backup CSV, normalised to the
    current field count (older files may lack message_id)."""
    rows = [row for row in csv.reader(io.StringIO(text)) if row]
    if rows and rows[0] and rows[0][0] == "timestamp":
        rows = rows[1:]
    return [row + [""] * (len(BACKUP_FIELDS) - len(row)) for row in rows]


def _csv_text(rows: list[list]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(BACKUP_FIELDS)
    writer.writerows(rows)
    return buffer.getvalue()


async def _gh_get_file(client: httpx.AsyncClient):
    resp = await client.get(
        f"https://api.github.com/repos/{BACKUP_REPO}/contents/{BACKUP_PATH}")
    if resp.status_code == 404:
        return None, None
    resp.raise_for_status()
    data = resp.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    return content, data["sha"]


async def _gh_put_file(client: httpx.AsyncClient, content: str,
                       sha: str | None):
    payload = {
        "message": f"Diet log backup {datetime.datetime.now(JST):%Y-%m-%d}",
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
    }
    if sha:
        payload["sha"] = sha
    resp = await client.put(
        f"https://api.github.com/repos/{BACKUP_REPO}/contents/{BACKUP_PATH}",
        json=payload)
    resp.raise_for_status()


async def backup_now() -> str:
    """Merge all in-memory records into the backup CSV (once per
    day from the reminder loop). Returns a short status string.

    Merge rules (message_id is the primary key):
    - goal rows are always refreshed from current state;
    - meal/weight rows whose message_id is in memory are re-added
      from memory;
    - rows inside the time window covered by memory (from the
      oldest in-memory record of that guild on) but NOT in memory
      are dropped — that is how /diet-undo deletions propagate;
    - older rows outside the window are kept untouched."""
    if not GITHUB_TOKEN or not BACKUP_REPO:
        return "disabled"
    all_records = []
    window_start: dict[str, datetime.datetime] = {}
    for state in _states.values():
        if not state.loaded:
            continue
        if state.channel_id is not None and not any(
                r["type"] in ("meal", "weight") for r in state.records):
            # Set up but currently holding no records at all: every
            # backed-up meal/weight row of this guild was undone.
            window_start[str(state.guild_id)] = \
                datetime.datetime.min.replace(tzinfo=JST)
        for record in state.records:
            if record["type"] in ("meal", "weight", "goal"):
                all_records.append((record, state.guild_id))
                if record["type"] in ("meal", "weight"):
                    key = str(state.guild_id)
                    if key not in window_start \
                            or record["ts"] < window_start[key]:
                        window_start[key] = record["ts"]
    all_records.sort(key=lambda pair: pair[0]["ts"])
    memory_rows = [_record_row(r, gid) for r, gid in all_records]
    memory_ids = {row[13] for row in memory_rows if row[13]}
    goal_keys = {(row[12], row[1]) for row in memory_rows
                 if row[2] == "goal"}
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "discord-gemini-bot",
    }
    async with httpx.AsyncClient(timeout=20, headers=headers) as client:
        for attempt in range(2):
            existing, sha = await _gh_get_file(client)
            if existing is None:
                await _gh_put_file(client, _csv_text(memory_rows), None)
                return f"created with {len(memory_rows)} records"
            kept = []
            for row in _parse_csv_rows(existing):
                if row[2] == "goal":
                    if (row[12], row[1]) not in goal_keys:
                        kept.append(row)
                    continue
                if row[13] and row[13] in memory_ids:
                    continue  # re-added from memory below
                start = window_start.get(row[12])
                if start is not None:
                    try:
                        row_ts = datetime.datetime.fromisoformat(row[0])
                    except ValueError:
                        row_ts = None
                    if row_ts is not None and row_ts >= start:
                        continue  # in covered window but gone = undone
                kept.append(row)
            merged = kept + memory_rows
            merged.sort(key=lambda row: row[0])
            merged_text = _csv_text(merged)
            if merged_text == _csv_text(_parse_csv_rows(existing)):
                return "up-to-date"
            try:
                await _gh_put_file(client, merged_text, sha)
                return f"saved {len(merged)} records"
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in (409, 422) and attempt == 0:
                    continue  # sha moved; re-read and merge again
                raise
    return "failed"


async def _ensure_user_access(guild: discord.Guild, state: GuildDiet,
                              user) -> None:
    """Give a diet user read access to the log channel: reminders
    mention users there and the shared log is meant to be readable
    by the people logging in it. Read-only (no posting). Best
    effort — without permission to edit channel overwrites this
    quietly does nothing and admins are unaffected."""
    try:
        if state.channel_id is None:
            return
        channel = guild.get_channel(state.channel_id)
        if channel is None:
            return
        if channel.permissions_for(user).view_channel:
            return
        await channel.set_permissions(
            user, view_channel=True, read_message_history=True,
            send_messages=False)
    except Exception as exc:  # noqa: BLE001
        print(f"Diet log access grant failed: {exc}")


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
                "/weight で体重記録ができます。"
            )
            return
        overwrites = {
            guild.default_role: discord.PermissionOverwrite(read_messages=False),
            guild.me: discord.PermissionOverwrite(
                read_messages=True, send_messages=True,
                read_message_history=True, manage_messages=True,
                manage_permissions=True
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
        _push_local(state, "config",
                    {"setup_by": str(interaction.user.id), "remind": "on"})
        await _save_settings(guild, state)
        await channel.send(
            "🥗 ダイエット記録をはじめます。ここに記録が保存されます（記録はユーザーごとです）。\n"
            "- 食事： /gohan で記録します。「マグロ200g 230kcal パックご飯180kcal」のように\n"
            "　複数まとめて書けます。「ご飯、納豆、オクラ（400kcal）」と合計だけでもOK\n"
            "　分かる品にはカロリーを付け、分からない品はそのまま書けばAIが推定します\n"
            "- 体重： /weight で記録します\n"
            "- 目標： /diet-goal で目標体重と1日の目標カロリーを設定\n"
            "- 集計： /today（今日）・/week（直近7日）。1日は朝4時〜翌朝4時で集計します（深夜4時までの夜食は前日分）\n"
            "- 取り消し： /diet-undo で直前の食事・体重の記録を1件削除できます\n"
            "- リマインド： 朝8時に体重、夜21時に記録もれの声かけ（/diet-remind で自分の分だけ切替）\n"
            "- このチャンネルは記録した人にだけ自動で閲覧権限が付きます（見るだけ・投稿はできません）"
        )
        await _ensure_user_access(guild, state, interaction.user)
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
        _push_local(state, "goal", fields, user_id=interaction.user.id)
        await _save_settings(guild, state)
        await _ensure_user_access(guild, state, interaction.user)
        msg = f"🎯 目標を設定しました： {target_kg:.1f} kg"
        if daily_kcal:
            msg += f" / 1日 {int(daily_kcal)} kcal"
        if latest:
            msg += f"（現在 {_f(latest['fields'], 'kg'):.1f} kg、あと {(_f(latest['fields'], 'kg') - target_kg):.1f} kg）"
        await interaction.response.send_message(msg)

    @tree.command(name="gohan", description="食事を記録（カロリーを書けばそのまま、分からない品はAI推定）")
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
        await _ensure_user_access(guild, state, interaction.user)
        await interaction.response.defer(thinking=True)
        parsed = parse_meal_input(text)
        if parsed:
            reply = await _record_manual(
                guild, state, interaction.user.id, parsed,
                user_name=interaction.user.display_name)
            await interaction.followup.send(reply)
            return
        partial = parse_partial_input(text)
        try:
            if partial:
                est = await estimate_meal_text(partial["estimate_text"])
                if est is None:
                    await interaction.followup.send(
                        "分からない品の推定ができませんでした。"
                        "カロリーの分かる品と分からない品は、分けて入力してください。")
                    return
                reply = await _record_estimated(
                    guild, state, interaction.user.id, est,
                    manual_items=partial["manual_items"],
                    user_name=interaction.user.display_name)
                await interaction.followup.send(reply)
                return
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
        reply = await _record_estimated(
            guild, state, interaction.user.id, est,
            user_name=interaction.user.display_name)
        await interaction.followup.send(reply)

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
        await _ensure_user_access(guild, state, interaction.user)
        prev = state.latest_weight(interaction.user.id)
        now = datetime.datetime.now(JST)
        display = (f"⚖️ {now:%H:%M} {interaction.user.display_name}："
                   f"体重 {kg:.1f}kg")
        if prev:
            display += f"（前回比 {kg - _f(prev['fields'], 'kg'):+.1f}）"
        await _append_record(guild, state, "weight", {"kg": f"{kg:.1f}"},
                             user_id=interaction.user.id, display=display)
        line = f"⚖️ 体重を記録しました： {kg:.1f} kg"
        if prev:
            line += f"（前回 {_f(prev['fields'], 'kg'):.1f} kg から {kg - _f(prev['fields'], 'kg'):+.1f} kg）"
        goal = state.goal(interaction.user.id)
        if goal:
            line += f"\n目標 {_f(goal['fields'], 'target_kg'):.1f} kg まで あと {kg - _f(goal['fields'], 'target_kg'):.1f} kg"
        await interaction.response.send_message(line)

    @tree.command(name="diet-undo", description="直前の食事・体重の記録を1件取り消す")
    async def diet_undo(interaction: discord.Interaction):
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
        user_id = interaction.user.id
        candidates = [
            r for r in state.records
            if r["type"] in ("meal", "weight")
            and r["fields"].get("user") == str(user_id)
        ]
        if not candidates:
            await interaction.response.send_message(
                "取り消せる記録がありません。", ephemeral=True)
            return
        record = candidates[-1]
        channel = guild.get_channel(state.channel_id)
        if record.get("msg_id") and channel is not None:
            try:
                await channel.get_partial_message(record["msg_id"]).delete()
            except Exception as exc:  # noqa: BLE001
                print(f"Undo message delete failed (continuing): {exc}")
        state.records.remove(record)
        if record["type"] == "meal":
            desc = (f"{record['text']} {_f(record['fields'], 'kcal'):.0f} kcal"
                    f"（{record['ts'].astimezone(JST):%m/%d %H:%M}）")
            reply = f"↩️ 直前の食事記録を取り消しました： {desc}\n" \
                    + _today_line(state, user_id)
        else:
            desc = (f"体重 {_f(record['fields'], 'kg'):.1f} kg"
                    f"（{record['ts'].astimezone(JST):%m/%d %H:%M}）")
            reply = f"↩️ 直前の体重記録を取り消しました： {desc}"
        await interaction.response.send_message(reply)

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

    @tree.command(name="diet-remind", description="自分の朝・夜リマインドのオン/オフ")
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
        _push_local(state, "pref", {"remind": on_off},
                    user_id=interaction.user.id)
        await _save_settings(guild, state)
        await interaction.response.send_message(
            f"あなたのリマインドを {on_off} にしました。"
            "（他の人の設定には影響しません）")


# ---------------------------------------------------------------------------
# Reminders (preloaded and started from main.on_ready)
# ---------------------------------------------------------------------------
_reminded: set[tuple[int, str, str]] = set()
_backed_up_dates: set[str] = set()
_loops_started = False
_preload_started = False


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
    today = _logical_date(now)
    for guild in list(_bot.guilds):
        key = (guild.id, today_key, kind)
        if key in _reminded:
            continue
        state = await _ensure_loaded(guild)
        if state.channel_id is None:
            continue
        channel = guild.get_channel(state.channel_id)
        if channel is None:
            continue
        _reminded.add(key)
        for user_id in state.users():
            if not state.remind_pref(user_id):
                continue
            try:
                if kind == "morning":
                    latest = state.latest_weight(user_id)
                    extra = (f"前回の記録は {_f(latest['fields'], 'kg'):.1f} kg でした。"
                             if latest else "")
                    await channel.send(
                        f"<@{user_id}> おはようございます☀️ 体重を測ったら"
                        "/weight で記録してください。" + extra)
                elif kind == "evening":
                    if state.meals_on(today, user_id):
                        await channel.send(
                            f"<@{user_id}> 今日の記録もれはないですか？\n"
                            + today_summary(state, user_id))
                    else:
                        await channel.send(
                            f"<@{user_id}> 今日はまだ食事の記録がありません。"
                            "記録し忘れはありませんか？")
                else:  # sunday-week: today's summary AND the week, one post
                    await channel.send(
                        f"<@{user_id}> 1週間お疲れさまでした。\n"
                        + today_summary(state, user_id)
                        + "\n\n" + week_summary(state, user_id))
            except Exception as exc:  # noqa: BLE001
                print(f"Diet reminder failed for guild {guild.id}: {exc}")
    if kind in ("evening", "sunday-week") \
            and today_key not in _backed_up_dates:
        _backed_up_dates.add(today_key)
        try:
            result = await backup_now()
            print(f"Diet backup: {result}")
        except Exception as exc:  # noqa: BLE001
            print(f"Diet backup failed: {exc}")


async def preload_and_start():
    """Load every guild's diet state up front (so the first slash
    command of the day answers fast and the 8:00 reminder works
    even if nobody has run a command since the last restart),
    then start the reminder loop."""
    global _preload_started
    if _preload_started:
        return
    _preload_started = True
    if _bot is not None:
        for guild in list(_bot.guilds):
            try:
                await _ensure_loaded(guild)
            except Exception as exc:  # noqa: BLE001
                print(f"Diet preload failed for guild {guild.id}: {exc}")
    start_reminders()


def start_reminders():
    global _loops_started
    if _loops_started:
        return
    _loops_started = True
    _reminder_loop.start()
