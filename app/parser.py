"""Turn an exported group chat into a clean list of messages.

Supported: WhatsApp exports (Android and iPhone styles), Telegram result.json,
and plain 'Name: message' text. Nothing here talks to a model.
"""
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

DATE = r"(\d{1,2})[\/\.\-](\d{1,2})[\/\.\-](\d{2,4})"
TIME = r"(\d{1,2}):(\d{2})(?::\d{2})?\s*([AaPp]\.?[Mm]\.?)?"
ANDROID = re.compile(rf"^{DATE},?\s+{TIME}\s+-\s+(.*)$")
IPHONE = re.compile(rf"^\[{DATE},?\s+{TIME}\]\s+(.*)$")
PLAIN = re.compile(r"^([^:\n\[\]]{1,40}):\s+(.+)$")

MEDIA = re.compile(
    r"^(<media omitted>|<attached:.*>|(image|video|audio|sticker|gif|document|contact card|voice message)( omitted)?|"
    r"this message was deleted\.?|you deleted this message\.?|null|missed (voice|video) call)$", re.I)
SYSTEM_HINT = re.compile(
    r"(end-to-end encrypted|created group|added you|changed the (group|subject|icon)|joined using|left$|"
    r"security code changed|messages and calls)", re.I)


@dataclass
class Msg:
    idx: int
    sender: str
    text: str
    dt: Optional[datetime] = None
    stamp: str = ""  # original date/time text for display

    def when(self) -> str:
        return self.dt.strftime("%d %b, %H:%M") if self.dt else self.stamp


def _clean(line: str) -> str:
    return (line.replace("‎", "").replace("‏", "").replace(" ", " ")
            .replace(" ", " ").replace("﻿", "").rstrip("\r"))


def _to_dt(d, m, y, hh, mm, ap, day_first) -> Optional[datetime]:
    try:
        a, b, y = int(d), int(m), int(y)
        day, mon = (a, b) if day_first else (b, a)
        if y < 100:
            y += 2000
        h = int(hh)
        if ap:
            pm = ap.lower().startswith("p")
            h = h % 12 + (12 if pm else 0)
        return datetime(y, mon, day, h, int(mm))
    except (ValueError, TypeError):
        return None


def _guess_day_first(lines: List[str]) -> bool:
    first_gt12 = second_gt12 = False
    for ln in lines:
        m = ANDROID.match(ln) or IPHONE.match(ln)
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        first_gt12 |= a > 12
        second_gt12 |= b > 12
    if second_gt12 and not first_gt12:
        return False
    return True  # day-first is the default (also when the dates are ambiguous)


def _parse_telegram(text: str) -> Optional[List[Msg]]:
    try:
        data = json.loads(text)
    except ValueError:
        return None
    raw = data.get("messages") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return None
    out = []
    for m in raw:
        if not isinstance(m, dict) or m.get("type") not in (None, "message"):
            continue
        t = m.get("text", "")
        if isinstance(t, list):
            t = "".join(p if isinstance(p, str) else str(p.get("text", "")) for p in t if p)
        t = str(t or "").strip()
        sender = str(m.get("from") or m.get("actor") or "").strip()
        if not t or not sender:
            continue
        dt = None
        try:
            dt = datetime.fromisoformat(str(m.get("date", "")))
        except ValueError:
            pass
        out.append(Msg(0, sender, t, dt, str(m.get("date", ""))))
    return out


def parse_chat(text: str):
    """Returns (messages, skipped_count). Messages are numbered from 1."""
    text = text or ""
    skipped = 0
    msgs: List[Msg] = []
    if text.lstrip().startswith("{"):
        tg = _parse_telegram(text)
        if tg is not None:
            msgs = tg
    if not msgs:
        lines = [_clean(x) for x in text.split("\n")]
        day_first = _guess_day_first(lines)
        cur: Optional[Msg] = None
        cur_ts = False  # was the current message started by a timestamped line?
        for ln in lines:
            m = ANDROID.match(ln) or IPHONE.match(ln)
            if m:
                d, mo, y, hh, mm, ap, rest = m.groups()
                dt = _to_dt(d, mo, y, hh, mm, ap, day_first)
                stamp = f"{d}/{mo}/{y} {hh}:{mm}{(' ' + ap) if ap else ''}"
                if ": " in rest:
                    sender, body = rest.split(": ", 1)
                    cur = Msg(0, sender.strip(), body.strip(), dt, stamp)
                    cur_ts = True
                    msgs.append(cur)
                else:
                    cur = None  # system line such as 'X added Y'
                    skipped += 1
                continue
            if cur is not None and cur_ts and ln.strip():
                cur.text += "\n" + ln.strip()  # continuation of a multi-line message
                continue
            p = PLAIN.match(ln.strip()) if ln.strip() else None
            if p and not SYSTEM_HINT.search(ln):
                cur = Msg(0, p.group(1).strip(), p.group(2).strip(), None, "")
                cur_ts = False
                msgs.append(cur)
            elif cur is not None and ln.strip():
                cur.text += "\n" + ln.strip()  # continuation of a plain message
            elif ln.strip():
                skipped += 1
    kept = []
    for m in msgs:
        t = m.text.strip()
        if not t or MEDIA.match(t) or (SYSTEM_HINT.search(t) and len(t) < 90 and ":" not in t[:25]):
            skipped += 1
            continue
        m.text = t
        kept.append(m)
    for i, m in enumerate(kept, 1):
        m.idx = i
    return kept, skipped


def select_window(msgs: List[Msg], last_hours: Optional[float] = None, last_n: Optional[int] = None) -> List[Msg]:
    """Keep only the part of the chat the user cares about (e.g. 'since last night')."""
    out = msgs
    if last_hours and any(m.dt for m in msgs):
        dated = [m for m in msgs if m.dt]
        cutoff = max(m.dt for m in dated) - timedelta(hours=float(last_hours))
        out = [m for m in msgs if (m.dt is None) or m.dt >= cutoff]
    if last_n and last_n > 0:
        out = out[-int(last_n):]
    return out


def people(msgs: List[Msg]) -> List[str]:
    seen = []
    for m in msgs:
        if m.sender not in seen:
            seen.append(m.sender)
    return seen
