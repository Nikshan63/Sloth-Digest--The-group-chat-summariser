"""History of past digests, stored only on this computer in data/history.json (git-ignored).

It holds the digest (headline, to-dos, dates, questions and their receipt messages), never the
raw chat file and never screenshots. The user can switch it off, delete one entry or clear all.
"""
import json
import re
import threading
import uuid
from datetime import datetime
from typing import List, Optional

from . import config

_lock = threading.Lock()
MAX_ENTRIES = 200


def _path():
    return config.DATA_DIR / "history.json"


def _load() -> List[dict]:
    try:
        d = json.loads(_path().read_text())
        return d if isinstance(d, list) else []
    except (OSError, ValueError):
        return []


def _save(items: List[dict]) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _path().with_suffix(".tmp")
    tmp.write_text(json.dumps(items))
    tmp.replace(_path())


def title_from(source: str, kind: str = "text") -> str:
    s = re.sub(r"\.(txt|json|pdf|png|jpe?g|webp)$", "", (source or "").strip(), flags=re.I)
    m = re.search(r"WhatsApp Chat (?:with|-)\s+(.+)", s, flags=re.I)
    if m:
        return m.group(1).strip()[:80]
    if s:
        return s[:80]
    return {"live": "Live scan", "image": "Screenshots", "pdf": "PDF"}.get(kind, "Pasted chat")


def add(result: dict, source: str = "", kind: str = "text", entry_id: Optional[str] = None) -> str:
    """Save a digest. Re-using entry_id updates that entry (a live scan keeps one entry)."""
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    with _lock:
        items = _load()
        eid = entry_id or uuid.uuid4().hex[:12]
        old = next((x for x in items if x["id"] == eid), None)
        entry = {"id": eid, "created_at": old["created_at"] if old else now, "updated_at": now,
                 "title": title_from(source, kind), "kind": kind, "result": result}
        items = [entry] + [x for x in items if x["id"] != eid]
        _save(items[:MAX_ENTRIES])
        return eid


def summaries() -> List[dict]:
    out = []
    for x in _load():
        r = x["result"]
        out.append({"id": x["id"], "created_at": x["created_at"], "updated_at": x.get("updated_at", x["created_at"]),
                    "title": x["title"], "kind": x["kind"], "mode": r.get("mode", ""),
                    "headline": (r.get("headline") or [""])[0],
                    "counts": {"messages": r["stats"]["messages"], "actions": len(r.get("actions", [])),
                               "dates": len(r.get("dates", [])), "questions": len(r.get("questions", []))},
                    "minutes_saved": r["stats"].get("minutes_saved", 0)})
    return out


def get(entry_id: str) -> Optional[dict]:
    return next((x for x in _load() if x["id"] == entry_id), None)


def delete(entry_id: str) -> bool:
    with _lock:
        items = _load()
        kept = [x for x in items if x["id"] != entry_id]
        if len(kept) == len(items):
            return False
        _save(kept)
        return True


def clear() -> int:
    with _lock:
        n = len(_load())
        _save([])
        return n
