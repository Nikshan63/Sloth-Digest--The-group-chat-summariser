"""Counts only. Chat text, names and receipts are never written to disk."""
import json
import threading

from . import config

_lock = threading.Lock()
_ZERO = {"digests": 0, "messages_read": 0, "minutes_saved": 0.0, "seconds_total": 0.0, "ai_runs": 0, "fallback_runs": 0}


def load() -> dict:
    try:
        d = json.loads(config.STATS_FILE.read_text())
        return {**_ZERO, **{k: d.get(k, v) for k, v in _ZERO.items()}}
    except (OSError, ValueError):
        return dict(_ZERO)


def record(result: dict) -> None:
    s = result["stats"]
    with _lock:
        d = load()
        d["digests"] += 1
        d["messages_read"] += int(s["messages"])
        d["minutes_saved"] = round(d["minutes_saved"] + float(s["minutes_saved"]), 1)
        d["seconds_total"] = round(d["seconds_total"] + float(s["seconds_taken"]), 1)
        d["ai_runs" if result["mode"] == "ai" else "fallback_runs"] += 1
        try:
            config.DATA_DIR.mkdir(parents=True, exist_ok=True)
            tmp = config.STATS_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(d))
            tmp.replace(config.STATS_FILE)
        except OSError:
            pass  # stats are a nicety, never a reason to fail a digest


def snapshot() -> dict:
    d = load()
    d["avg_seconds"] = round(d["seconds_total"] / d["digests"], 1) if d["digests"] else 0
    return d
