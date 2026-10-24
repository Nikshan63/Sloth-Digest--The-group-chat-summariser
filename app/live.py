"""Live scan and screenshot sessions. Everything here lives in memory only.

A session collects messages read from pictures (a screen share frame, uploaded screenshots,
or a scanned PDF), keeps a running digest, and is thrown away after a few hours.
Pictures themselves are never stored: they are read and dropped.
"""
import threading
import time
import uuid
from typing import Dict, List, Optional

from . import history, stats
from .digest import run_digest
from .llm import LLMError
from .parser import Msg, parse_chat
from .vision import merge_messages, pdf_extract, transcribe_image

MAX_MESSAGES = 2500
TTL_SECONDS = 6 * 3600
AUTO_DIGEST_MIN_NEW = 3
AUTO_DIGEST_MIN_GAP = 20


class Session:
    def __init__(self, kind: str):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind                      # "live" or "upload"
        self.created = time.time()
        self.messages: List[dict] = []
        self.chat_name = ""
        self.scans = 0
        self.busy = False
        self.error = ""
        self.last_seconds = 0.0
        self.last_added = 0
        self.digest: Optional[dict] = None
        self.digest_ts = 0.0
        self.new_since_digest = 0
        self.cache: dict = {}
        self.state = "running"                # running -> finishing -> finished
        self.saved_id: Optional[str] = None
        self.stage = ""
        self.lock = threading.Lock()

    # ---- data
    def add(self, chat_name: str, msgs: List[dict]) -> int:
        with self.lock:
            before = len(self.messages)
            self.messages = merge_messages(self.messages, msgs)[-MAX_MESSAGES:]
            added = max(0, len(self.messages) - before)
            if chat_name and not self.chat_name:
                self.chat_name = chat_name
            self.new_since_digest += added
            self.last_added = added
            return added

    def transcript(self) -> str:
        return "\n".join(f"{m['sender'] or 'Unknown'}: {m['text']}" for m in self.messages)

    def as_msgs(self) -> List[Msg]:
        return [Msg(i, m["sender"] or "Unknown", m["text"], None, m["time"]) for i, m in enumerate(self.messages, 1)]

    def status(self) -> dict:
        return {"id": self.id, "kind": self.kind, "state": self.state, "busy": self.busy, "error": self.error,
                "scans": self.scans, "messages": len(self.messages), "last_added": self.last_added,
                "last_seconds": round(self.last_seconds, 1), "chat_name": self.chat_name, "stage": self.stage,
                "digest": self.digest, "digest_ts": self.digest_ts, "saved_id": self.saved_id,
                "transcript": self.transcript() if self.kind == "upload" else ""}


SESSIONS: Dict[str, Session] = {}
_slock = threading.Lock()


def create(kind: str) -> Session:
    s = Session(kind if kind in ("live", "upload") else "upload")
    with _slock:
        cutoff = time.time() - TTL_SECONDS
        for k in [k for k, v in SESSIONS.items() if v.created < cutoff]:
            SESSIONS.pop(k, None)
        SESSIONS[s.id] = s
    return s


def get(sid: str) -> Optional[Session]:
    return SESSIONS.get(sid)


# ---------------------------------------------------------------- workers (run in a thread)
def digest_now(s: Session, llm) -> None:
    msgs = s.as_msgs()
    if not msgs:
        return
    s.stage = "Writing the digest"
    s.digest = run_digest(msgs, llm, 0, lambda st, d, n: setattr(s, "stage", st), True, cache=s.cache)
    s.digest_ts = time.time()
    s.new_since_digest = 0
    s.stage = ""


def process_picture(s: Session, llm, data: bytes, kind: str) -> None:
    """kind: 'image' or 'pdf'. Always clears s.busy at the end."""
    t0 = time.time()
    try:
        s.error = ""
        s.stage = "Reading the picture with Gemma vision"
        if kind == "pdf":
            how, payload = pdf_extract(data)
            if how == "text":
                parsed, _ = parse_chat(payload)
                s.add("", [{"sender": m.sender, "time": m.when(), "text": m.text} for m in parsed])
            else:
                for n, page in enumerate(payload, 1):
                    s.stage = f"Reading page {n} of {len(payload)}"
                    name, msgs = transcribe_image(llm, page)
                    s.add(name, msgs)
        else:
            name, msgs = transcribe_image(llm, data)
            s.add(name, msgs)
        s.scans += 1
        if s.kind == "live" and s.messages and s.new_since_digest >= (AUTO_DIGEST_MIN_NEW if s.digest else 1) \
                and time.time() - s.digest_ts >= AUTO_DIGEST_MIN_GAP:
            digest_now(s, llm)
    except (LLMError, ValueError) as e:
        s.error = str(e)
    except Exception as e:  # never leave the page waiting on a crashed worker
        s.error = f"{type(e).__name__}: {e}"
    finally:
        s.last_seconds = time.time() - t0
        s.stage = ""
        s.busy = False


def finish(s: Session, llm, save_history: bool, source: str) -> None:
    try:
        s.state = "finishing"
        s.stage = "Finishing up"
        if s.messages and (s.digest is None or s.new_since_digest > 0):
            digest_now(s, llm)
        if s.digest:
            stats.record(s.digest)
            if save_history:
                s.saved_id = history.add(s.digest, source or s.chat_name or "", "live" if s.kind == "live" else "image", s.id)
    except (LLMError, ValueError) as e:
        s.error = str(e)
    except Exception as e:
        s.error = f"{type(e).__name__}: {e}"
    finally:
        s.stage = ""
        s.busy = False
        s.state = "finished"
