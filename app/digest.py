"""Chat -> digest. The model proposes; code checks every claim against the real messages.

Every action item, date and question must point at a real message number. The user
sees that original message as the 'receipt', so a made-up item cannot slip through.
"""
import hashlib
import re
import time
from typing import Callable, Dict, List, Optional, Set

from . import config
from .llm import LLMError
from .parser import Msg, people

STOP = set("""the and for that this with you your are was were will have has had not but can all any from they them
their there then than what when where who how why just about out get got one two too also its our let lets would could
should may might been being into over under again very much more some such only own same other each few most both
than yes yeah okay ok hey hii hello please pls thanks thank bro guys""".split())

WORD = re.compile(r"[A-Za-z0-9ঀ-৿ऀ-ॿ']+")
NUM = re.compile(r"\d+")


def toks(text: str) -> Set[str]:
    return {w for w in (x.lower().strip("'") for x in WORD.findall(text or "")) if len(w) >= 3 and w not in STOP}


def nums(text: str) -> Set[str]:
    return set(NUM.findall(text or ""))


def when_ok(when: str, window: str) -> bool:
    """A date/deadline phrase is trusted only if its numbers AND its words are really in the chat text."""
    if not nums(when) <= nums(window):
        return False
    words = {t for t in toks(when) if not t.isdigit()}
    return not words or len(words & toks(window)) / len(words) >= 0.5


def jaccard(a: Set[str], b: Set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


SYSTEM_MAP = """You read ONE slice of a group chat. Every line looks like: [number] Sender: message
Extract only what is literally in the messages. Do not guess, do not add advice.
Reply with JSON only, in exactly this shape:
{
 "points": [{"text": "one short sentence about something that happened or was decided", "ids": [message numbers]}],
 "actions": [{"who": "person who must do it, or empty", "task": "what must be done", "due": "deadline words from the chat, or empty", "id": message number}],
 "dates": [{"what": "event or deadline", "when": "date or time words from the chat", "id": message number}],
 "questions": [{"who": "who asked", "question": "the question", "id": message number}]
}
Rules:
- "id" must be the number of the message that says it. Never invent a number.
- Use names exactly as they appear as senders. If nobody is named, leave "who" empty.
- Only include real requests, promises or deadlines as actions. Skip jokes and small talk.
- Never write a number, date or name that is not in the messages.
- Keep each sentence under 20 words. Empty lists are fine."""

SYSTEM_HEADLINE = """You get bullet points from a group chat. Write a two-line catch-up for someone who missed it.
Reply with JSON only: {"lines": ["first line", "second line"]}
Each line is one plain sentence under 25 words. Use only facts from the bullet points. Do not add numbers or names that are not in them."""


# ---------- helpers to survive messy model output ----------
def _as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def _s(x) -> str:
    return " ".join(str(x).split()) if x is not None and not isinstance(x, (dict, list)) else ""


def _id(x) -> Optional[int]:
    m = re.search(r"\d+", str(x)) if x is not None else None
    return int(m.group(0)) if m else None


def _ids(x) -> List[int]:
    out = []
    for v in _as_list(x):
        i = _id(v)
        if i is not None:
            out.append(i)
    return out


def clean_raw(raw: dict) -> dict:
    """Normalise whatever shape the model returned into the shape we expect."""
    raw = raw if isinstance(raw, dict) else {}
    points = []
    for p in _as_list(raw.get("points")):
        if isinstance(p, dict):
            points.append({"text": _s(p.get("text") or p.get("point")), "ids": _ids(p.get("ids") or p.get("id"))})
        else:
            points.append({"text": _s(p), "ids": []})
    actions = [{"who": _s(a.get("who")), "task": _s(a.get("task") or a.get("action")), "due": _s(a.get("due")), "id": _id(a.get("id"))}
               for a in _as_list(raw.get("actions")) if isinstance(a, dict)]
    dates = [{"what": _s(d.get("what")), "when": _s(d.get("when")), "id": _id(d.get("id"))}
             for d in _as_list(raw.get("dates")) if isinstance(d, dict)]
    questions = [{"who": _s(q.get("who")), "question": _s(q.get("question")), "id": _id(q.get("id"))}
                 for q in _as_list(raw.get("questions")) if isinstance(q, dict)]
    return {"points": points, "actions": actions, "dates": dates, "questions": questions}


# ---------- guardrails ----------
class Grounder:
    def __init__(self, msgs: List[Msg]):
        self.by_id: Dict[int, Msg] = {m.idx: m for m in msgs}
        self.people = people(msgs)
        self.chat_nums = nums(" ".join(m.text for m in msgs))

    def window(self, i: int, around: int) -> str:
        return " ".join(self.by_id[j].text for j in range(i - around, i + around + 1) if j in self.by_id)

    def receipt(self, i: int) -> dict:
        m = self.by_id[i]
        return {"id": i, "sender": m.sender, "when": m.when(), "text": m.text[:600]}

    def person(self, who: str, fallback: str = "") -> str:
        who = (who or "").strip()
        if not who:
            return fallback
        low = who.lower()
        for p in self.people:
            if p.lower() == low or p.lower().split()[0] == low.split()[0]:
                return p
        return fallback

    def action(self, a) -> Optional[dict]:
        i = a["id"]
        if i not in self.by_id or not a["task"]:
            return None
        if not (toks(a["task"]) & toks(self.window(i, 2))):
            return None  # the task shares no words with the message it points at
        if not nums(a["task"]) <= nums(self.window(i, 3)):
            return None  # a number the chat never had
        due = a["due"]
        if due and not when_ok(due, self.window(i, 3)):
            due = ""  # deadline words that are not in the chat are dropped, the task stays
        who = self.person(a["who"])
        return {"who": who or "Unassigned", "task": a["task"][:200], "due": due[:60], "receipt": self.receipt(i)}

    def date(self, d) -> Optional[dict]:
        i = d["id"]
        if i not in self.by_id or not d["what"] or not d["when"]:
            return None
        if not when_ok(d["when"], self.window(i, 1)):
            return None
        return {"what": d["what"][:140], "when": d["when"][:60], "receipt": self.receipt(i)}

    def question(self, q) -> Optional[dict]:
        i = q["id"]
        if i not in self.by_id or not q["question"]:
            return None
        text = self.by_id[i].text
        if "?" not in text and not re.search(r"\b(who|what|when|where|why|how|can|could|any)\b", text, re.I):
            return None
        if not (toks(q["question"]) & toks(text)):
            return None
        return {"who": self.by_id[i].sender, "question": q["question"][:200], "receipt": self.receipt(i)}

    def point(self, p) -> Optional[dict]:
        if not p["text"] or not nums(p["text"]) <= self.chat_nums:
            return None
        ids = [i for i in p["ids"] if i in self.by_id]
        pool = " ".join(self.by_id[i].text for i in ids) if ids else " ".join(m.text for m in self.by_id.values())
        if not (toks(p["text"]) & toks(pool)):
            return None
        return {"text": p["text"][:220], "ids": ids}


def _dedupe(items: List[dict], key: str, limit: int) -> List[dict]:
    out: List[dict] = []
    for it in items:
        t = toks(it[key])
        if any(jaccard(t, toks(o[key])) >= 0.6 for o in out):
            continue
        out.append(it)
    return out[:limit]


# ---------- chunking ----------
def chunk_messages(msgs: List[Msg]) -> List[List[Msg]]:
    chunks, cur, size = [], [], 0
    for m in msgs:
        line = _line(m)
        if cur and size + len(line) > config.CHUNK_CHARS:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(m)
        size += len(line) + 1
    if cur:
        chunks.append(cur)
    return chunks


def _line(m: Msg) -> str:
    t = m.text.replace("\n", " ")[: config.MAX_MSG_CHARS_IN_PROMPT]
    return f"[{m.idx}] {m.sender}: {t}"


# ---------- no-AI fallback ----------
ACTION_RX = re.compile(r"\b(please|pls|can you|could you|need to|needs to|have to|must|don't forget|dont forget|remember to|"
                       r"i'll|i will|will do|submit|send|bring|share|upload|prepare|finish|complete)\b", re.I)
DATE_RX = re.compile(r"\b(today|tonight|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
                     r"\d{1,2}(:\d{2})?\s?(am|pm)|\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?|deadline|due)\b", re.I)


def fallback_digest(msgs: List[Msg], g: Grounder) -> dict:
    actions, dates, questions = [], [], []
    for m in msgs:
        if ACTION_RX.search(m.text) and DATE_RX.search(m.text) or (ACTION_RX.search(m.text) and len(actions) < 6):
            actions.append({"who": "Unassigned", "task": m.text.split("\n")[0][:160], "due": (DATE_RX.search(m.text) or [None])[0] or "",
                            "receipt": g.receipt(m.idx)})
        elif DATE_RX.search(m.text) and re.search(r"deadline|due|meet|exam|submit|class|call|demo", m.text, re.I):
            dates.append({"what": m.text.split("\n")[0][:120], "when": DATE_RX.search(m.text).group(0), "receipt": g.receipt(m.idx)})
        if m.text.strip().endswith("?"):
            questions.append({"who": m.sender, "question": m.text.split("\n")[0][:160], "receipt": g.receipt(m.idx)})
    return {"actions": _dedupe(actions, "task", 10), "dates": _dedupe(dates, "what", 6), "questions": _dedupe(questions, "question", 5)}


# ---------- main entry ----------
def run_digest(msgs: List[Msg], llm, skipped: int = 0, progress: Optional[Callable] = None,
               allow_fallback: bool = True, cache: Optional[dict] = None) -> dict:
    t0 = time.time()
    progress = progress or (lambda *a: None)
    notes: List[str] = []
    g = Grounder(msgs)
    chunks = chunk_messages(msgs)
    if len(chunks) > config.MAX_CHUNKS:
        notes.append(f"Long chat: only the latest {config.MAX_CHUNKS} slices were read ({len(chunks) - config.MAX_CHUNKS} older slices skipped).")
        chunks = chunks[-config.MAX_CHUNKS:]
    mode = "ai"
    points: List[dict] = []
    actions: List[dict] = []
    dates: List[dict] = []
    questions: List[dict] = []
    dropped = 0
    failed = 0
    last_err = ""
    for n, ch in enumerate(chunks, 1):
        progress(f"Reading slice {n} of {len(chunks)}", n - 1, len(chunks) + 1)
        user = "Participants: " + ", ".join(people(ch)) + "\n\nChat slice:\n" + "\n".join(_line(m) for m in ch)
        try:
            ck = hashlib.sha1(user.encode()).hexdigest()
            if cache is not None and ck in cache:
                raw = cache[ck]  # this exact slice was already read earlier in the live session
            else:
                raw = clean_raw(llm.chat_json(SYSTEM_MAP, user))
                if cache is not None:
                    cache[ck] = raw
        except LLMError as e:
            failed += 1
            last_err = str(e)
            if n == 1 and failed == 1 and not allow_fallback:
                raise
            continue
        for key, fn, bucket in (("points", g.point, points), ("actions", g.action, actions),
                                ("dates", g.date, dates), ("questions", g.question, questions)):
            for item in raw[key]:
                ok = fn(item)
                if ok:
                    bucket.append(ok)
                else:
                    dropped += 1
    if failed == len(chunks):
        if not allow_fallback:
            raise LLMError(last_err or "The model could not read the chat.")
        mode = "fallback"
        notes.append(f"AI unavailable ({last_err}). Showing a simpler keyword-based digest instead.")
        fb = fallback_digest(msgs, g)
        actions, dates, questions, points = fb["actions"], fb["dates"], fb["questions"], []
    elif failed:
        notes.append(f"{failed} of {len(chunks)} slices could not be read, so the digest may be missing things.")
    actions = _dedupe(sorted(actions, key=lambda a: a["receipt"]["id"]), "task", 12)
    dates = _dedupe(sorted(dates, key=lambda d: d["receipt"]["id"]), "what", 8)
    questions = _dedupe(sorted(questions, key=lambda q: q["receipt"]["id"]), "question", 6)
    points = _dedupe(points, "text", 14)

    progress("Writing the two-line catch-up", len(chunks), len(chunks) + 1)
    headline: List[str] = []
    if mode == "ai" and points:
        try:
            raw = llm.chat_json(SYSTEM_HEADLINE, "Bullet points:\n" + "\n".join("- " + p["text"] for p in points))
            for ln in _as_list(raw.get("lines") if isinstance(raw, dict) else None)[:2]:
                ln = _s(ln)
                if ln and nums(ln) <= g.chat_nums and len(ln) <= 220:
                    headline.append(ln)
        except LLMError:
            pass
        if not headline:
            headline = [p["text"] for p in points[:2]]
            notes.append("The headline model call failed or was not trusted, so the top points are shown.")
    if not headline:
        ppl = len(g.people)
        headline = [f"{len(msgs)} messages from {ppl} {'person' if ppl == 1 else 'people'}.",
                    f"{len(actions)} things to do, {len(dates)} dates, {len(questions)} questions found."]

    words_read = sum(len(m.text.split()) for m in msgs)
    digest_text = " ".join(headline + [a["task"] for a in actions] + [d["what"] + " " + d["when"] for d in dates]
                           + [q["question"] for q in questions])
    min_read = words_read / config.WORDS_PER_MINUTE
    min_digest = len(digest_text.split()) / config.WORDS_PER_MINUTE
    dated = [m for m in msgs if m.dt]
    return {
        "mode": mode, "model": getattr(llm, "model", "none"),
        "headline": headline, "points": [p["text"] for p in points],
        "actions": actions, "dates": dates, "questions": questions, "notes": notes,
        "stats": {
            "messages": len(msgs), "people": len(g.people), "skipped": skipped, "chunks": len(chunks),
            "from": dated[0].when() if dated else "", "to": dated[-1].when() if dated else "",
            "words_read": words_read, "minutes_to_read": round(min_read, 1), "minutes_digest": round(min_digest, 1),
            "minutes_saved": round(max(0.0, min_read - min_digest), 1),
            "dropped_by_guardrails": dropped, "seconds_taken": round(time.time() - t0, 1),
        },
    }
