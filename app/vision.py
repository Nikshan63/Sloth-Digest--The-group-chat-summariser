"""Pictures and PDFs -> chat messages, using Gemma's vision locally.

A 4B vision model squashes every picture to about 896 px, so a whole desktop screenshot would turn
small chat text into mush. We cut the picture into readable tiles, read each tile, then stitch the
results together without duplicating the lines that two tiles (or two screenshots) both show.
"""
import base64
import difflib
import io
import math
import re
from typing import List, Tuple

from PIL import Image, ImageOps

from .llm import LLMError
from .parser import MEDIA

MAX_TILES = 6
MAX_PDF_PAGES = 12

SYSTEM_VISION = """You are an OCR assistant. The picture is part of a chat app screen (WhatsApp, Telegram or similar).
Copy the chat messages that are visible in the OPEN conversation, from top to bottom.
Ignore the chat list or sidebar, menus, buttons, the typing box, ads and the phone status bar.
Copy the text exactly as shown. Never invent, translate or fix text. If a message is cut off or unreadable, skip it.
For each message give the sender name shown above it. In a one-to-one chat, messages on the right side are from "Me" and
messages on the left are from the contact name shown at the top. If you cannot tell, leave sender empty.
Reply with JSON only: {"chat_name": "name at the top of the conversation, or empty",
"messages": [{"sender": "...", "time": "time shown next to the message, or empty", "text": "..."}]}
If the picture has no chat messages, reply {"chat_name": "", "messages": []}."""

UI_NOISE = re.compile(r"^(type a message|search or start new chat|online|typing\.\.\.|last seen.*|"
                      r"messages and calls are end-to-end encrypted.*|today|yesterday|unread messages?|click here.*)$", re.I)


# ---------------------------------------------------------------- tiling
def tile_image(data: bytes) -> List[bytes]:
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    w, h = img.size
    if w < 150 or h < 150:
        raise ValueError("That picture is too small to read.")
    if max(w, h) > 3400:
        k = 3400 / max(w, h)
        img = img.resize((int(w * k), int(h * k)), Image.LANCZOS)
        w, h = img.size
    boxes = []
    if w / h > 1.3:                       # wide (desktop window): split left to right
        n = min(MAX_TILES, math.ceil((w / h) / 1.3))
        step, ov = w / n, w / n * 0.10
        boxes = [(max(0, i * step - ov), 0, min(w, (i + 1) * step + ov), h) for i in range(n)]
    elif h / w > 1.3:                     # tall (phone screenshot): split top to bottom
        n = min(MAX_TILES, math.ceil((h / w) / 1.3))
        step, ov = h / n, h / n * 0.12
        boxes = [(0, max(0, i * step - ov), w, min(h, (i + 1) * step + ov)) for i in range(n)]
    else:
        boxes = [(0, 0, w, h)]
    out = []
    for b in boxes:
        t = img.crop(tuple(int(v) for v in b))
        if max(t.size) > 1024:
            k = 1024 / max(t.size)
            t = t.resize((max(1, int(t.size[0] * k)), max(1, int(t.size[1] * k))), Image.LANCZOS)
        buf = io.BytesIO()
        t.save(buf, "JPEG", quality=86)
        out.append(buf.getvalue())
    return out


# ---------------------------------------------------------------- cleaning + merging
def _s(x, n) -> str:
    return " ".join(str(x).split())[:n] if x is not None and not isinstance(x, (dict, list)) else ""


def clean_vision(raw) -> Tuple[str, List[dict]]:
    raw = raw if isinstance(raw, dict) else {}
    msgs = []
    for m in (raw.get("messages") if isinstance(raw.get("messages"), list) else []):
        if not isinstance(m, dict):
            continue
        text = _s(m.get("text"), 1500)
        if not text or UI_NOISE.match(text) or MEDIA.match(text):
            continue
        msgs.append({"sender": _s(m.get("sender"), 40), "time": _s(m.get("time"), 20), "text": text})
    out = []
    for m in msgs:                         # collapse an accidental repeat of the very same line
        if out and same(out[-1], m):
            continue
        out.append(m)
    return _s(raw.get("chat_name"), 60), out


def _key(m: dict) -> str:
    return re.sub(r"[^a-z0-9ऀ-ॿঀ-৿]+", "", (m["sender"] + m["text"]).lower())[:90]


def same(a: dict, b: dict) -> bool:
    ka, kb = _key(a), _key(b)
    if not ka or not kb:
        return False
    return ka == kb or difflib.SequenceMatcher(None, ka, kb).ratio() >= 0.88   # OCR is never exact


def merge_messages(existing: List[dict], new: List[dict]) -> List[dict]:
    """Add `new` (one more screenshot or tile) to `existing`, skipping lines already seen."""
    if not new:
        return existing
    if not existing:
        return list(new)
    # the new picture continues where the old one ended (scrolled down / newer messages)
    for k in range(min(len(existing), len(new)), 0, -1):
        if all(same(existing[len(existing) - k + i], new[i]) for i in range(k)):
            return existing + new[k:]
    # the new picture is older (scrolled up)
    for k in range(min(len(existing), len(new)), 0, -1):
        if all(same(existing[i], new[len(new) - k + i]) for i in range(k)):
            return new[:len(new) - k] + existing
    # overlap in the middle or none
    seen = [m for m in new if any(same(m, e) for e in existing)]
    if len(seen) >= max(1, len(new) // 2):
        return existing + [m for m in new if m not in seen]    # mostly known: add only the unseen lines
    return existing + new                                      # nothing in common: treat as later messages


# ---------------------------------------------------------------- model calls
def transcribe_image(llm, data: bytes) -> Tuple[str, List[dict]]:
    chat_name, merged = "", []
    for tile in tile_image(data):
        raw = llm.chat_json_images(SYSTEM_VISION, "Copy the chat messages in this picture.",
                                   [base64.b64encode(tile).decode()])
        name, msgs = clean_vision(raw)
        chat_name = chat_name or name
        merged = merge_messages(merged, msgs)
    return chat_name, merged


def pdf_extract(data: bytes):
    """('text', str) when the PDF has real text, else ('images', [png bytes per page])."""
    import pypdf
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        text = "\n".join((p.extract_text() or "") for p in reader.pages[:200])
    except Exception as e:  # a corrupt or encrypted PDF
        raise ValueError(f"I could not open that PDF ({type(e).__name__}).") from e
    if len(re.sub(r"\s+", "", text)) >= 200:
        return "text", text
    import pypdfium2 as pdfium
    try:
        pdf = pdfium.PdfDocument(data)
        pages = []
        for i in range(min(len(pdf), MAX_PDF_PAGES)):
            im = pdf[i].render(scale=1.6).to_pil().convert("RGB")
            buf = io.BytesIO()
            im.save(buf, "PNG")
            pages.append(buf.getvalue())
    except Exception as e:
        raise ValueError(f"I could not read the pages of that PDF ({type(e).__name__}).") from e
    if not pages:
        raise ValueError("That PDF has no pages.")
    return "images", pages
