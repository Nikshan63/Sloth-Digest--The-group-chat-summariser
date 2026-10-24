import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, history, live, stats
from .digest import run_digest
from .llm import LLMError, OllamaLLM
from .parser import parse_chat, select_window

app = FastAPI(title="Sloth Digest")
STATIC = Path(__file__).resolve().parent.parent / "static"
MAX_CHARS = 6_000_000


def get_llm():
    return OllamaLLM()


class DigestRequest(BaseModel):
    text: str
    last_hours: Optional[float] = None
    last_n: Optional[int] = None
    allow_fallback: bool = True
    save_history: bool = True
    source: str = ""      # file name or label, used for the history title
    kind: str = "text"


def _prepare(req: DigestRequest):
    if len(req.text) > MAX_CHARS:
        raise HTTPException(413, "That chat file is too large (limit about 6 MB of text).")
    msgs, skipped = parse_chat(req.text)
    if not msgs:
        raise HTTPException(400, "I could not find any messages. Export the chat from WhatsApp (Export chat > Without media) "
                                 "or paste lines like 'Name: message'. Telegram result.json also works.")
    msgs = select_window(msgs, req.last_hours, req.last_n)
    if not msgs:
        raise HTTPException(400, "No messages fall inside that time window. Try 'Everything'.")
    return msgs, skipped


def _run(req: DigestRequest, progress=None) -> dict:
    msgs, skipped = _prepare(req)
    try:
        result = run_digest(msgs, get_llm(), skipped, progress, req.allow_fallback)
    except LLMError as e:
        raise HTTPException(502, str(e))
    stats.record(result)
    if req.save_history:
        result["history_id"] = history.add(result, req.source, req.kind)
    return result


@app.get("/api/health")
def health():
    llm = get_llm()
    url = getattr(llm, "url", config.OLLAMA_URL)
    model = getattr(llm, "model", config.OLLAMA_MODEL)
    up, has_model = False, False
    try:
        r = httpx.get(f"{url}/api/tags", timeout=2)
        up = r.status_code == 200
        names = [m.get("name", "") for m in r.json().get("models", [])]
        has_model = any(n == model or (":" not in model and n.split(":")[0] == model) for n in names)
    except (httpx.HTTPError, ValueError, AttributeError):
        pass
    return {"ok": True, "model": model, "ollama_running": up, "model_ready": has_model}


@app.post("/api/digest")
def digest(req: DigestRequest):
    return _run(req)


# ---- background jobs so the page can show progress. Results live in memory only. ----
JOBS: dict = {}
JLOCK = threading.Lock()


def _purge():
    cutoff = time.time() - 1800
    for k in [k for k, v in JOBS.items() if v["t"] < cutoff]:
        JOBS.pop(k, None)


@app.post("/api/jobs")
def start_job(req: DigestRequest):
    _prepare(req)  # fail fast with a clear message
    jid = uuid.uuid4().hex[:12]
    with JLOCK:
        _purge()
        JOBS[jid] = {"state": "running", "stage": "Starting slowly...", "done": 0, "total": 1, "t": time.time()}

    def progress(stage, done, total):
        JOBS[jid].update(stage=stage, done=done, total=total)

    def work():
        try:
            JOBS[jid]["result"] = _run(req, progress)
            JOBS[jid]["state"] = "done"
        except HTTPException as e:
            JOBS[jid].update(state="error", error=str(e.detail))
        except Exception as e:  # never leave the page waiting forever
            JOBS[jid].update(state="error", error=f"{type(e).__name__}: {e}")

    threading.Thread(target=work, daemon=True).start()
    return {"job": jid}


@app.get("/api/jobs/{jid}")
def job(jid: str):
    j = JOBS.get(jid)
    if not j:
        raise HTTPException(404, "Unknown or expired job.")
    return j


@app.get("/api/stats")
def get_stats():
    return stats.snapshot()


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


# ---------------------------------------------------------------- history
@app.get("/api/history")
def history_list():
    return {"items": history.summaries()}


@app.get("/api/history/{eid}")
def history_get(eid: str):
    e = history.get(eid)
    if not e:
        raise HTTPException(404, "That history entry no longer exists.")
    return e


@app.delete("/api/history/{eid}")
def history_delete(eid: str):
    if not history.delete(eid):
        raise HTTPException(404, "That history entry no longer exists.")
    return {"deleted": eid}


@app.delete("/api/history")
def history_clear():
    return {"deleted": history.clear()}


# ---------------------------------------------------------------- pictures, PDFs and live scan
MAX_PICTURE_BYTES = 25 * 1024 * 1024


class StartReq(BaseModel):
    kind: str = "upload"


@app.post("/api/live/start")
def live_start(req: StartReq):
    return {"session": live.create(req.kind).id}


def _session(sid: str) -> "live.Session":
    s = live.get(sid)
    if not s:
        raise HTTPException(404, "That scan session has expired. Start a new one.")
    return s


@app.post("/api/live/{sid}/image")
async def live_image(sid: str, request: Request):
    s = _session(sid)
    data = await request.body()
    kind = "pdf" if request.headers.get("x-kind") == "pdf" else "image"
    if not data:
        raise HTTPException(400, "Empty upload.")
    if len(data) > MAX_PICTURE_BYTES:
        raise HTTPException(413, "That file is too large (limit 25 MB).")
    if s.state != "running":
        raise HTTPException(409, "This scan has already been stopped.")
    with s.lock:
        if s.busy:
            return {"accepted": False, "reason": "busy"}   # still reading the previous picture: skip, do not queue
        s.busy = True
    llm = get_llm()
    threading.Thread(target=live.process_picture, args=(s, llm, data, kind), daemon=True).start()
    return {"accepted": True}


@app.get("/api/live/{sid}")
def live_status(sid: str):
    return _session(sid).status()


@app.post("/api/live/{sid}/digest")
def live_digest(sid: str):
    s = _session(sid)
    if not s.messages:
        raise HTTPException(400, "Nothing has been read yet.")
    with s.lock:
        if s.busy:
            return {"started": False, "reason": "busy"}
        s.busy = True

    def work():
        try:
            live.digest_now(s, get_llm())
        except Exception as e:
            s.error = f"{type(e).__name__}: {e}"
        finally:
            s.busy = False

    threading.Thread(target=work, daemon=True).start()
    return {"started": True}


class StopReq(BaseModel):
    save_history: bool = True
    source: str = ""


@app.post("/api/live/{sid}/stop")
def live_stop(sid: str, req: StopReq):
    s = _session(sid)
    if s.state != "running":
        return {"state": s.state}
    s.state = "finishing"
    threading.Thread(target=live.finish, args=(s, get_llm(), req.save_history, req.source), daemon=True).start()
    return {"state": "finishing"}
