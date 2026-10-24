import io
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app import config, history, live, main
from app.parser import parse_chat
from app.vision import clean_vision, merge_messages, pdf_extract, tile_image
from tests.test_digest import SAMPLE, FakeLLM, tmp_data  # noqa: F401  (tmp_data keeps tests off the real data dir)


def M(sender, text):
    return {"sender": sender, "time": "", "text": text}


def png(w=700, h=700, text="hello", fmt="PNG"):
    im = Image.new("RGB", (w, h), "white")
    ImageDraw.Draw(im).text((20, 20), text, fill="black")
    b = io.BytesIO()
    im.save(b, fmt)
    return b.getvalue()


class VisionLLM(FakeLLM):
    """Stand-in for Gemma vision: returns scripted pages, one per call."""
    def __init__(self, pages, **kw):
        super().__init__([], **kw)
        self.pages, self.i, self.seen_images = pages, 0, 0

    def chat_json_images(self, system, user, images):
        self.seen_images += len(images)
        p = self.pages[min(self.i, len(self.pages) - 1)]
        self.i += 1
        return p

    def chat_json(self, system, user):
        # digest calls: build an honest reply from whatever the session holds
        self.calls += 1
        from app.digest import SYSTEM_HEADLINE
        if system == SYSTEM_HEADLINE:
            return {"lines": ["Sam asked for the ER diagram."]}
        return {"points": [], "actions": [], "dates": [], "questions": []}


@pytest.fixture
def client(monkeypatch):
    return TestClient(main.app)


# ---------------- tiling + merging
def test_tile_wide_tall_and_small():
    assert len(tile_image(png(2400, 900))) >= 2
    assert len(tile_image(png(600, 1800))) >= 2
    assert len(tile_image(png(700, 700))) == 1
    with pytest.raises(ValueError):
        tile_image(png(50, 50))
    for t in tile_image(png(3000, 800)):
        assert max(Image.open(io.BytesIO(t)).size) <= 1024


def test_merge_scroll_down_up_and_ocr_noise():
    a = [M("A", "one is here"), M("B", "two is here"), M("A", "three is here")]
    down = [M("B", "two is here"), M("A", "three is here"), M("B", "four is here")]
    assert [m["text"] for m in merge_messages(a, down)] == ["one is here", "two is here", "three is here", "four is here"]
    up = [M("B", "zero is here"), M("A", "one is here"), M("B", "two is here")]
    assert [m["text"] for m in merge_messages(a, up)][0] == "zero is here"
    noisy = [M("B", "twо is here"), M("A", "three is herе")]   # look-alike letters
    assert len(merge_messages(a, noisy)) == 3
    assert merge_messages(a, a) == a


def test_clean_vision_drops_noise_and_junk():
    name, msgs = clean_vision({"chat_name": "Team", "messages": [
        {"sender": "A", "text": "hi"}, {"sender": "A", "text": "hi"}, {"sender": "A", "text": "Type a message"},
        {"sender": "A", "text": "<Media omitted>"}, "junk", {"sender": 5, "text": ""}]})
    assert name == "Team" and [m["text"] for m in msgs] == ["hi"]
    assert clean_vision("nonsense") == ("", [])


# ---------------- PDF
def test_pdf_text_and_scanned():
    from reportlab.pdfgen import canvas
    b = io.BytesIO()
    c = canvas.Canvas(b)
    for i, line in enumerate(SAMPLE.splitlines()[:40]):
        c.drawString(30, 800 - i * 14, line[:95])
    c.save()
    how, payload = pdf_extract(b.getvalue())
    assert how == "text" and "Riya" in payload
    img = Image.new("RGB", (600, 800), "white")
    buf = io.BytesIO()
    img.save(buf, "PDF")
    how, pages = pdf_extract(buf.getvalue())
    assert how == "images" and len(pages) == 1
    with pytest.raises(ValueError):
        pdf_extract(b"not a pdf")


# ---------------- history
def test_history_add_list_get_delete_clear_cap(monkeypatch):
    res = {"mode": "ai", "headline": ["H"], "actions": [{"x": 1}], "dates": [], "questions": [], "points": [],
           "stats": {"messages": 5, "minutes_saved": 2}}
    a = history.add(res, "WhatsApp Chat with Study Group.txt", "text")
    b = history.add(res, "", "live", "fixed")
    history.add({**res, "headline": ["H2"]}, "", "live", "fixed")        # upsert, not duplicate
    items = history.summaries()
    assert len(items) == 2 and items[0]["id"] == "fixed" and items[0]["headline"] == "H2"
    assert next(i for i in items if i["id"] == a)["title"] == "Study Group"
    assert items[0]["created_at"] and items[0]["counts"]["actions"] == 1
    assert history.get(a)["result"]["headline"] == ["H"]
    assert history.delete(a) and not history.delete(a)
    assert history.clear() == 1 and history.summaries() == []
    monkeypatch.setattr(history, "MAX_ENTRIES", 3)
    for i in range(6):
        history.add(res, f"c{i}", "text")
    assert len(history.summaries()) == 3


def test_history_api_and_optout(client, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "get_llm", lambda: FakeLLM(parse_chat(SAMPLE)[0]))
    r = client.post("/api/digest", json={"text": SAMPLE, "save_history": False})
    assert "history_id" not in r.json() and not (tmp_path / "history.json").exists()
    r = client.post("/api/digest", json={"text": SAMPLE, "source": "WhatsApp Chat with Lab.txt"})
    hid = r.json()["history_id"]
    lst = client.get("/api/history").json()["items"]
    assert lst[0]["id"] == hid and lst[0]["title"] == "Lab"
    full = client.get(f"/api/history/{hid}").json()
    assert full["result"]["actions"][0]["receipt"]["text"]
    assert client.delete(f"/api/history/{hid}").status_code == 200
    assert client.get(f"/api/history/{hid}").status_code == 404
    assert client.delete("/api/history").json() == {"deleted": 0}


# ---------------- sessions: screenshots, live
def wait(client, sid, cond, tries=100):
    for _ in range(tries):
        s = client.get(f"/api/live/{sid}").json()
        if cond(s):
            return s
        time.sleep(0.05)
    raise AssertionError(s)


PAGE1 = {"chat_name": "Lab Group", "messages": [
    {"sender": "Sam", "text": "Does anyone have the ER diagram?"}, {"sender": "Riya", "text": "I will send it tonight"},
    {"sender": "Arjun", "text": "login module is nearly done"}]}
PAGE2 = {"chat_name": "Lab Group", "messages": [
    {"sender": "Riya", "text": "I will send it tonight"}, {"sender": "Arjun", "text": "login module is nearly done"},
    {"sender": "Dev", "text": "demo is at 10:30 on Monday"}, {"sender": "Sam", "text": "ok see you all there"},
    {"sender": "Riya", "text": "bring the printed report please"}]}


def test_live_flow_dedupes_busy_skips_digests_and_saves(client, monkeypatch, tmp_path):
    llm = VisionLLM([PAGE1, PAGE2])
    monkeypatch.setattr(main, "get_llm", lambda: llm)
    monkeypatch.setattr(live, "AUTO_DIGEST_MIN_GAP", 0)
    sid = client.post("/api/live/start", json={"kind": "live"}).json()["session"]
    assert client.post(f"/api/live/{sid}/image", content=png(), headers={"X-Kind": "image"}).json()["accepted"]
    s = wait(client, sid, lambda s: not s["busy"] and s["scans"] == 1)
    assert s["messages"] == 3 and s["chat_name"] == "Lab Group" and s["digest"]
    assert client.post(f"/api/live/{sid}/image", content=png(), headers={"X-Kind": "image"}).json()["accepted"]
    s = wait(client, sid, lambda s: not s["busy"] and s["scans"] == 2)
    assert s["messages"] == 6 and s["last_added"] == 3          # overlap not duplicated
    assert s["transcript"] == ""                                  # live sessions never hand the transcript back
    # busy frames are skipped, not queued
    live.get(sid).busy = True
    assert client.post(f"/api/live/{sid}/image", content=png(), headers={"X-Kind": "image"}).json() == {"accepted": False, "reason": "busy"}
    live.get(sid).busy = False
    assert not (tmp_path / "history.json").exists()               # nothing saved while scanning
    client.post(f"/api/live/{sid}/stop", json={"save_history": True, "source": ""})
    s = wait(client, sid, lambda s: s["state"] == "finished")
    assert s["saved_id"]
    h = client.get("/api/history").json()["items"]
    assert h[0]["kind"] == "live" and h[0]["title"] == "Lab Group"
    assert client.post(f"/api/live/{sid}/image", content=png(), headers={"X-Kind": "image"}).status_code == 409
    # a picture is never written to disk
    assert not [p for p in tmp_path.rglob("*") if p.suffix in (".png", ".jpg", ".jpeg", ".pdf")]


def test_live_stop_without_saving_leaves_no_history(client, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "get_llm", lambda: VisionLLM([PAGE1]))
    sid = client.post("/api/live/start", json={"kind": "live"}).json()["session"]
    client.post(f"/api/live/{sid}/image", content=png(), headers={"X-Kind": "image"})
    wait(client, sid, lambda s: s["scans"] == 1 and not s["busy"])
    client.post(f"/api/live/{sid}/stop", json={"save_history": False})
    s = wait(client, sid, lambda s: s["state"] == "finished")
    assert s["saved_id"] is None and not (tmp_path / "history.json").exists()


def test_upload_session_returns_transcript_for_review_and_rejects_bad_input(client, monkeypatch):
    monkeypatch.setattr(main, "get_llm", lambda: VisionLLM([PAGE1]))
    sid = client.post("/api/live/start", json={"kind": "upload"}).json()["session"]
    assert client.post(f"/api/live/{sid}/image", content=b"").status_code == 400
    client.post(f"/api/live/{sid}/image", content=png(), headers={"X-Kind": "image"})
    s = wait(client, sid, lambda s: s["scans"] == 1 and not s["busy"])
    assert "Sam: Does anyone have the ER diagram?" in s["transcript"] and s["digest"] is None
    # garbage picture -> clear error, not a crash
    client.post(f"/api/live/{sid}/image", content=b"not an image", headers={"X-Kind": "image"})
    s = wait(client, sid, lambda s: not s["busy"] and s["error"])
    assert s["error"]
    assert client.get("/api/live/nope").status_code == 404


def test_pdf_upload_session(client, monkeypatch):
    from reportlab.pdfgen import canvas
    b = io.BytesIO()
    c = canvas.Canvas(b)
    for i, line in enumerate(SAMPLE.splitlines()[:45]):
        c.drawString(30, 800 - i * 14, line[:95])
    c.save()
    monkeypatch.setattr(main, "get_llm", lambda: VisionLLM([PAGE1]))
    sid = client.post("/api/live/start", json={"kind": "upload"}).json()["session"]
    client.post(f"/api/live/{sid}/image", content=b.getvalue(), headers={"X-Kind": "pdf"})
    s = wait(client, sid, lambda s: s["scans"] == 1 and not s["busy"])
    assert s["messages"] > 10 and "Riya" in s["transcript"]
