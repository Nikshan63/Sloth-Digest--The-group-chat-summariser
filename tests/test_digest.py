import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import config, main
from app.digest import SYSTEM_HEADLINE, clean_raw, run_digest, chunk_messages
from app.llm import LLMError
from app.parser import parse_chat, select_window

SAMPLE = (Path(__file__).parent.parent / "data" / "sample_chat.txt").read_text()


def mid(msgs, needle):
    return next(m.idx for m in msgs if needle in m.text)


class FakeLLM:
    """Plays the role of Gemma. `map_reply` is a function msgs->dict so tests can lie on purpose."""
    model = "fake"

    def __init__(self, msgs, map_reply=None, headline=None, fail=False):
        self.msgs, self.fail, self.calls = msgs, fail, 0
        self.map_reply = map_reply or self.honest
        self.headline = headline if headline is not None else {"lines": ["Demo is Monday 26 Oct at 10:30 am in Lab 3.", "Report goes to Riya on Sunday."]}

    def honest(self, m):
        return {
            "points": [{"text": "The demo with Prof. Banerjee is on Monday at 10:30 am in Lab 3.", "ids": [mid(m, "Lab 3")]}],
            "actions": [
                {"who": "Arjun", "task": "Finish the login module", "due": "Saturday night", "id": mid(m, "login module is 80%")},
                {"who": "Sam", "task": "Print 4 copies of the report", "due": "Sunday evening", "id": mid(m, "print 4 copies")},
                {"who": "Dev", "task": "Send the report draft to Riya", "due": "Sunday 6 pm", "id": mid(m, "draft goes to Riya")},
            ],
            "dates": [{"what": "Final demo", "when": "Monday 26 Oct 10:30 am", "id": mid(m, "Lab 3")}],
            "questions": [{"who": "Sam", "question": "Does anyone have the ER diagram?", "id": mid(m, "ER diagram? i need")}],
        }

    def chat_json(self, system, user):
        self.calls += 1
        if self.fail:
            raise LLMError("Cannot reach Ollama.")
        if system == SYSTEM_HEADLINE:
            return self.headline
        return self.map_reply(self.msgs)


@pytest.fixture
def msgs():
    m, _ = parse_chat(SAMPLE)
    return m


@pytest.fixture(autouse=True)
def tmp_data(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "STATS_FILE", tmp_path / "stats.json")


# ---------------- parser ----------------
def test_whatsapp_android_and_system_and_media_lines(msgs):
    m, skipped = parse_chat(SAMPLE)
    assert len(m) == 50 and skipped == 4  # 2 system lines + 2 media lines
    assert m[0].sender == "Riya" and "Lab 3" in m[0].text
    assert all("<Media omitted>" not in x.text for x in m)


def test_iphone_style_and_multiline_and_narrow_nbsp():
    txt = "‎[24/10/26, 8:05:11 AM] Sam: first line\nsecond line\n[24/10/26, 8:06:00 AM] Riya: ok"
    m, _ = parse_chat(txt)
    assert [x.sender for x in m] == ["Sam", "Riya"]
    assert m[0].text == "first line\nsecond line"
    assert m[0].dt.hour == 8 and m[0].dt.month == 10


def test_month_first_dates_are_detected_when_unambiguous():
    m, _ = parse_chat("10/25/2026, 9:00 PM - A: hi\n10/26/2026, 9:00 PM - B: yo")
    assert m[0].dt.month == 10 and m[0].dt.day == 25


def test_telegram_json():
    data = {"messages": [{"type": "message", "from": "Riya", "date": "2026-10-23T20:05:00", "text": ["see ", {"type": "bold", "text": "Lab 3"}]}]}
    m, _ = parse_chat(json.dumps(data))
    assert m[0].sender == "Riya" and m[0].text == "see Lab 3" and m[0].dt.hour == 20


def test_plain_name_colon_text():
    m, _ = parse_chat("Riya: hello there\nSam: hi\nRiya: bring the charger")
    assert len(m) == 3 and m[2].text == "bring the charger"


def test_window_filters(msgs):
    last = select_window(msgs, last_hours=12)
    assert 0 < len(last) < len(msgs) and last[-1].idx == msgs[-1].idx
    assert len(select_window(msgs, last_n=5)) == 5


def test_chunking_keeps_every_message_once(msgs, monkeypatch):
    monkeypatch.setattr(config, "CHUNK_CHARS", 600)
    chunks = chunk_messages(msgs)
    assert len(chunks) > 3 and [x.idx for c in chunks for x in c] == [x.idx for x in msgs]


# ---------------- digest with an honest model ----------------
def test_honest_model_gives_items_with_receipts(msgs):
    r = run_digest(msgs, FakeLLM(msgs))
    assert r["mode"] == "ai" and len(r["headline"]) == 2
    who = {a["who"]: a for a in r["actions"]}
    assert set(who) == {"Arjun", "Sam", "Dev"}
    assert who["Arjun"]["due"] == "Saturday night"
    assert "login module" in who["Arjun"]["receipt"]["text"] and who["Arjun"]["receipt"]["sender"] == "Arjun"
    assert r["dates"][0]["receipt"]["text"].startswith("yes") is False  # receipt is the real message
    assert r["stats"]["minutes_saved"] >= 0 and r["stats"]["dropped_by_guardrails"] == 0


# ---------------- guardrails against a model that makes things up ----------------
def test_invented_items_are_dropped(msgs):
    def liar(m):
        good = FakeLLM(m).honest(m)
        good["actions"] += [
            {"who": "Arjun", "task": "Deploy the app to AWS", "due": "tomorrow", "id": mid(m, "login module is 80%")},   # task not in that message
            {"who": "Sam", "task": "Print 400 copies of the report", "due": "", "id": mid(m, "print 4 copies")},          # number not in chat
            {"who": "Ghost", "task": "Finish the login module", "due": "Friday 9 pm", "id": mid(m, "login module is 80%")},  # unknown person
            {"who": "Dev", "task": "Send the report draft", "due": "", "id": 9999},                                          # message does not exist
        ]
        good["dates"].append({"what": "Hackathon", "when": "Friday 3 pm", "id": mid(m, "Lab 3")})                          # date not in message
        good["questions"].append({"who": "Riya", "question": "Who wants pizza?", "id": mid(m, "Lab 3")})                  # not a question
        good["points"].append({"text": "The team raised 5000 rupees.", "ids": []})                                         # number not in chat
        return good

    r = run_digest(msgs, FakeLLM(msgs, liar))
    tasks = [a["task"] for a in r["actions"]]
    assert "Deploy the app to AWS" not in tasks and "Print 400 copies of the report" not in tasks
    assert all(a["who"] in {"Arjun", "Sam", "Dev", "Unassigned"} for a in r["actions"])
    ghost = [a for a in r["actions"] if a["task"] == "Finish the login module" and a["who"] == "Unassigned"]
    assert ghost == [] or ghost[0]["due"] == ""  # a person/deadline that is not in the chat never survives
    assert not any(a["receipt"]["id"] == 9999 for a in r["actions"])
    assert all(d["what"] != "Hackathon" for d in r["dates"])
    assert all(q["question"] != "Who wants pizza?" for q in r["questions"])
    assert all("5000" not in p for p in r["points"])
    assert r["stats"]["dropped_by_guardrails"] >= 6


def test_untrusted_headline_falls_back_to_points(msgs):
    r = run_digest(msgs, FakeLLM(msgs, headline={"lines": ["The team raised 9999 rupees.", "Party at 7."]}))
    assert "9999" not in " ".join(r["headline"]) and r["headline"]
    assert any("headline" in n.lower() for n in r["notes"])


def test_messy_model_output_does_not_crash(msgs):
    weird = lambda m: {"points": None, "actions": [{"who": None, "task": ["x"], "due": 5, "id": "[7]"}, "junk", None],
                       "dates": "none", "questions": [{"id": None}]}
    r = run_digest(msgs, FakeLLM(msgs, weird))
    assert r["mode"] == "ai" and r["actions"] == []
    assert clean_raw(None) == {"points": [], "actions": [], "dates": [], "questions": []}


def test_ai_down_gives_keyword_fallback(msgs):
    r = run_digest(msgs, FakeLLM(msgs, fail=True))
    assert r["mode"] == "fallback" and r["actions"] and any("AI unavailable" in n for n in r["notes"])
    with pytest.raises(LLMError):
        run_digest(msgs, FakeLLM(msgs, fail=True), allow_fallback=False)


# ---------------- API + privacy ----------------
@pytest.fixture
def client(msgs, monkeypatch):
    monkeypatch.setattr(main, "get_llm", lambda: FakeLLM(msgs))
    return TestClient(main.app)


def test_api_digest_and_stats_without_chat_text(client, tmp_path):
    r = client.post("/api/digest", json={"text": SAMPLE, "save_history": False})
    assert r.status_code == 200 and r.json()["actions"]
    s = client.get("/api/stats").json()
    assert s["digests"] == 1 and s["messages_read"] == 50 and s["ai_runs"] == 1
    on_disk = " ".join(p.read_text() for p in tmp_path.rglob("*") if p.is_file())
    for secret in ("Riya", "Lab 3", "Prof. Banerjee", "login module", "ER diagram"):
        assert secret not in on_disk


def test_api_rejects_empty_or_unreadable_text(client):
    for bad in ("", "   ", "just some words without any chat format at all"):
        assert client.post("/api/digest", json={"text": bad}).status_code == 400


def test_api_job_flow_reports_progress_and_result(client):
    jid = client.post("/api/jobs", json={"text": SAMPLE}).json()["job"]
    for _ in range(50):
        j = client.get(f"/api/jobs/{jid}").json()
        if j["state"] != "running":
            break
        time.sleep(0.05)
    assert j["state"] == "done" and j["result"]["headline"]
    assert client.get("/api/jobs/nope").status_code == 404


def test_api_ollama_down_returns_fallback_not_a_crash(monkeypatch):
    monkeypatch.setattr(main, "get_llm", lambda: FakeLLM([], fail=True))
    r = TestClient(main.app).post("/api/digest", json={"text": SAMPLE})
    assert r.status_code == 200 and r.json()["mode"] == "fallback"
    r2 = TestClient(main.app).post("/api/digest", json={"text": SAMPLE, "allow_fallback": False})
    assert r2.status_code == 502
