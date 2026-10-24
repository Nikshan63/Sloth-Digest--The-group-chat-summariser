# Sloth Digest 🦥

Drop in an exported group chat. Get a **two-line catch-up**, the **to-dos**, the **dates** and the **questions asked**, each with a **receipt** (the real message it came from).
A Gemma model runs **locally through Ollama**, so your chat never leaves your computer and nothing is saved.

Built for the DEV Hacktoberfest Open-Source AI Challenge, Week 3: *Slow and Steady*.

## Why it saves effort

A busy group chat can have hundreds of messages overnight. You do not need all of them, you need: what was decided, what was I asked to do, and what are the deadlines.
The page also estimates how many minutes of reading you skipped (word count / 220 words per minute, labelled as an estimate).

## How it works

1. **Parse** the export (WhatsApp Android or iPhone style, Telegram `result.json`, or plain `Name: message` lines). System lines and "media omitted" lines are dropped. Every message gets a number.
2. **Slice** the chat into chunks small enough for a 4B model, and ask Gemma to return strict JSON for each slice: points, actions, dates, questions, each pointing at a message number.
3. **Check every claim with code** before you see it:
   - an item must point at a real message, and share words with it;
   - numbers must exist in the chat; deadlines must appear in the nearby messages (otherwise the deadline is removed, or the item dropped);
   - a person must be a real sender (otherwise it shows "Unassigned");
   - the final two-line headline is rejected if it contains a number the chat never had.
4. **Show receipts.** Tap "receipt" on any to-do to see the original message, sender and time. If the model made something up, it has no receipt and does not appear.
5. If Ollama is not available, a **keyword-based fallback** still produces a simple digest and says so clearly.

## Privacy

- Runs on your machine. The only network call is to your own Ollama (`127.0.0.1`).
- Chat text is **not written to disk**. `data/stats.json` stores counts only (digests, messages, estimated minutes). A test checks that no names or message text ever land in it.
- Results live in memory for 30 minutes so the page can fetch them, then are discarded.

## Run it on a Mac

```bash
# 1. Install Ollama (https://ollama.com), open it, then:
ollama pull gemma3:4b
ollama list                      # confirm the exact tag

# 2. Set up the app
cd sloth-digest
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. (Optional) try the real model in the terminal first
export OLLAMA_MODEL="gemma3:4b"
python scripts/try_gemma.py

# 4. Start the page
uvicorn app.main:app --reload
# open http://127.0.0.1:8000
```

In the page: **Load a sample chat** → **Digest it, slowly**. For your own chat, in WhatsApp open the chat → ⋮ / contact name → **Export chat** → **Without media**, and drop the `.txt`. (An iPhone export is a `.zip`: unzip it and use `_chat.txt`.)
Run the tests any time: `python -m pytest -q` (they use a stand-in model, so no Ollama is needed).

## Honest limits

- A 4B model is small. It can miss things; the checks make it **miss rather than invent**, so read "0 items" as "nothing trustworthy found", not "nothing happened".
- "Questions asked" lists questions, it does not check whether someone answered them later.
- Dates are shown as the chat wrote them ("Friday", "10:30 am"); they are not converted to calendar dates.
- Very long chats: only the latest 24 slices (about 120,000 characters) are read, and the page says so.
- Chats in mixed languages work only as well as Gemma handles them.
- The time-saved figure is an estimate from word counts, not a stopwatch.

## Layout

```
app/parser.py    exports -> numbered messages
app/digest.py    slicing, prompts, checks (the core), keyword fallback
app/llm.py       Ollama client (JSON mode)
app/stats.py     counts-only stats
app/main.py      FastAPI + background jobs for the progress bar
static/          the web page
scripts/try_gemma.py   run the real model from the terminal
tests/           parser, guardrail, fallback, API and privacy tests
data/sample_chat.txt   fictional sample (no real people)
```


## New in v2: screenshots, PDFs, History and Live scan

- **Drop screenshots (png/jpg/webp) or a PDF** next to the chat file. Pictures are read by Gemma vision on your computer, the text appears in the box for you to check, then you press Digest. Text PDFs are read directly; scanned PDFs are rendered to pages and read as pictures (max 12 pages).
- **History** tab: every finished digest with its date and time, searchable, open or delete. Saved only in `data/history.json` (git-ignored). It holds the digest and its receipt messages, never the chat file or any picture. Untick "Save finished digests to History" to turn it off.
- **Live scan** tab: share the chat window or browser tab (WhatsApp Web works), and Sloth Digest looks every 15/30/60 s, skips unchanged screens, merges new messages without duplicates and keeps the digest updated. Frames are read and dropped, never stored.

### Honest limits
- Near-live, not instant: a local 4B vision model needs several seconds per picture.
- It only sees what is on screen. Scroll up to cover older messages.
- Reading small text from pictures can make mistakes. Zoom the chat in; check the receipts.
- The browser-side flow and the server were tested with a stand-in model; real Gemma vision quality is for you to judge on your own chats.
- macOS: allow Screen Recording for your browser (System Settings > Privacy and Security).
- There is no WhatsApp API involved. Nothing is sent to WhatsApp or the cloud.
