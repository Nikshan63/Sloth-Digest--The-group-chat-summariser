# POST DRAFT: Hacktoberfest Week 3 (Slow and Steady)

Open the Week 3 submission template on the challenge page, keep its header and tags (`devchallenge`, `hf26challenge`), and paste the sections below.
Replace only the 3 things marked **[ADD]** with numbers from your own real run (`python scripts/try_gemma.py`). Do not invent numbers.

Title: **Sloth Digest: catch up on a 300-message group chat in two lines, without it leaving your laptop**

---

*This is a submission for the [Hacktoberfest Open-Source AI Challenge Week 3: Slow and Steady](https://dev.to/challenges/hacktoberfest-week3-2026-10-19)*

## What I Built

I have a group chat for every part of my life, and overnight they pile up. I do not need all 300 messages, I need three things: what was decided, what was I asked to do, and when is it due.

Sloth Digest does that reading for me. I export a chat, drop the file in, and a Gemma model on my own laptop returns a two-line catch-up, a to-do list, the dates, and the questions people asked. Every item has a **receipt**: tap it and you see the exact original message it came from.

Who gets to be lazier: anyone in a busy class, team or family group who just wants the useful part of the chat.

## Demo

{% embed PASTE-YOUR-VIDEO-LINK-HERE %}

## Code

{% embed https://github.com/YOUR-USERNAME/sloth-digest %}

## How I Built It

- **Open-source AI:** Gemma (`gemma3:4b`), an open-weight model, served locally by Ollama in JSON mode. FastAPI backend, one plain HTML page.
- **Slices, not one giant prompt:** a 4B model does better on short inputs, so the chat is cut into numbered slices. The model reads one slice at a time and answers in strict JSON, and every item must name the message number it came from.
- **Code checks the model before you see it.** An item must point at a real message and share words with it. Numbers must exist in the chat. Deadlines must appear in the nearby messages, otherwise the deadline is removed. People must be real senders. The two-line headline is rejected if it contains a number the chat never had.
- **Receipts as the safety net:** a made-up item has no real message behind it, so it cannot appear.
- **No AI? Still useful:** if Ollama is down, a keyword-based digest runs and the page says it is a simple digest.
- **Privacy by design:** chat text is never written to disk. A test checks that the stats file holds only counts, never names or message text.
- 16 automated tests cover the parsers (WhatsApp Android and iPhone, Telegram, plain text), the checks, the fallback, the API and privacy. One test uses a fake model that lies on purpose to prove the checks catch invented tasks, numbers, people and dates.
- **A bug I hit:** my first date check let a made-up "Friday 3 pm" through because the "3" matched "Lab 3" in the chat. I made the check require the words as well as the numbers.

## Why Does Open Innovation Matter?

- **A group chat is the most private data I own.** Sending it to a hosted API is the thing I am trying to avoid. With an open model on my laptop, nothing leaves the machine.
- **It costs nothing per digest,** so running it on every chat every morning is free.
- **I could control the model.** Because I run it, I can force JSON output and put checks around it. The "receipt" idea only works because I decide exactly what the model is allowed to return.
- **Where open worked well:** [ADD one real thing Gemma did well on your own chat].
- **Where it fell short:** [ADD one real miss or slow moment]. A 4B model can miss items, so the checks are built to miss rather than invent.

## The Lazy Payoff

For one sample chat of [ADD: N messages], the page estimated about [ADD: Y minutes] of reading saved (words / 220 per minute, so it is an estimate, not a stopwatch) and the digest took [ADD: Z seconds] on `gemma3:4b` on my MacBook Air.
[If nobody else has used it yet, say so plainly here.]

## Prize Categories

Gemma (the app runs on `gemma3:4b` through Ollama). I did not use MongoDB Atlas or a .tech domain.
