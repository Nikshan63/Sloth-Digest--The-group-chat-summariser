"""Run the sample chat through your REAL local Gemma and print what came out.

    python scripts/try_gemma.py              # sample chat
    python scripts/try_gemma.py my_chat.txt  # your own export

Use it to see the true time and quality on your machine before you trust the app.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config                      # noqa: E402
from app.digest import run_digest           # noqa: E402
from app.llm import OllamaLLM, LLMError     # noqa: E402
from app.parser import parse_chat           # noqa: E402

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "sample_chat.txt"
msgs, skipped = parse_chat(path.read_text(errors="ignore"))
print(f"{path.name}: {len(msgs)} messages read, {skipped} lines skipped. Model: {config.OLLAMA_MODEL}")
t = time.time()
try:
    r = run_digest(msgs, OllamaLLM(), skipped, lambda s, d, n: print(f"  ... {s}"), allow_fallback=False)
except LLMError as e:
    sys.exit(f"\nFAILED: {e}")
print(f"\nDone in {time.time() - t:.1f}s  (mode: {r['mode']}, slices: {r['stats']['chunks']}, "
      f"blocked by checks: {r['stats']['dropped_by_guardrails']})\n")
print("CATCH-UP:");  [print("  " + l) for l in r["headline"]]
print("\nTO DO:");    [print(f"  - {a['who']}: {a['task']} [{a['due'] or 'no deadline'}]  <- #{a['receipt']['id']} {a['receipt']['sender']}: {a['receipt']['text'][:60]}") for a in r["actions"]]
print("\nDATES:");    [print(f"  - {d['what']}: {d['when']}") for d in r["dates"]]
print("\nQUESTIONS:"); [print(f"  - {q['who']}: {q['question']}") for q in r["questions"]]
for n in r["notes"]:
    print("\nNOTE:", n)
