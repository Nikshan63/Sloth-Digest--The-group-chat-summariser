import os
from pathlib import Path

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "240"))
NUM_CTX = int(os.environ.get("SLOTH_NUM_CTX", "8192"))

# How much chat goes into one model call. A 4B model does better on short windows.
CHUNK_CHARS = int(os.environ.get("SLOTH_CHUNK_CHARS", "5000"))
MAX_CHUNKS = int(os.environ.get("SLOTH_MAX_CHUNKS", "24"))
MAX_MSG_CHARS_IN_PROMPT = 500

# Only counts are ever written to disk (never chat text).
DATA_DIR = Path(os.environ.get("SLOTH_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))
STATS_FILE = DATA_DIR / "stats.json"

WORDS_PER_MINUTE = 220  # average silent reading speed used for the time-saved estimate
