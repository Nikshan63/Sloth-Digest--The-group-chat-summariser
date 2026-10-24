"""Thin client for a local Ollama server. Returns parsed JSON or raises LLMError."""
import json
import re

import httpx

from . import config


class LLMError(Exception):
    pass


def _loads(text: str):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    raise LLMError("The model did not return valid JSON.")


class OllamaLLM:
    def __init__(self, url=None, model=None, timeout=None):
        self.url = (url or config.OLLAMA_URL).rstrip("/")
        self.model = model or config.OLLAMA_MODEL
        self.timeout = timeout or config.OLLAMA_TIMEOUT

    def chat_json(self, system: str, user: str) -> dict:
        return self._call(system, {"role": "user", "content": user})

    def chat_json_images(self, system: str, user: str, images_b64: list) -> dict:
        """Same as chat_json but the user turn also carries pictures (Gemma 3 can read images)."""
        return self._call(system, {"role": "user", "content": user, "images": list(images_b64)})

    def _call(self, system: str, user_msg: dict) -> dict:
        body = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1, "num_ctx": config.NUM_CTX},
            "messages": [{"role": "system", "content": system}, user_msg],
        }
        try:
            r = httpx.post(f"{self.url}/api/chat", json=body, timeout=self.timeout)
            r.raise_for_status()
            content = r.json()["message"]["content"]
        except httpx.ConnectError as e:
            raise LLMError("Cannot reach Ollama. Is it running? (open the Ollama app or run: ollama serve)") from e
        except httpx.HTTPStatusError as e:
            raise LLMError(f"Ollama returned {e.response.status_code}. Is the model pulled? (ollama pull {self.model})") from e
        except httpx.TimeoutException as e:
            raise LLMError("The model took too long to answer.") from e
        except (httpx.HTTPError, KeyError, ValueError) as e:
            raise LLMError(f"Unexpected reply from Ollama: {type(e).__name__}") from e
        data = _loads(content)
        if not isinstance(data, dict):
            raise LLMError("The model returned JSON, but not an object.")
        return data
