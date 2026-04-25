from __future__ import annotations

import abc
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

GEMINI_MODEL_NAME = "gemma-3-27b-it"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".llm_cache"

log = logging.getLogger(__name__)


class LLMError(Exception):
    """Wraps any underlying provider error with a stable type."""


class LLMClient(abc.ABC):
    MODEL: str = "unknown"

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
    ) -> str:
        ...


class StubLLMClient(LLMClient):
    MODEL = "stub"

    def __init__(self, responses: dict[str, str] | list[str]) -> None:
        self._responses = responses
        self._call_index = 0

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
    ) -> str:
        haystack = (system or "") + prompt
        if isinstance(self._responses, dict):
            for needle, value in self._responses.items():
                if needle in haystack:
                    return value
            raise KeyError(
                f"StubLLMClient: no response key matched prompt; keys={list(self._responses)}"
            )
        if self._call_index >= len(self._responses):
            raise IndexError(
                f"StubLLMClient: call {self._call_index} exceeds list length {len(self._responses)}"
            )
        value = self._responses[self._call_index]
        self._call_index += 1
        return value


class GeminiClient(LLMClient):
    MODEL = GEMINI_MODEL_NAME

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Missing GEMINI_API_KEY")
        from google import genai  # lazy: keeps import-time cost off the test path

        self._client = genai.Client(api_key=key)

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
    ) -> str:
        contents = "\n\n".join(part for part in (system, prompt) if part)
        try:
            from google.genai import types as gtypes

            config = gtypes.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            response = self._client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            log.exception("GeminiClient.generate failed")
            raise LLMError(str(exc)) from exc
        return (getattr(response, "text", None) or "").strip()


class CachedLLMClient(LLMClient):
    def __init__(self, inner: LLMClient, cache_dir: Path | None = None) -> None:
        self._inner = inner
        self._cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    @property
    def MODEL(self) -> str:  # type: ignore[override]
        return getattr(self._inner, "MODEL", "unknown")

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def _key(
        self,
        prompt: str,
        system: str | None,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        payload = json.dumps(
            [self.MODEL, system, prompt, temperature, max_output_tokens],
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
    ) -> str:
        key = self._key(prompt, system, temperature, max_output_tokens)
        path = self._cache_dir / f"{key}.json"
        if path.exists():
            try:
                blob = json.loads(path.read_text(encoding="utf-8"))
                self._hits += 1
                return blob["response"]
            except (json.JSONDecodeError, KeyError) as exc:
                log.warning("Cache file %s is corrupt (%s); falling through", path, exc)
        response = self._inner.generate(
            prompt,
            system=system,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        self._misses += 1
        path.write_text(
            json.dumps(
                {
                    "response": response,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": self.MODEL,
                }
            ),
            encoding="utf-8",
        )
        return response
