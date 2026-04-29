from __future__ import annotations

import abc
import hashlib
import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

GEMINI_MODEL_NAME = "gemma-4-31b-it"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".llm_cache"

# Retry policy for the GeminiClient. The free tier on Gemma 4 caps at
# 15 RPM, so a serial eval run that issues ~20 cold calls will trip 429
# at least once. Backoff lets the run finish cleanly without manual
# pacing or chunking.
_GEMINI_MAX_RETRIES = 5
_GEMINI_BASE_BACKOFF_SECS = 4.0  # 4s = 15 RPM ceiling

log = logging.getLogger(__name__)


def _is_retryable_error(exc: BaseException) -> bool:
    """Best-effort detection of a transient Gemini failure worth retrying.

    Two categories qualify:

    - Rate-limit / quota: 429 / RESOURCE_EXHAUSTED / "rate limit".
    - Transient server-side: 500 INTERNAL / 502 / 503 UNAVAILABLE / 504.
      Gemini occasionally returns 500 INTERNAL ("Internal error
      encountered") on otherwise-valid requests; backing off and
      retrying clears these in practice.

    The google-genai SDK surfaces these with varying exception types
    across versions, so we sniff the string form rather than coupling
    to a class that may move between releases. Auth, malformed-request,
    and permanent-quota-exceeded errors don't match either category and
    propagate immediately.
    """
    text = (str(exc) or "").lower()
    if "429" in text or "resource_exhausted" in text or "resourceexhausted" in text or "rate limit" in text:
        return True
    if any(marker in text for marker in (
        "500 internal",
        "internal error encountered",
        "502 ",
        "503 ",
        "504 ",
        "unavailable",
        "deadline exceeded",
    )):
        return True
    return False


# Backwards-compatible alias for any external callers / older tests.
_is_rate_limit_error = _is_retryable_error


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
        from google.genai import types as gtypes

        config = gtypes.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        last_exc: BaseException | None = None
        for attempt in range(_GEMINI_MAX_RETRIES):
            try:
                response = self._client.models.generate_content(
                    model=GEMINI_MODEL_NAME,
                    contents=contents,
                    config=config,
                )
                return (getattr(response, "text", None) or "").strip()
            except Exception as exc:
                last_exc = exc
                # Retry transient failures (rate limits + 5xx server
                # errors). Auth / malformed-request errors don't match
                # and propagate immediately.
                if not _is_retryable_error(exc):
                    log.exception("GeminiClient.generate failed (non-retryable)")
                    raise LLMError(str(exc)) from exc
                if attempt >= _GEMINI_MAX_RETRIES - 1:
                    break
                # Exponential backoff with jitter: 4, 8, 16, 32 seconds plus
                # up to 1 second of randomness so concurrent retries don't
                # stampede.
                delay = _GEMINI_BASE_BACKOFF_SECS * (2 ** attempt) + random.uniform(0, 1)
                log.warning(
                    "GeminiClient.generate transient failure (attempt %d/%d): %s; "
                    "sleeping %.1fs before retry",
                    attempt + 1,
                    _GEMINI_MAX_RETRIES,
                    exc.__class__.__name__,
                    delay,
                )
                time.sleep(delay)

        log.exception(
            "GeminiClient.generate exhausted retries (%d attempts)",
            _GEMINI_MAX_RETRIES,
        )
        raise LLMError(str(last_exc)) from last_exc


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
