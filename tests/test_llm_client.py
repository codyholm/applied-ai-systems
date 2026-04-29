from __future__ import annotations

import pytest

from src.llm.client import (
    CachedLLMClient,
    GeminiClient,
    LLMClient,
    StubLLMClient,
)


def test_stub_client_returns_canned_response_by_substring() -> None:
    client = StubLLMClient({"hello": "hi back"})
    assert client.generate("say hello to me") == "hi back"


def test_stub_client_dict_raises_keyerror_on_no_match() -> None:
    client = StubLLMClient({"foo": "bar"})
    with pytest.raises(KeyError):
        client.generate("nothing matches")


def test_stub_client_list_returns_by_index() -> None:
    client = StubLLMClient(["first", "second"])
    assert client.generate("anything") == "first"
    assert client.generate("anything else") == "second"
    with pytest.raises(IndexError):
        client.generate("third call")


def test_stub_client_includes_system_in_substring_search() -> None:
    client = StubLLMClient({"sys-marker": "matched"})
    assert client.generate("body", system="sys-marker prefix") == "matched"


class _CountingStub(LLMClient):
    MODEL = "counting-stub"

    def __init__(self, response: str) -> None:
        self._response = response
        self.calls = 0

    def generate(self, prompt, *, system=None, temperature=0.2, max_output_tokens=1024):  # type: ignore[override]
        self.calls += 1
        return self._response


def test_cached_client_round_trips_through_disk(tmp_path) -> None:
    inner = _CountingStub("abc")
    cached = CachedLLMClient(inner, cache_dir=tmp_path)

    first = cached.generate("hello", system="sys", temperature=0.5, max_output_tokens=256)
    second = cached.generate("hello", system="sys", temperature=0.5, max_output_tokens=256)

    assert first == "abc"
    assert second == "abc"
    assert inner.calls == 1
    assert cached.misses == 1
    assert cached.hits == 1


def test_cached_client_distinct_args_produce_distinct_cache_keys(tmp_path) -> None:
    inner = _CountingStub("abc")
    cached = CachedLLMClient(inner, cache_dir=tmp_path)

    cached.generate("hello", temperature=0.2)
    cached.generate("hello", temperature=0.9)

    assert inner.calls == 2
    assert cached.misses == 2
    assert cached.hits == 0
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 2


def test_gemini_client_raises_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Missing GEMINI_API_KEY"):
        GeminiClient()


# --- retry / backoff on 429 ---------------------------------------------------


class _FakeGenAIModels:
    """Stand-in for the genai Client.models surface so we can drive
    GeminiClient.generate without hitting the real API."""

    def __init__(self, *, failures: int, error_message: str = "429 RESOURCE_EXHAUSTED") -> None:
        self.calls = 0
        self._failures = failures
        self._error_message = error_message

    def generate_content(self, *, model, contents, config):  # noqa: ARG002 — match SDK shape
        self.calls += 1
        if self.calls <= self._failures:
            raise RuntimeError(self._error_message)

        class _Response:
            text = "ok"

        return _Response()


class _FakeGenAIClient:
    def __init__(self, models: _FakeGenAIModels) -> None:
        self.models = models


def _gemini_with_fake(fake: _FakeGenAIModels) -> GeminiClient:
    """Build a GeminiClient that talks to a fake genai client."""
    instance = GeminiClient.__new__(GeminiClient)
    instance._client = _FakeGenAIClient(fake)  # type: ignore[attr-defined]
    return instance


def test_gemini_client_retries_on_rate_limit_then_succeeds(monkeypatch) -> None:
    # Don't actually sleep during the backoff loop.
    monkeypatch.setattr("src.llm.client.time.sleep", lambda _s: None)
    fake = _FakeGenAIModels(failures=2)  # 429 twice, then success
    client = _gemini_with_fake(fake)

    assert client.generate("hi") == "ok"
    assert fake.calls == 3


def test_gemini_client_gives_up_after_max_retries(monkeypatch) -> None:
    from src.llm.client import LLMError, _GEMINI_MAX_RETRIES

    monkeypatch.setattr("src.llm.client.time.sleep", lambda _s: None)
    fake = _FakeGenAIModels(failures=999)
    client = _gemini_with_fake(fake)

    with pytest.raises(LLMError, match="429"):
        client.generate("hi")
    assert fake.calls == _GEMINI_MAX_RETRIES


def test_gemini_client_does_not_retry_non_rate_limit_errors(monkeypatch) -> None:
    from src.llm.client import LLMError

    monkeypatch.setattr("src.llm.client.time.sleep", lambda _s: None)
    fake = _FakeGenAIModels(failures=999, error_message="invalid argument")
    client = _gemini_with_fake(fake)

    with pytest.raises(LLMError, match="invalid argument"):
        client.generate("hi")
    assert fake.calls == 1


def test_gemini_client_detects_resource_exhausted_message(monkeypatch) -> None:
    monkeypatch.setattr("src.llm.client.time.sleep", lambda _s: None)
    fake = _FakeGenAIModels(failures=1, error_message="ResourceExhausted: quota")
    client = _gemini_with_fake(fake)

    assert client.generate("hi") == "ok"
    assert fake.calls == 2


def test_gemini_client_retries_on_500_internal_error(monkeypatch) -> None:
    # Real-world example: Gemini occasionally returns 500 INTERNAL on
    # otherwise-valid requests. Backoff + retry clears it.
    monkeypatch.setattr("src.llm.client.time.sleep", lambda _s: None)
    fake = _FakeGenAIModels(
        failures=2,
        error_message=(
            "500 INTERNAL. {'error': {'code': 500, 'message': "
            "'Internal error encountered.', 'status': 'INTERNAL'}}"
        ),
    )
    client = _gemini_with_fake(fake)

    assert client.generate("hi") == "ok"
    assert fake.calls == 3


def test_gemini_client_retries_on_503_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("src.llm.client.time.sleep", lambda _s: None)
    fake = _FakeGenAIModels(failures=1, error_message="503 Service Unavailable")
    client = _gemini_with_fake(fake)

    assert client.generate("hi") == "ok"
    assert fake.calls == 2
