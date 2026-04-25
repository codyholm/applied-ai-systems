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
