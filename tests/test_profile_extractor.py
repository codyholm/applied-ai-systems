from __future__ import annotations

import json

import pytest

from src.agents.profile_extractor import (
    ProfileExtractionError,
    extract_profile,
)
from src.llm.client import LLMClient


VALID_PROFILE_JSON = json.dumps(
    {
        "favorite_genre": "lofi",
        "favorite_mood": "chill",
        "target_energy": 0.4,
        "target_tempo_bpm": 78.0,
        "target_valence": 0.55,
        "target_danceability": 0.5,
        "target_acousticness": 0.8,
    }
)


class _CountingStub(LLMClient):
    """Returns canned responses by call index AND records each prompt seen."""

    MODEL = "counting-stub"

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    def generate(self, prompt, *, system=None, temperature=0.2, max_output_tokens=1024):
        self.prompts.append(prompt)
        return self.responses[len(self.prompts) - 1]


def test_extract_happy_path():
    llm = _CountingStub([VALID_PROFILE_JSON])

    profile, warnings = extract_profile("chill lofi for studying", llm)

    assert profile.favorite_genre == "lofi"
    assert profile.favorite_mood == "chill"
    assert profile.target_energy == 0.4
    assert profile.target_tempo_bpm == 78.0
    assert warnings == []


def test_extract_retries_once_on_parse_failure():
    llm = _CountingStub(["not json", VALID_PROFILE_JSON])

    profile, _warnings = extract_profile("anything", llm)

    assert profile.favorite_genre == "lofi"
    assert len(llm.prompts) == 2
    assert llm.prompts[0].count("Your previous response failed JSON parsing") == 0
    assert llm.prompts[1].startswith("Your previous response failed JSON parsing")


def test_extract_raises_after_two_parse_failures():
    llm = _CountingStub(["not json", "still not json"])

    with pytest.raises(ProfileExtractionError):
        extract_profile("anything", llm)
    assert len(llm.prompts) == 2


def test_extract_clamps_out_of_range_values():
    out_of_range = json.dumps(
        {
            "favorite_genre": "lofi",
            "favorite_mood": "chill",
            "target_energy": 1.5,
            "target_tempo_bpm": 5.0,
            "target_valence": -0.3,
            "target_danceability": 2.0,
            "target_acousticness": -0.2,
        }
    )
    llm = _CountingStub([out_of_range])

    profile, _warnings = extract_profile("hyperbolic input", llm)

    assert profile.target_energy == 1.0
    assert profile.target_tempo_bpm == 40.0
    assert profile.target_valence == 0.0
    assert profile.target_danceability == 1.0
    assert profile.target_acousticness == 0.0


def test_extract_falls_back_unknown_genre_and_records_warning(caplog):
    payload = json.loads(VALID_PROFILE_JSON)
    payload["favorite_genre"] = "drum and bass"
    llm = _CountingStub([json.dumps(payload)])

    with caplog.at_level("WARNING"):
        profile, warnings = extract_profile("dnb please", llm)

    # The fallback is the first allowed genre alphabetically.
    assert profile.favorite_genre == "acoustic"
    assert any("not in allowed list" in record.message for record in caplog.records)
    assert len(warnings) == 1
    assert "favorite_genre" in warnings[0]
    assert "drum and bass" in warnings[0]
    assert "acoustic" in warnings[0]
