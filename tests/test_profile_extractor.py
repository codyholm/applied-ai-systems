from __future__ import annotations

import json

import pytest

from src.agents.profile_extractor import (
    ProfileExtractionError,
    extract_profile,
)
from src.llm.client import LLMClient
from src.pipeline import BuildInputs
from src.recommender import UserProfile


VALID_PROFILE_JSON = json.dumps(
    {
        "favorite_genre": "lofi",
        "favorite_mood": "chill",
        "target_energy": 0.4,
        "target_tempo_bpm": 78.0,
        "target_valence": 0.55,
        "target_danceability": 0.5,
        "target_acousticness": 0.8,
        "avoid_genres": [],
    }
)


_INPUTS = BuildInputs(
    activity="studying late at night",
    feeling="calm and focused",
    description="quiet headphones-on stuff with a vinyl warmth",
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

    profile, warnings = extract_profile(_INPUTS, llm)

    assert profile.favorite_genre == "lofi"
    assert profile.favorite_mood == "chill"
    assert profile.target_energy == 0.4
    assert profile.target_tempo_bpm == 78.0
    assert warnings == []


def test_extract_prompt_includes_only_filled_fields():
    """Blank/None fields are skipped; filled fields appear with their labels."""
    inputs = BuildInputs(
        activity="studying",
        description="quiet vinyl",
        feeling=None,
        movement="   ",  # whitespace-only is treated as blank
    )
    llm = _CountingStub([VALID_PROFILE_JSON])

    extract_profile(inputs, llm)

    prompt = llm.prompts[0]
    assert "Activity" in prompt
    assert "studying" in prompt
    assert "Description" in prompt
    assert "quiet vinyl" in prompt
    assert "Feeling" not in prompt
    assert "Movement" not in prompt


def test_extract_with_starting_from_includes_seed_block():
    """starting_from snippet is rendered into the prompt."""
    seed = UserProfile(
        favorite_genre="lofi",
        favorite_mood="chill",
        target_energy=0.4,
        target_tempo_bpm=78.0,
        target_valence=0.6,
        target_danceability=0.6,
        target_acousticness=0.78,
    )
    llm = _CountingStub([VALID_PROFILE_JSON])

    extract_profile(_INPUTS, llm, starting_from=seed)

    prompt = llm.prompts[0]
    assert "An existing profile is provided as a seed" in prompt
    assert "favorite_genre:       lofi" in prompt


def test_extract_retries_once_on_parse_failure():
    llm = _CountingStub(["not json", VALID_PROFILE_JSON])

    profile, _warnings = extract_profile(_INPUTS, llm)

    assert profile.favorite_genre == "lofi"
    assert len(llm.prompts) == 2
    assert llm.prompts[0].count("Your previous response failed JSON parsing") == 0
    assert llm.prompts[1].startswith("Your previous response failed JSON parsing")


def test_extract_raises_after_two_parse_failures():
    llm = _CountingStub(["not json", "still not json"])

    with pytest.raises(ProfileExtractionError):
        extract_profile(_INPUTS, llm)
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
            "avoid_genres": [],
        }
    )
    llm = _CountingStub([out_of_range])

    profile, _warnings = extract_profile(_INPUTS, llm)

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
        profile, warnings = extract_profile(_INPUTS, llm)

    # The fallback is the first allowed genre alphabetically.
    assert profile.favorite_genre == "acoustic"
    assert any("not in allowed list" in record.message for record in caplog.records)
    assert len(warnings) == 1
    assert "favorite_genre" in warnings[0]
    assert "drum and bass" in warnings[0]
    assert "acoustic" in warnings[0]


# --- avoid_genres ------------------------------------------------------------


def test_extract_parses_avoid_genres_list():
    payload = json.loads(VALID_PROFILE_JSON)
    payload["avoid_genres"] = ["pop", "rock"]
    llm = _CountingStub([json.dumps(payload)])

    profile, warnings = extract_profile(_INPUTS, llm)

    assert profile.avoid_genres == ["pop", "rock"]
    assert warnings == []


def test_extract_drops_invalid_avoid_genres_with_warning():
    payload = json.loads(VALID_PROFILE_JSON)
    payload["avoid_genres"] = ["pop", "drum and bass"]
    llm = _CountingStub([json.dumps(payload)])

    profile, warnings = extract_profile(_INPUTS, llm)

    assert profile.avoid_genres == ["pop"]
    assert any("drum and bass" in w for w in warnings)


def test_extract_avoid_genres_empty_list_no_warning():
    # VALID_PROFILE_JSON already has avoid_genres=[]; this is the explicit
    # regression that the empty case produces no spurious warnings.
    llm = _CountingStub([VALID_PROFILE_JSON])

    profile, warnings = extract_profile(_INPUTS, llm)

    assert profile.avoid_genres == []
    assert warnings == []


def test_extract_avoid_genres_lowercases_and_dedupes():
    payload = json.loads(VALID_PROFILE_JSON)
    payload["avoid_genres"] = ["Pop", "POP", "rock"]
    llm = _CountingStub([json.dumps(payload)])

    profile, _warnings = extract_profile(_INPUTS, llm)

    assert profile.avoid_genres == ["pop", "rock"]


def test_extract_raises_when_avoid_genres_key_missing():
    payload = json.loads(VALID_PROFILE_JSON)
    payload.pop("avoid_genres")
    llm = _CountingStub([json.dumps(payload)])

    with pytest.raises(ProfileExtractionError, match="avoid_genres"):
        extract_profile(_INPUTS, llm)


def test_extract_drops_avoid_genre_that_matches_favorite_genre():
    payload = json.loads(VALID_PROFILE_JSON)
    payload["favorite_genre"] = "lofi"
    payload["avoid_genres"] = ["lofi", "pop"]
    llm = _CountingStub([json.dumps(payload)])

    profile, warnings = extract_profile(_INPUTS, llm)

    assert profile.favorite_genre == "lofi"
    assert profile.avoid_genres == ["pop"]
    assert any("favorite_genre" in w for w in warnings)
