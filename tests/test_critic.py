"""Tests for the profile-faithfulness critic.

The critic's job (rewritten in milestone 5.2) is to check that an
extracted UserProfile faithfully reflects the listener's plain-English
description, not whether the recommender's top-5 matched intent. These
tests drive the new question with stub LLM responses.
"""

from __future__ import annotations

import json

from src.agents.critic import critique_extraction
from src.llm.client import StubLLMClient
from src.recommender import UserProfile


CANDIDATE = UserProfile(
    favorite_genre="lofi",
    favorite_mood="chill",
    target_energy=0.4,
    target_tempo_bpm=78.0,
    target_valence=0.55,
    target_danceability=0.5,
    target_acousticness=0.8,
)


def test_critic_returns_ok_when_profile_is_faithful():
    response = json.dumps(
        {"verdict": "ok", "adjustments": None, "reason": "encodes description faithfully"}
    )
    llm = StubLLMClient([response])

    result = critique_extraction(
        "Quiet, mellow lofi to focus on writing.", CANDIDATE, llm
    )

    assert result.verdict == "ok"
    assert result.adjustments is None
    assert "faithfully" in result.reason


def test_critic_returns_refine_with_corrected_absolute_values():
    response = json.dumps(
        {
            "verdict": "refine",
            "adjustments": {"target_tempo_bpm": 90.0, "target_energy": 0.5},
            "reason": (
                "listener said 'around 90 BPM' but candidate has 78; "
                "energy at 0.4 is below the 'mid-tempo' band"
            ),
        }
    )
    llm = StubLLMClient([response])

    result = critique_extraction(
        "Around 90 BPM, jazzy, mid-tempo.", CANDIDATE, llm
    )

    assert result.verdict == "refine"
    assert result.adjustments == {"target_tempo_bpm": 90.0, "target_energy": 0.5}
    assert "around 90 BPM" in result.reason


def test_critic_parse_failure_degrades_to_ok():
    llm = StubLLMClient(["this is not json"])

    result = critique_extraction("anything", CANDIDATE, llm)

    assert result.verdict == "ok"
    assert result.adjustments is None
    assert result.reason == "parse_failure"


def test_critic_refine_without_adjustments_degrades_to_ok():
    response = json.dumps(
        {"verdict": "refine", "adjustments": None, "reason": "no useful change"}
    )
    llm = StubLLMClient([response])

    result = critique_extraction("anything", CANDIDATE, llm)

    assert result.verdict == "ok"
    assert result.reason == "parse_failure"


def test_critic_clamps_adjustments_into_valid_ranges():
    response = json.dumps(
        {
            "verdict": "refine",
            "adjustments": {
                "target_energy": 1.4,        # > 1.0 → clamp to 1.0
                "target_tempo_bpm": 5.0,     # < 40 → clamp to 40
                "target_valence": -0.1,      # < 0 → clamp to 0
                "favorite_genre": "rock",    # string passes through
            },
            "reason": "out-of-range values",
        }
    )
    llm = StubLLMClient([response])

    result = critique_extraction("test", CANDIDATE, llm)

    assert result.verdict == "refine"
    assert result.adjustments == {
        "target_energy": 1.0,
        "target_tempo_bpm": 40.0,
        "target_valence": 0.0,
        "favorite_genre": "rock",
    }
