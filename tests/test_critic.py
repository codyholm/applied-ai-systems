"""Tests for the profile-faithfulness critic.

The critic's job (rewritten in milestone 5.2 and re-shaped in 5.3) is to
check that an extracted UserProfile faithfully reflects the listener's
BuildInputs bundle, not whether the recommender's top-5 matched intent.
These tests drive the new question with stub LLM responses.
"""

from __future__ import annotations

import json

from src.agents.critic import critique_extraction
from src.llm.client import StubLLMClient
from src.pipeline import BuildInputs
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

_INPUTS = BuildInputs(
    activity="focused writing session",
    feeling="calm and quiet",
    description="Quiet, mellow lofi to focus on writing.",
)


def test_critic_returns_ok_when_profile_is_faithful():
    response = json.dumps(
        {"verdict": "ok", "adjustments": None, "reason": "encodes inputs faithfully"}
    )
    llm = StubLLMClient([response])

    result = critique_extraction(_INPUTS, CANDIDATE, llm)

    assert result.verdict == "ok"
    assert result.adjustments is None
    assert "faithfully" in result.reason


def test_critic_prompt_includes_inputs_bundle():
    """The critic prompt receives a labeled bundle, not a single NL string."""

    captured: dict[str, str] = {}

    class _CapturingStub(StubLLMClient):
        def generate(self, prompt, *, system=None, temperature=0.2, max_output_tokens=1024):
            captured["prompt"] = prompt
            return json.dumps({"verdict": "ok", "adjustments": None, "reason": "ok"})

    llm = _CapturingStub([])
    critique_extraction(_INPUTS, CANDIDATE, llm)

    prompt = captured["prompt"]
    assert "Activity" in prompt
    assert "focused writing session" in prompt
    assert "Description" in prompt
    assert "Quiet, mellow lofi" in prompt
    # Fields the listener didn't fill should not appear under their labels.
    assert "Movement" not in prompt
    assert "Genres" not in prompt


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

    inputs = BuildInputs(description="Around 90 BPM, jazzy, mid-tempo.")
    result = critique_extraction(inputs, CANDIDATE, llm)

    assert result.verdict == "refine"
    assert result.adjustments == {"target_tempo_bpm": 90.0, "target_energy": 0.5}
    assert "around 90 BPM" in result.reason


def test_critic_parse_failure_degrades_to_ok():
    llm = StubLLMClient(["this is not json"])

    result = critique_extraction(_INPUTS, CANDIDATE, llm)

    assert result.verdict == "ok"
    assert result.adjustments is None
    assert result.reason == "parse_failure"


def test_critic_refine_without_adjustments_degrades_to_ok():
    response = json.dumps(
        {"verdict": "refine", "adjustments": None, "reason": "no useful change"}
    )
    llm = StubLLMClient([response])

    result = critique_extraction(_INPUTS, CANDIDATE, llm)

    assert result.verdict == "ok"
    assert result.reason == "parse_failure"


def test_critic_clamps_adjustments_into_valid_ranges():
    response = json.dumps(
        {
            "verdict": "refine",
            "adjustments": {
                "target_energy": 1.4,        # > 1.0 -> clamp to 1.0
                "target_tempo_bpm": 5.0,     # < 40 -> clamp to 40
                "target_valence": -0.1,      # < 0 -> clamp to 0
                "favorite_genre": "rock",    # string passes through
            },
            "reason": "out-of-range values",
        }
    )
    llm = StubLLMClient([response])

    inputs = BuildInputs(description="test")
    result = critique_extraction(inputs, CANDIDATE, llm)

    assert result.verdict == "refine"
    assert result.adjustments == {
        "target_energy": 1.0,
        "target_tempo_bpm": 40.0,
        "target_valence": 0.0,
        "favorite_genre": "rock",
    }


# --- avoid_genres adjustments -----------------------------------------------


def test_critic_can_adjust_avoid_genres():
    response = json.dumps(
        {
            "verdict": "refine",
            "adjustments": {"avoid_genres": ["pop"]},
            "reason": "listener said no pop",
        }
    )
    llm = StubLLMClient([response])

    inputs = BuildInputs(description="lofi but no pop please")
    result = critique_extraction(inputs, CANDIDATE, llm)

    assert result.verdict == "refine"
    assert result.adjustments == {"avoid_genres": ["pop"]}


def test_critic_drops_invalid_avoid_genres_entries_in_adjustments():
    response = json.dumps(
        {
            "verdict": "refine",
            "adjustments": {"avoid_genres": ["pop", "made_up_genre"]},
            "reason": "trim list",
        }
    )
    llm = StubLLMClient([response])

    inputs = BuildInputs(description="no pop")
    result = critique_extraction(inputs, CANDIDATE, llm)

    assert result.verdict == "refine"
    assert result.adjustments == {"avoid_genres": ["pop"]}


def test_critic_rejects_non_list_avoid_genres_in_adjustments():
    # Only the avoid_genres adjustment is malformed; with no other valid
    # adjustments, the critic degrades to ok.
    response = json.dumps(
        {
            "verdict": "refine",
            "adjustments": {"avoid_genres": "pop"},
            "reason": "wrong shape",
        }
    )
    llm = StubLLMClient([response])

    inputs = BuildInputs(description="no pop")
    result = critique_extraction(inputs, CANDIDATE, llm)

    assert result.verdict == "ok"
    assert result.adjustments is None


def test_critic_can_adjust_avoid_genres_to_empty_list():
    # Removing an over-zealous avoid is a valid refine path.
    response = json.dumps(
        {
            "verdict": "refine",
            "adjustments": {"avoid_genres": []},
            "reason": "listener never said avoid",
        }
    )
    llm = StubLLMClient([response])

    inputs = BuildInputs(description="just lofi")
    result = critique_extraction(inputs, CANDIDATE, llm)

    assert result.verdict == "refine"
    assert result.adjustments == {"avoid_genres": []}
