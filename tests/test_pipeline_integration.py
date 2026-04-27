"""End-to-end tests for the build_profile and recommend pipelines.

Both pipelines are tested with StubLLMClient — no real API calls. The
KB used for retrieval is a fixture written under tmp_path so the tests
are independent of the on-disk KB content.
"""

from __future__ import annotations

import json

import pytest

from src.kb import retriever as retriever_module
from src.llm.client import StubLLMClient
from src.pipeline import (
    MAX_REFINEMENT_ITERS,
    BuildInputs,
    EmptyBuildInputsError,
    ProfileBuildResult,
    RecommendationResult,
    build_profile,
    recommend,
)
from src.recommender import Song, UserProfile


def _write_doc(path, body, **frontmatter):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["---"]
    for key, value in frontmatter.items():
        lines.append(f"{key}: {value}")
    lines.append("---")
    lines.append("")
    lines.append(body)
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_fixture_kb(tmp_path, songs):
    _write_doc(
        tmp_path / "genres" / "lofi.md",
        "Lofi sits between 60 and 90 BPM and leans on warm vinyl-style filters.",
        type="genre",
        name="lofi",
    )
    _write_doc(
        tmp_path / "moods" / "chill.md",
        "Chill describes a low-energy listening state with mid valence.",
        type="mood",
        name="chill",
    )
    for song in songs:
        _write_doc(
            tmp_path / "songs" / f"{song.id}.md",
            f"Song {song.id} fact: track is built for headphones.",
            type="song",
            id=song.id,
        )


def _fixture_songs():
    return [
        Song(
            id=1, title="Library Hours", artist="Tester",
            genre="lofi", mood="chill",
            energy=0.40, tempo_bpm=78.0, valence=0.60,
            danceability=0.60, acousticness=0.78,
        ),
        Song(
            id=2, title="Tape Loop", artist="Tester",
            genre="lofi", mood="chill",
            energy=0.42, tempo_bpm=80.0, valence=0.58,
            danceability=0.62, acousticness=0.75,
        ),
        Song(
            id=3, title="Quiet Window", artist="Tester",
            genre="lofi", mood="chill",
            energy=0.38, tempo_bpm=76.0, valence=0.62,
            danceability=0.58, acousticness=0.80,
        ),
    ]


_FIXTURE_PROFILE = UserProfile(
    favorite_genre="lofi",
    favorite_mood="chill",
    target_energy=0.40,
    target_tempo_bpm=78.0,
    target_valence=0.60,
    target_danceability=0.60,
    target_acousticness=0.78,
)


_EXTRACTOR_JSON = json.dumps(
    {
        "favorite_genre": "lofi",
        "favorite_mood": "chill",
        "target_energy": 0.40,
        "target_tempo_bpm": 78.0,
        "target_valence": 0.60,
        "target_danceability": 0.60,
        "target_acousticness": 0.78,
    }
)


def _explainer_json(song_ids):
    return json.dumps(
        {
            "explanations": [
                {
                    "song_id": sid,
                    "text": "Fits because lofi sits between 60 and 90 BPM.",
                    "cited_snippets": [
                        "Lofi sits between 60 and 90 BPM",
                        f"Song {sid} fact: track is built for headphones",
                    ],
                }
                for sid in song_ids
            ]
        }
    )


_NL_INPUTS = BuildInputs(
    activity="studying late at night",
    feeling="calm and focused",
    description="Quiet, headphones-on stuff with a vinyl warmth.",
)


# --- build_profile tests -----------------------------------------------------


def test_build_profile_happy_path_critic_ok_on_iter_0():
    """Extractor produces a profile, critic says ok immediately, no refinement."""
    critic_ok = json.dumps({"verdict": "ok", "adjustments": None, "reason": "faithful"})
    llm = StubLLMClient([_EXTRACTOR_JSON, critic_ok])

    result = build_profile(_NL_INPUTS, llm)

    assert isinstance(result, ProfileBuildResult)
    assert result.candidate_profile == _FIXTURE_PROFILE
    assert result.extracted_profile == _FIXTURE_PROFILE
    assert result.inputs == _NL_INPUTS
    assert len(result.refinement_history) == 1
    assert result.refinement_history[0].iter_index == 0
    assert result.refinement_history[0].verdict == "ok"
    assert result.ambiguous_match is False
    assert result.extractor_warnings == []


def test_build_profile_critic_refines_then_oks():
    """Critic refines once with a target shift, then approves on iter 1."""
    critic_refine = json.dumps(
        {
            "verdict": "refine",
            "adjustments": {"target_tempo_bpm": 70.0},
            "reason": "listener said 'slower' but candidate has 78",
        }
    )
    critic_ok = json.dumps({"verdict": "ok", "adjustments": None, "reason": "now faithful"})
    llm = StubLLMClient([_EXTRACTOR_JSON, critic_refine, critic_ok])

    inputs = BuildInputs(description="slower lofi please")
    result = build_profile(inputs, llm)

    assert len(result.refinement_history) == 2
    assert result.refinement_history[0].verdict == "refine"
    assert result.refinement_history[1].verdict == "ok"
    assert result.ambiguous_match is False
    assert result.extracted_profile.target_tempo_bpm == 78.0   # untouched
    assert result.candidate_profile.target_tempo_bpm == 70.0   # adjustment applied


def test_build_profile_ambiguous_match_when_critic_never_oks():
    """Critic refines on every iteration up to the cap; ambiguous flag set."""
    critic_a = json.dumps(
        {"verdict": "refine", "adjustments": {"target_energy": 0.3}, "reason": "still off"}
    )
    critic_b = json.dumps(
        {"verdict": "refine", "adjustments": {"target_energy": 0.2}, "reason": "still off"}
    )
    llm = StubLLMClient([_EXTRACTOR_JSON, critic_a, critic_b])

    inputs = BuildInputs(description="calmer calmer calmer")
    result = build_profile(inputs, llm)

    assert len(result.refinement_history) == MAX_REFINEMENT_ITERS
    assert all(step.verdict == "refine" for step in result.refinement_history)
    assert result.ambiguous_match is True
    assert result.candidate_profile.target_energy == 0.2


def test_build_profile_rejects_empty_inputs():
    """Empty BuildInputs raise EmptyBuildInputsError before any LLM call."""
    llm = StubLLMClient([])

    with pytest.raises(EmptyBuildInputsError):
        build_profile(BuildInputs(), llm)
    with pytest.raises(EmptyBuildInputsError):
        # whitespace-only fields count as empty
        build_profile(BuildInputs(activity="   ", description="\t\n"), llm)


def test_build_profile_with_starting_from_seed():
    """starting_from is forwarded to the extractor (seed appears in prompt)."""
    seed = UserProfile(
        favorite_genre="lofi",
        favorite_mood="chill",
        target_energy=0.5,
        target_tempo_bpm=85.0,
        target_valence=0.5,
        target_danceability=0.5,
        target_acousticness=0.5,
    )
    critic_ok = json.dumps({"verdict": "ok", "adjustments": None, "reason": "faithful"})
    captured_prompts: list[str] = []

    class _CapturingStub(StubLLMClient):
        def generate(self, prompt, *, system=None, temperature=0.2, max_output_tokens=1024):
            captured_prompts.append(prompt)
            return super().generate(
                prompt, system=system, temperature=temperature, max_output_tokens=max_output_tokens
            )

    llm = _CapturingStub([_EXTRACTOR_JSON, critic_ok])

    inputs = BuildInputs(description="same vibe but a touch more upbeat")
    build_profile(inputs, llm, starting_from=seed)

    extractor_prompt = captured_prompts[0]
    assert "An existing profile is provided as a seed" in extractor_prompt
    assert "favorite_genre:       lofi" in extractor_prompt


def test_build_profile_surfaces_extractor_warnings():
    """Unknown genre falls back and the warning surfaces in the result."""
    payload = json.loads(_EXTRACTOR_JSON)
    payload["favorite_genre"] = "drum and bass"   # not in catalog
    extractor_with_unknown = json.dumps(payload)
    critic_ok = json.dumps({"verdict": "ok", "adjustments": None, "reason": "faithful"})
    llm = StubLLMClient([extractor_with_unknown, critic_ok])

    inputs = BuildInputs(description="unknown genre test")
    result = build_profile(inputs, llm)

    assert len(result.extractor_warnings) == 1
    assert "drum and bass" in result.extractor_warnings[0]


# --- recommend tests ---------------------------------------------------------


def test_recommend_happy_path_with_grounded_explanations(tmp_path, monkeypatch):
    songs = _fixture_songs()
    _build_fixture_kb(tmp_path, songs)
    monkeypatch.setattr(retriever_module, "KB_ROOT", tmp_path)

    llm = StubLLMClient([_explainer_json([1, 2, 3])])

    result = recommend(_FIXTURE_PROFILE, llm, songs=songs, k=3)

    assert isinstance(result, RecommendationResult)
    assert result.profile == _FIXTURE_PROFILE
    assert len(result.recommendations) == 3
    assert len(result.retrieved_contexts) == 3
    assert len(result.explanations) == 3
    for expl in result.explanations:
        assert expl.text is not None
        assert expl.fallback_reason is None


def test_recommend_falls_back_to_mechanical_on_fabricated_citation(tmp_path, monkeypatch):
    songs = _fixture_songs()
    _build_fixture_kb(tmp_path, songs)
    monkeypatch.setattr(retriever_module, "KB_ROOT", tmp_path)

    fabricated = json.dumps(
        {
            "explanations": [
                {
                    "song_id": song.id,
                    "text": "Plausible-sounding paragraph.",
                    "cited_snippets": ["this exact string is not in any retrieved doc"],
                }
                for song in songs
            ]
        }
    )
    llm = StubLLMClient([fabricated])

    result = recommend(_FIXTURE_PROFILE, llm, songs=songs, k=3)

    for expl in result.explanations:
        assert expl.text is None
        assert expl.fallback_reason == "fabricated_citation"
