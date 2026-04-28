"""Unit tests for the build-eval neighborhood assertion."""

from __future__ import annotations

from src.eval.assertions import assert_build_neighborhood
from src.eval.cases import BuildCase
from src.pipeline import BuildInputs, ProfileBuildResult
from src.recommender import UserProfile


def _expected() -> UserProfile:
    return UserProfile(
        favorite_genre="acoustic",
        favorite_mood="relaxed",
        target_energy=0.30,
        target_tempo_bpm=80.0,
        target_acousticness=0.85,
        target_valence=0.65,
        target_danceability=0.25,
        avoid_genres=["electronic", "synthwave"],
    )


def _case() -> BuildCase:
    return BuildCase(
        name="acoustic_morning_test",
        inputs=BuildInputs(description="quiet acoustic, no electronic"),
        expected_profile=_expected(),
    )


def _result(candidate: UserProfile) -> ProfileBuildResult:
    return ProfileBuildResult(
        inputs=BuildInputs(description="ignored"),
        candidate_profile=candidate,
        extracted_profile=candidate,
    )


def test_assert_build_neighborhood_passes_when_avoid_genres_match():
    candidate = _expected()
    ok, failures = assert_build_neighborhood(_case(), _result(candidate))
    assert ok
    assert failures == []


def test_assert_build_neighborhood_fails_when_avoid_genres_missing():
    candidate = UserProfile(
        favorite_genre="acoustic",
        favorite_mood="relaxed",
        target_energy=0.30,
        target_tempo_bpm=80.0,
        target_acousticness=0.85,
        target_valence=0.65,
        target_danceability=0.25,
        avoid_genres=[],   # extractor missed the avoid signal
    )
    ok, failures = assert_build_neighborhood(_case(), _result(candidate))
    assert not ok
    assert any("avoid_genres mismatch" in f for f in failures)


def test_assert_build_neighborhood_fails_when_avoid_genres_extra():
    candidate = UserProfile(
        favorite_genre="acoustic",
        favorite_mood="relaxed",
        target_energy=0.30,
        target_tempo_bpm=80.0,
        target_acousticness=0.85,
        target_valence=0.65,
        target_danceability=0.25,
        avoid_genres=["electronic", "synthwave", "pop"],   # over-extracted
    )
    ok, failures = assert_build_neighborhood(_case(), _result(candidate))
    assert not ok
    assert any("avoid_genres mismatch" in f for f in failures)


def test_assert_build_neighborhood_avoid_check_is_case_insensitive():
    candidate = UserProfile(
        favorite_genre="acoustic",
        favorite_mood="relaxed",
        target_energy=0.30,
        target_tempo_bpm=80.0,
        target_acousticness=0.85,
        target_valence=0.65,
        target_danceability=0.25,
        avoid_genres=["Electronic", "SYNTHWAVE"],
    )
    ok, _failures = assert_build_neighborhood(_case(), _result(candidate))
    assert ok
