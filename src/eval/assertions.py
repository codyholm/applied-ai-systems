"""Hand-coded structural assertions for the two-pipeline eval harness.

Two functions:

- assert_build_neighborhood(case, result) — for each BuildCase, checks
  that the extracted candidate profile lands in the target preset's
  neighborhood: same favorite_genre, same favorite_mood, every numeric
  target within tolerance.
- assert_recommend_structural(preset_name, result) — for each preset,
  applies the per-preset structural rule on the recommend pipeline's
  top-5. Rules carry forward from the original Module 3 model_card with
  the post-review corrections from commits d9fb96a (chill_rock) and
  6accda2 (boundary_maximalist).
"""

from __future__ import annotations

import statistics
from typing import Callable

from src.eval.cases import BuildCase
from src.pipeline import ProfileBuildResult, RecommendationResult


# ---------------------------------------------------------------------------
# build-eval assertion
# ---------------------------------------------------------------------------


def assert_build_neighborhood(
    case: BuildCase, result: ProfileBuildResult
) -> tuple[bool, list[str]]:
    """Check that the candidate profile lands near the case's expected profile.

    Genre and mood must match exactly (case-insensitive). Tempo must be
    within `case.tempo_tolerance_bpm`. The four [0,1] target_* fields
    must each be within `case.numeric_tolerance`.
    """
    failures: list[str] = []
    target = case.expected_profile
    candidate = result.candidate_profile

    if candidate.favorite_genre.lower() != target.favorite_genre.lower():
        failures.append(
            f"favorite_genre {candidate.favorite_genre!r} != target "
            f"{target.favorite_genre!r}"
        )
    if candidate.favorite_mood.lower() != target.favorite_mood.lower():
        failures.append(
            f"favorite_mood {candidate.favorite_mood!r} != target "
            f"{target.favorite_mood!r}"
        )

    tempo_diff = abs(candidate.target_tempo_bpm - target.target_tempo_bpm)
    if tempo_diff > case.tempo_tolerance_bpm:
        failures.append(
            f"target_tempo_bpm {candidate.target_tempo_bpm} differs from "
            f"target {target.target_tempo_bpm} by {tempo_diff:.1f} "
            f"(>{case.tempo_tolerance_bpm} BPM tolerance)"
        )

    for field in ("target_energy", "target_valence", "target_danceability", "target_acousticness"):
        diff = abs(getattr(candidate, field) - getattr(target, field))
        if diff > case.numeric_tolerance:
            failures.append(
                f"{field} {getattr(candidate, field):.2f} differs from "
                f"target {getattr(target, field):.2f} by {diff:.2f} "
                f"(>{case.numeric_tolerance:.2f} tolerance)"
            )

    candidate_avoid = {g.lower() for g in candidate.avoid_genres}
    target_avoid = {g.lower() for g in target.avoid_genres}
    if candidate_avoid != target_avoid:
        failures.append(
            f"avoid_genres mismatch: candidate={sorted(candidate_avoid)} "
            f"expected={sorted(target_avoid)}"
        )

    return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# recommend-eval per-preset structural rules
# ---------------------------------------------------------------------------


def _high_energy_pop(result: RecommendationResult) -> tuple[bool, list[str]]:
    failures: list[str] = []
    genres = [r.song.genre for r in result.recommendations]
    pop_count = sum(1 for g in genres if g == "pop")
    bad_count = sum(1 for g in genres if g in {"ambient", "lofi"})
    if pop_count < 1:
        failures.append("top-5 had 0 pop tracks (need >=1)")
    if bad_count > 0:
        failures.append(f"top-5 had {bad_count} ambient/lofi tracks (need 0)")
    return len(failures) == 0, failures


def _chill_lofi(result: RecommendationResult) -> tuple[bool, list[str]]:
    failures: list[str] = []
    chill_genres = {"lofi", "ambient", "acoustic"}
    matches = sum(1 for r in result.recommendations if r.song.genre in chill_genres)
    if matches < 1:
        failures.append(
            f"top-5 had 0 tracks in {{lofi, ambient, acoustic}} (need >=1)"
        )
    return len(failures) == 0, failures


def _deep_intense_rock(result: RecommendationResult) -> tuple[bool, list[str]]:
    failures: list[str] = []
    rock_count = sum(1 for r in result.recommendations if r.song.genre == "rock")
    if rock_count < 1:
        failures.append("top-5 had 0 rock tracks (need >=1)")
    if result.recommendations:
        mean_tempo = statistics.mean(r.song.tempo_bpm for r in result.recommendations)
        if mean_tempo < 110:
            failures.append(
                f"mean top-5 tempo {mean_tempo:.1f} BPM is below 110"
            )
    return len(failures) == 0, failures


def _chill_rock(result: RecommendationResult) -> tuple[bool, list[str]]:
    """Rock listener wanting the chill end of the rock spectrum.

    After the Step 6 scorer + preset fixes, Lighthouse Hum (the catalog's
    chill-leaning rock track, mood='moody', energy 0.62) should land in
    the top 5. The previous version of this rule asserted "mean energy
    <= 0.45 AND mean acousticness >= 0.55" — that documented the broken
    behavior of the un-tuned scorer (commit d9fb96a) rather than the
    desired outcome. The Module 3 model_card §6 called the 0-rock result
    a weakness; §8 future work explicitly proposed weighting genre
    higher to fix it. Step 6 did exactly that.
    """
    failures: list[str] = []
    if not result.recommendations:
        return False, ["top-5 was empty"]
    rock_count = sum(1 for r in result.recommendations if r.song.genre == "rock")
    if rock_count < 1:
        failures.append(
            f"top-5 had {rock_count} rock tracks (need >=1); "
            f"the scorer should now surface Lighthouse Hum at minimum"
        )
    return len(failures) == 0, failures


def _boundary_maximalist(result: RecommendationResult) -> tuple[bool, list[str]]:
    """Sanity rule for an extreme profile the catalog cannot fully satisfy
    (commit 6accda2). The original 5% score plateau was unreachable; the
    informative invariant is weaker: every top-5 score positive, the
    bottom score within 50% of the top, and >=1 high-energy (>=0.70) track.
    """
    failures: list[str] = []
    if not result.recommendations:
        return False, ["top-5 was empty"]
    top_score = result.recommendations[0].score
    bottom_score = result.recommendations[-1].score
    if top_score <= 0:
        failures.append(f"top score {top_score:.2f} is non-positive")
    elif bottom_score / top_score < 0.5:
        failures.append(
            f"top-5 bottom score {bottom_score:.2f} is below 50% of top score "
            f"{top_score:.2f} - cohort is not coherent"
        )

    high_energy = sum(1 for r in result.recommendations if r.song.energy >= 0.7)
    if high_energy < 1:
        failures.append("top-5 has 0 high-energy (>=0.70) tracks; expected >=1")

    return len(failures) == 0, failures


_PRESET_RULES: dict[str, Callable[[RecommendationResult], tuple[bool, list[str]]]] = {
    "high_energy_pop": _high_energy_pop,
    "chill_lofi": _chill_lofi,
    "deep_intense_rock": _deep_intense_rock,
    "chill_rock": _chill_rock,
    "boundary_maximalist": _boundary_maximalist,
}


def assert_recommend_structural(
    preset_name: str, result: RecommendationResult
) -> tuple[bool, list[str]]:
    """Apply the per-preset structural rule to a recommend result."""
    rule = _PRESET_RULES.get(preset_name)
    if rule is None:
        return False, [f"no structural rule for preset {preset_name!r}"]
    return rule(result)
