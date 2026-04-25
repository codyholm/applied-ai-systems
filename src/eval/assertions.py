"""Per-case structural assertions.

Each baseline case has a hand-coded boolean rule about properties of the
top-5. Stress cases share one rule: the system must converge gracefully
(not flag ambiguous and not exceed the refinement cap).
"""

from __future__ import annotations

import statistics
from typing import Callable

from src.eval.cases import EvalCase
from src.pipeline import PipelineResult


def _high_energy_pop(result: PipelineResult) -> tuple[bool, list[str]]:
    failures: list[str] = []
    genres = [r.song.genre for r in result.recommendations]
    pop_count = sum(1 for g in genres if g == "pop")
    bad_count = sum(1 for g in genres if g in {"ambient", "lofi"})
    if pop_count < 1:
        failures.append(f"top-5 had 0 pop tracks (need >=1)")
    if bad_count > 0:
        failures.append(f"top-5 had {bad_count} ambient/lofi tracks (need 0)")
    return len(failures) == 0, failures


def _chill_lofi(result: PipelineResult) -> tuple[bool, list[str]]:
    failures: list[str] = []
    chill_genres = {"lofi", "ambient", "acoustic"}
    matches = sum(1 for r in result.recommendations if r.song.genre in chill_genres)
    if matches < 1:
        failures.append(
            f"top-5 had 0 tracks in {{lofi, ambient, acoustic}} (need >=1)"
        )
    return len(failures) == 0, failures


def _deep_intense_rock(result: PipelineResult) -> tuple[bool, list[str]]:
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


def _chill_rock(result: PipelineResult) -> tuple[bool, list[str]]:
    failures: list[str] = []
    rock_count = sum(1 for r in result.recommendations if r.song.genre == "rock")
    if rock_count < 1:
        failures.append("top-5 had 0 rock tracks (need >=1)")
    if result.recommendations:
        mean_energy = statistics.mean(r.song.energy for r in result.recommendations)
        if mean_energy > 0.6:
            failures.append(f"mean top-5 energy {mean_energy:.2f} is above 0.6")
    return len(failures) == 0, failures


def _boundary_maximalist(result: PipelineResult) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if not result.recommendations:
        return False, ["top-5 was empty"]
    top_score = result.recommendations[0].score
    bottom_score = result.recommendations[-1].score
    if top_score <= 0:
        failures.append(f"top score {top_score:.2f} is non-positive")
    else:
        spread = (top_score - bottom_score) / top_score
        if spread > 0.05:
            failures.append(
                f"top-5 score spread {spread:.3f} exceeds 0.05 (no plateau)"
            )
    return len(failures) == 0, failures


def _stress_default(result: PipelineResult) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if result.ambiguous_match:
        failures.append("ambiguous_match flag is set")
    if len(result.refinement_history) > 2:
        failures.append(
            f"refinement_history length {len(result.refinement_history)} exceeds 2"
        )
    return len(failures) == 0, failures


BASELINE_RULES: dict[str, Callable[[PipelineResult], tuple[bool, list[str]]]] = {
    "high_energy_pop": _high_energy_pop,
    "chill_lofi": _chill_lofi,
    "deep_intense_rock": _deep_intense_rock,
    "chill_rock": _chill_rock,
    "boundary_maximalist": _boundary_maximalist,
}


def assert_structural(
    case: EvalCase, result: PipelineResult
) -> tuple[bool, list[str]]:
    if case.category == "baseline":
        rule = BASELINE_RULES.get(case.baseline_profile_name or "")
        if rule is None:
            return False, [
                f"no structural rule for baseline_profile_name={case.baseline_profile_name!r}"
            ]
        return rule(result)
    return _stress_default(result)
