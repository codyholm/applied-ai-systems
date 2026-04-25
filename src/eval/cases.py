"""Eval test cases.

Two batches:
- BASELINE_CASES: 5 NL paraphrases of the named profiles in src/main.py.
  Each deliberately omits exact 7-field numeric values so the extractor
  has to map descriptive language to numbers.
- STRESS_CASES: 5 adversarial NL inputs targeting documented failure
  modes of the extractor or critic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCase:
    name: str
    nl_input: str
    category: str  # "baseline" | "stress"
    baseline_profile_name: str | None


BASELINE_CASES: list[EvalCase] = [
    EvalCase(
        name="high_energy_pop",
        nl_input=(
            "I want fun, upbeat pop for a workout - high energy, "
            "danceable, around the speed of a fast jog."
        ),
        category="baseline",
        baseline_profile_name="high_energy_pop",
    ),
    EvalCase(
        name="chill_lofi",
        nl_input=(
            "Quiet, mellow lofi to focus on writing - slow tempo, "
            "lots of warmth, not too cheerful but not sad."
        ),
        category="baseline",
        baseline_profile_name="chill_lofi",
    ),
    EvalCase(
        name="deep_intense_rock",
        nl_input=(
            "Driving rock for cardio - intense energy, fast tempo, "
            "distorted guitars, dark mood."
        ),
        category="baseline",
        baseline_profile_name="deep_intense_rock",
    ),
    EvalCase(
        name="chill_rock",
        nl_input=(
            "Mellow rock for a slow Sunday morning - soft, low-energy, "
            "acoustic-leaning, calm pace."
        ),
        category="baseline",
        baseline_profile_name="chill_rock",
    ),
    EvalCase(
        name="boundary_maximalist",
        nl_input=(
            "Push everything to the edge - fastest, loudest, "
            "most-electronic dance music you have, no compromise."
        ),
        category="baseline",
        baseline_profile_name="boundary_maximalist",
    ),
]


STRESS_CASES: list[EvalCase] = [
    # Vague: extractor must default to neutral midpoints without crashing.
    EvalCase(
        name="vague_request",
        nl_input="Just give me something I can think to.",
        category="stress",
        baseline_profile_name=None,
    ),
    # Compound: must reconcile high-valence + indie-pop + acoustic-leaning.
    EvalCase(
        name="compound_preference",
        nl_input="Upbeat but not too poppy, kind of indie, with real instruments not synths.",
        category="stress",
        baseline_profile_name=None,
    ),
    # Contradiction: extractor will produce mid-band targets; critic
    # MUST NOT loop forever asking for clarification.
    EvalCase(
        name="contradiction",
        nl_input="Calm but high-energy, sad but fun.",
        category="stress",
        baseline_profile_name=None,
    ),
    # Tempo-anchored: extractor must honor the explicit BPM number.
    EvalCase(
        name="tempo_anchored",
        nl_input="Around 90 BPM, jazzy, late-night feel.",
        category="stress",
        baseline_profile_name=None,
    ),
    # Mood-only: minimal information; pipeline must still produce 5 results
    # without raising or hitting the refinement cap.
    EvalCase(
        name="mood_only",
        nl_input="Just sad stuff.",
        category="stress",
        baseline_profile_name=None,
    ),
]


ALL_CASES: list[EvalCase] = BASELINE_CASES + STRESS_CASES
