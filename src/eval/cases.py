"""Eval test cases for the two-pipeline architecture.

Two collections, run independently by the eval harness:

- BUILD_CASES: 10 cases (5 baseline + 5 stress) feeding the build_profile
  pipeline. Each case carries a typed BuildInputs bundle (the listener's
  five question-answers + free-form description) and a target_preset
  whose neighborhood the extracted profile should land within.
- RECOMMEND_CASES: 5 preset names. Each name is run directly through
  the recommend pipeline; the per-preset structural rules in
  src/eval/assertions.py validate the top-5.

D39 split rationale: the previous single-pipeline harness conflated
profile extraction quality and recommendation quality. Splitting them
gives independently actionable signal — build-eval surfaces extractor
faithfulness, recommend-eval surfaces RAG + scoring quality.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.pipeline import BuildInputs


@dataclass(frozen=True)
class BuildCase:
    """One build-pipeline test case.

    Attributes:
      name: short human-readable identifier (e.g. 'high_energy_pop_paraphrase').
      inputs: the listener's BuildInputs bundle.
      target_preset: key of the PRESET_PROFILES entry the extracted
        profile should land in the neighborhood of.
      tempo_tolerance_bpm: max |Δ tempo| allowed against the preset.
      numeric_tolerance: max |Δ| allowed on the four [0,1]-valued targets
        (energy, valence, danceability, acousticness).
      category: 'baseline' (paraphrase of the preset's intent) or
        'stress' (adversarial — vague, contradictory, or under-specified
        inputs that should still land in some sensible neighborhood).
    """

    name: str
    inputs: BuildInputs
    target_preset: str
    tempo_tolerance_bpm: float = 30.0
    numeric_tolerance: float = 0.20
    category: str = "baseline"


# Note: target_preset choices for stress cases reflect the closest
# preset under the system's defaults; stress cases CAN fail the
# neighborhood check — that is the point. The failure is the eval signal.
BUILD_CASES: list[BuildCase] = [
    # ---- baseline paraphrases (5) -------------------------------------
    BuildCase(
        name="high_energy_pop_paraphrase",
        inputs=BuildInputs(
            activity="workout, fast jog",
            feeling="upbeat and fun",
            movement="moving and dancing",
            instruments="doesn't matter, just energetic production",
            genres="pop",
            description=(
                "Fun, upbeat pop for a workout - high energy, danceable, "
                "around the speed of a fast jog."
            ),
        ),
        target_preset="high_energy_pop",
    ),
    BuildCase(
        name="chill_lofi_paraphrase",
        inputs=BuildInputs(
            activity="focused writing session",
            feeling="quiet and calm, neutral mood",
            movement="sitting still",
            instruments="warm acoustic and lofi production",
            genres="lofi",
            description=(
                "Quiet, mellow lofi to focus on writing - slow tempo, "
                "lots of warmth, not too cheerful but not sad."
            ),
        ),
        target_preset="chill_lofi",
    ),
    BuildCase(
        name="deep_intense_rock_paraphrase",
        inputs=BuildInputs(
            activity="cardio workout",
            feeling="intense and dark",
            movement="moving fast",
            instruments="distorted guitars, heavy production",
            genres="rock",
            description=(
                "Driving rock for cardio - intense energy, fast tempo, "
                "distorted guitars, dark mood."
            ),
        ),
        target_preset="deep_intense_rock",
    ),
    BuildCase(
        name="chill_rock_paraphrase",
        inputs=BuildInputs(
            activity="slow Sunday morning, light reading",
            feeling="calm and gentle",
            movement="sitting still",
            instruments="acoustic-leaning, soft production",
            genres="rock",
            description=(
                "Mellow rock for a slow Sunday morning - soft, "
                "low-energy, acoustic-leaning, calm pace."
            ),
        ),
        target_preset="chill_rock",
    ),
    BuildCase(
        name="boundary_maximalist_paraphrase",
        inputs=BuildInputs(
            activity="high-intensity dance party",
            feeling="wild and euphoric",
            movement="moving full out",
            instruments="fully electronic and synthesized",
            genres="electronic",
            description=(
                "Push everything to the edge - fastest, loudest, "
                "most-electronic dance music you have, no compromise."
            ),
        ),
        target_preset="boundary_maximalist",
    ),
    # ---- stress (5) ---------------------------------------------------
    # Vague: extractor must default to neutral midpoints without crashing.
    BuildCase(
        name="vague_request",
        inputs=BuildInputs(
            description="Just give me something I can think to.",
        ),
        target_preset="chill_lofi",
        category="stress",
    ),
    # Compound: must reconcile high valence + indie pop + acoustic.
    BuildCase(
        name="compound_preference",
        inputs=BuildInputs(
            activity="casual listening",
            feeling="upbeat but not too poppy",
            instruments="real instruments, not synths",
            genres="indie pop",
            description=(
                "Upbeat but not too poppy, kind of indie, with real "
                "instruments not synths."
            ),
        ),
        target_preset="high_energy_pop",
        category="stress",
    ),
    # Contradiction: extractor will land at mid-band targets; critic
    # MUST NOT loop forever asking for clarification.
    BuildCase(
        name="contradiction",
        inputs=BuildInputs(
            feeling="calm but high-energy, sad but fun",
            description="Calm but high-energy, sad but fun.",
        ),
        target_preset="chill_lofi",
        category="stress",
    ),
    # Tempo-anchored: extractor must honor the explicit BPM number.
    BuildCase(
        name="tempo_anchored",
        inputs=BuildInputs(
            activity="late-night listening",
            feeling="smooth, jazzy, late-night",
            genres="jazz",
            description="Around 90 BPM, jazzy, late-night feel.",
        ),
        target_preset="chill_lofi",
        category="stress",
    ),
    # Mood-only: minimal information; pipeline must still produce a
    # profile without raising or hitting the refinement cap.
    BuildCase(
        name="mood_only",
        inputs=BuildInputs(
            feeling="sad",
            description="Just sad stuff.",
        ),
        target_preset="chill_lofi",
        category="stress",
    ),
]


# RECOMMEND_CASES: each preset name is run directly through `recommend`.
# Order matches PRESET_PROFILES so the scorecard reads predictably.
RECOMMEND_CASES: list[str] = [
    "high_energy_pop",
    "chill_lofi",
    "deep_intense_rock",
    "chill_rock",
    "boundary_maximalist",
]
