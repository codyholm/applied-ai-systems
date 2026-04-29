"""Eval test cases for the two-pipeline architecture.

Two collections, run independently by the eval harness:

- BUILD_CASES: 5 cases feeding the build_profile pipeline. Each case
  carries a typed BuildInputs bundle (the listener's three question-
  answers + free-form description) and an expected UserProfile authored
  by hand. The cases describe FRESH listener personas — they deliberately
  don't paraphrase the existing presets. Build-eval verifies the build
  pipeline can produce a profile in the neighborhood of the expected
  ground truth from a natural-language description.
- RECOMMEND_CASES: 5 preset names. Each name is run directly through
  the recommend pipeline; the per-preset structural rules in
  src/eval/assertions.py validate the top-5 against the preset profile.

D39 split rationale: the previous single-pipeline harness conflated
profile extraction quality and recommendation quality. Splitting them
gives independently actionable signal — build-eval surfaces extractor
faithfulness on novel listener descriptions, recommend-eval surfaces
RAG + scoring quality on the canonical presets.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.pipeline import BuildInputs
from src.recommender import UserProfile


@dataclass(frozen=True)
class BuildCase:
    """One build-pipeline test case.

    Attributes:
      name: short human-readable identifier (e.g. 'jazz_focus').
      inputs: the listener's BuildInputs bundle (3 questions + description).
      expected_profile: hand-authored ground-truth UserProfile that the
        build pipeline should produce for this listener. Neighborhood
        check happens against this, not against a preset.
      tempo_tolerance_bpm: max |Δ tempo| allowed against the expected.
      numeric_tolerance: max |Δ| allowed on the four [0,1]-valued targets
        (energy, valence, danceability, acousticness).
      category: 'baseline' (clear listener with all fields filled) or
        'stress' (pushes the limits — extreme values, edge-of-catalog
        territory — without being designed to fail).
    """

    name: str
    inputs: BuildInputs
    expected_profile: UserProfile
    tempo_tolerance_bpm: float = 30.0
    numeric_tolerance: float = 0.20
    category: str = "baseline"


# Five fresh listener personas. Genres deliberately chosen to NOT
# overlap the five preset genres (pop, lofi, rock, electronic) so
# build-eval and recommend-eval cover different territory.
BUILD_CASES: list[BuildCase] = [
    BuildCase(
        name="jazz_focus",
        inputs=BuildInputs(
            activity="late-night studying or quiet writing",
            instruments="real instruments — jazz quartet, warm bass",
            genres="jazz",
            description=(
                "Smooth late-night jazz for focus. Low energy but not "
                "sad, quiet drums and warm bass, no vocals."
            ),
        ),
        expected_profile=UserProfile(
            favorite_genre="jazz",
            favorite_mood="focused",
            target_energy=0.35,
            target_tempo_bpm=85.0,
            target_acousticness=0.65,
            target_valence=0.45,
            target_danceability=0.30,
        ),
    ),
    BuildCase(
        name="hip_hop_cardio",
        inputs=BuildInputs(
            activity="treadmill cardio, sprint intervals",
            instruments="heavy beats, sub bass, hip hop production",
            genres="hip hop",
            description=(
                "High-energy hip hop for sprint intervals — fast tempo, "
                "hard-hitting beats, aggressive production."
            ),
        ),
        expected_profile=UserProfile(
            favorite_genre="hip hop",
            favorite_mood="intense",
            target_energy=0.85,
            target_tempo_bpm=140.0,
            target_acousticness=0.10,
            target_valence=0.60,
            target_danceability=0.85,
        ),
    ),
    BuildCase(
        name="synthwave_drive",
        inputs=BuildInputs(
            activity="long evening drives on the highway",
            instruments="synthesizers, retro electronic production",
            genres="synthwave",
            description=(
                "Dreamy synthwave for night drives — moody, mid-tempo, "
                "washes of analog synth, neon-lit atmosphere, nostalgic."
            ),
        ),
        expected_profile=UserProfile(
            favorite_genre="synthwave",
            favorite_mood="moody",
            target_energy=0.55,
            target_tempo_bpm=105.0,
            target_acousticness=0.20,
            target_valence=0.45,
            target_danceability=0.55,
        ),
    ),
    BuildCase(
        name="acoustic_morning",
        inputs=BuildInputs(
            activity="slow weekend mornings, making coffee",
            instruments="fingerpicked acoustic guitar, soft vocals",
            genres="acoustic, definitely no electronic or synthwave",
            description=(
                "Quiet, warm acoustic for slow mornings — fingerpicked "
                "guitar, gentle vocals, vinyl warmth, no drums, low "
                "energy. Please avoid anything electronic or synthwave; "
                "I want it to feel unplugged."
            ),
        ),
        expected_profile=UserProfile(
            favorite_genre="acoustic",
            favorite_mood="relaxed",
            target_energy=0.30,
            target_tempo_bpm=80.0,
            target_acousticness=0.85,
            target_valence=0.65,
            target_danceability=0.25,
            avoid_genres=["electronic", "synthwave"],
        ),
    ),
    BuildCase(
        # Pushes the limits — extremely low energy / very slow tempo /
        # no percussion. Not designed to fail; designed to test whether
        # the extractor honors a request at the catalog's outer edge.
        name="ambient_meditation",
        inputs=BuildInputs(
            activity="meditation and deep breathing, completely still",
            instruments="drone, atmospheric pads, no percussion at all",
            genres="ambient",
            description=(
                "Very slow ambient drone for meditation — minimal, no "
                "beat, deeply still, tranquil, like floating."
            ),
        ),
        expected_profile=UserProfile(
            favorite_genre="ambient",
            favorite_mood="chill",
            target_energy=0.10,
            target_tempo_bpm=60.0,
            target_acousticness=0.50,
            target_valence=0.40,
            target_danceability=0.10,
        ),
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
