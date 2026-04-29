"""Two-pipeline orchestration for the applied AI music recommender.

The system splits user-visible work into two independent pipelines:

- build_profile(inputs, llm, *, starting_from=None) -> ProfileBuildResult
    Profile-creation pipeline. Validates the BuildInputs bundle (5
    question answers + free-form description), runs the LLM Extractor +
    Critic refinement loop to translate it into a 7-dimension UserProfile
    that conforms to the system's standard. The user (or caller) decides
    whether to save the candidate profile via src.profiles.save_profile.
    Run rarely, deliberately.

- recommend(profile, llm, *, songs=None, k=5) -> RecommendationResult
    Recommendation pipeline. Takes a profile (saved, preset, or
    just-built) and returns top-k scored recommendations with KB-grounded
    LLM explanations. No critic, no extractor — pure forward pass through
    the deterministic scorer + RAG explainer. Run anytime.

The split mirrors the user's mental model: profile creation is the
agentic AI feature; recommendation is the deterministic search system
with grounded explanations layered on top. RAG retrieval lives only in
the recommend pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from src.agents.critic import critique_extraction
from src.agents.explainer import Explanation, explain_recommendations
from src.agents.profile_extractor import extract_profile
from src.kb.retriever import RetrievedContext, retrieve_for_recommendation
from src.llm.client import CachedLLMClient, LLMClient
from src.recommender import (
    ScoredRecommendation,
    Song,
    UserProfile,
    load_songs,
    recommend_songs,
)


MAX_REFINEMENT_ITERS = 2

_PROFILE_FIELDS = {
    "favorite_genre",
    "favorite_mood",
    "target_energy",
    "target_tempo_bpm",
    "target_valence",
    "target_danceability",
    "target_acousticness",
    "avoid_genres",
}


class EmptyBuildInputsError(ValueError):
    """Raised when build_profile is called with no usable inputs."""


@dataclass
class BuildInputs:
    """Hybrid input bundle: 3 question answers + an optional free-form description.

    Each field is a short natural-language string. The three questions
    cover what the listener is doing (activity), what kinds of sounds
    they want (instruments), and which genres they want or want to avoid
    (genres). The description field carries any free-form text the
    listener wants to add.

    Validation gate: at least one field must be non-blank before the
    bundle is sent to the build pipeline (build_profile raises
    EmptyBuildInputsError otherwise).
    """

    activity:    str | None = None
    instruments: str | None = None
    genres:      str | None = None
    description: str | None = None

    def is_empty(self) -> bool:
        """All four fields are None or whitespace-only."""
        return not any(
            (v or "").strip()
            for v in (
                self.activity, self.instruments, self.genres, self.description,
            )
        )

    def has_minimum(self) -> bool:
        """At least one field has non-whitespace content."""
        return not self.is_empty()


@dataclass
class RefinementStep:
    """One iteration of the build-pipeline refinement loop."""

    iter_index: int
    verdict: str
    candidate_after_iter: dict
    adjustments_applied: dict | None
    reason: str


@dataclass
class ProfileBuildResult:
    """Output of build_profile().

    Carries the listener's inputs (for replay), the candidate profile
    (final, possibly refined), the original extraction before refinement,
    the refinement loop's history, the ambiguous flag (True iff the loop
    hit the cap without the critic agreeing), any extractor warnings
    (e.g. unknown genre fallbacks), the suggested save-as name the
    extractor proposed in the same response, and cache stats.
    """

    inputs: BuildInputs
    candidate_profile: UserProfile
    extracted_profile: UserProfile
    refinement_history: list[RefinementStep] = field(default_factory=list)
    ambiguous_match: bool = False
    extractor_warnings: list[str] = field(default_factory=list)
    suggested_name: str = "My Vibe Profile"
    cache_stats: dict[str, int] | None = None


@dataclass
class RecommendationResult:
    """Output of recommend()."""

    profile: UserProfile
    recommendations: list[ScoredRecommendation]
    retrieved_contexts: list[RetrievedContext]
    explanations: list[Explanation]
    cache_stats: dict[str, int] | None = None


def _cache_stats(llm: LLMClient) -> dict[str, int] | None:
    if isinstance(llm, CachedLLMClient):
        return {"hits": llm.hits, "misses": llm.misses}
    return None


def _apply_adjustments(profile: UserProfile, adjustments: dict[str, Any]) -> UserProfile:
    safe = {k: v for k, v in adjustments.items() if k in _PROFILE_FIELDS}
    if not safe:
        return profile
    return replace(profile, **safe)


def _profile_snapshot(profile: UserProfile) -> dict:
    return {
        "favorite_genre": profile.favorite_genre,
        "favorite_mood": profile.favorite_mood,
        "target_energy": profile.target_energy,
        "target_tempo_bpm": profile.target_tempo_bpm,
        "target_valence": profile.target_valence,
        "target_danceability": profile.target_danceability,
        "target_acousticness": profile.target_acousticness,
        "avoid_genres": list(profile.avoid_genres),
    }


def build_profile(
    inputs: BuildInputs,
    llm: LLMClient,
    *,
    starting_from: UserProfile | None = None,
) -> ProfileBuildResult:
    """Build a UserProfile from the listener's question answers + optional description.

    Flow:
      1. Validate inputs.has_minimum() — raise EmptyBuildInputsError if not.
      2. extract_profile(inputs, llm, starting_from=starting_from) -> candidate + warnings.
      3. Loop up to MAX_REFINEMENT_ITERS times:
           - critique_extraction(inputs, candidate, llm)
           - if verdict == 'ok': break
           - else apply adjustments and continue
      4. ambiguous_match = True iff the loop exited by hitting the cap
         (final verdict was 'refine').

    The extractor itself can raise ProfileExtractionError on consecutive
    JSON parse failures; that propagates so callers can render a
    "try rephrasing" UX.
    """
    if not inputs.has_minimum():
        raise EmptyBuildInputsError(
            "build_profile requires at least one non-blank field "
            "(answer one question or fill the description)."
        )

    extracted_profile, extractor_warnings, suggested_name = extract_profile(
        inputs, llm, starting_from=starting_from
    )

    current_profile = extracted_profile
    refinement_history: list[RefinementStep] = []
    last_verdict = "ok"

    for iter_index in range(MAX_REFINEMENT_ITERS):
        verdict = critique_extraction(inputs, current_profile, llm)
        if verdict.verdict == "refine" and verdict.adjustments:
            current_profile = _apply_adjustments(current_profile, verdict.adjustments)
        refinement_history.append(
            RefinementStep(
                iter_index=iter_index,
                verdict=verdict.verdict,
                candidate_after_iter=_profile_snapshot(current_profile),
                adjustments_applied=dict(verdict.adjustments) if verdict.adjustments else None,
                reason=verdict.reason,
            )
        )
        last_verdict = verdict.verdict
        if verdict.verdict == "ok":
            break

    ambiguous_match = last_verdict == "refine"

    return ProfileBuildResult(
        inputs=inputs,
        candidate_profile=current_profile,
        extracted_profile=extracted_profile,
        refinement_history=refinement_history,
        ambiguous_match=ambiguous_match,
        extractor_warnings=extractor_warnings,
        suggested_name=suggested_name,
        cache_stats=_cache_stats(llm),
    )


def recommend(
    profile: UserProfile,
    llm: LLMClient,
    *,
    songs: list[Song] | None = None,
    k: int = 5,
) -> RecommendationResult:
    """Rank the catalog against a profile and produce grounded explanations.

    Flow:
      1. recommend_songs(profile, catalog, k) -> top-k ScoredRecommendations
      2. retrieve_for_recommendation(rec) per top-k -> RetrievedContext list
      3. explain_recommendations(profile, recs, contexts, llm) -> Explanations

    No critic, no refinement. The profile is treated as already valid
    (it came from build_profile, save_profile, or a preset). RAG
    grounding only happens here, not in the build pipeline.
    """
    catalog = songs if songs is not None else load_songs()
    recommendations = recommend_songs(profile, catalog, k=k)
    retrieved_contexts = [retrieve_for_recommendation(rec) for rec in recommendations]
    explanations = explain_recommendations(profile, recommendations, retrieved_contexts, llm)

    return RecommendationResult(
        profile=profile,
        recommendations=recommendations,
        retrieved_contexts=retrieved_contexts,
        explanations=explanations,
        cache_stats=_cache_stats(llm),
    )
