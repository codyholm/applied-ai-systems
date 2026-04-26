from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from src.agents.critic import critique
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
}


@dataclass
class RefinementStep:
    iter_index: int
    verdict: str
    top5_song_ids: list[int]
    adjustments_applied: dict | None
    reason: str


@dataclass
class PipelineResult:
    nl_input: str
    extracted_profile: UserProfile
    final_profile: UserProfile
    recommendations: list[ScoredRecommendation]
    retrieved_contexts: list[RetrievedContext]
    explanations: list[Explanation]
    refinement_history: list[RefinementStep] = field(default_factory=list)
    ambiguous_match: bool = False
    cache_stats: dict[str, int] | None = None
    extractor_warnings: list[str] = field(default_factory=list)


def _cache_stats(llm: LLMClient) -> dict[str, int] | None:
    if isinstance(llm, CachedLLMClient):
        return {"hits": llm.hits, "misses": llm.misses}
    return None


def _apply_adjustments(profile: UserProfile, adjustments: dict[str, Any]) -> UserProfile:
    safe = {k: v for k, v in adjustments.items() if k in _PROFILE_FIELDS}
    if not safe:
        return profile
    return replace(profile, **safe)


def run_pipeline(
    nl_input: str,
    *,
    llm: LLMClient,
    songs: list[Song] | None = None,
    k: int = 5,
) -> PipelineResult:
    catalog = songs if songs is not None else load_songs()

    extracted_profile, extractor_warnings = extract_profile(nl_input, llm)

    current_profile = extracted_profile
    refinement_history: list[RefinementStep] = []
    final_recommendations: list[ScoredRecommendation] = []
    last_verdict = "ok"

    for iter_index in range(MAX_REFINEMENT_ITERS):
        recommendations = recommend_songs(current_profile, catalog, k=k)
        verdict = critique(nl_input, current_profile, recommendations, llm)
        refinement_history.append(
            RefinementStep(
                iter_index=iter_index,
                verdict=verdict.verdict,
                top5_song_ids=[r.song.id for r in recommendations],
                adjustments_applied=dict(verdict.adjustments) if verdict.adjustments else None,
                reason=verdict.reason,
            )
        )
        last_verdict = verdict.verdict
        final_recommendations = recommendations

        if verdict.verdict == "ok":
            break
        if verdict.adjustments:
            current_profile = _apply_adjustments(current_profile, verdict.adjustments)

    ambiguous_match = last_verdict == "refine"

    retrieved_contexts = [retrieve_for_recommendation(rec) for rec in final_recommendations]
    explanations = explain_recommendations(
        current_profile, final_recommendations, retrieved_contexts, llm
    )

    return PipelineResult(
        nl_input=nl_input,
        extracted_profile=extracted_profile,
        final_profile=current_profile,
        recommendations=final_recommendations,
        retrieved_contexts=retrieved_contexts,
        explanations=explanations,
        refinement_history=refinement_history,
        ambiguous_match=ambiguous_match,
        cache_stats=_cache_stats(llm),
        extractor_warnings=extractor_warnings,
    )
