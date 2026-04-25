from __future__ import annotations

from dataclasses import dataclass, field

from src.agents.explainer import Explanation, explain_recommendations
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


def _cache_stats(llm: LLMClient) -> dict[str, int] | None:
    if isinstance(llm, CachedLLMClient):
        return {"hits": llm.hits, "misses": llm.misses}
    return None


def run_pipeline(
    profile: UserProfile,
    *,
    llm: LLMClient,
    songs: list[Song] | None = None,
    k: int = 5,
) -> PipelineResult:
    catalog = songs if songs is not None else load_songs()
    recommendations = recommend_songs(profile, catalog, k=k)
    retrieved_contexts = [retrieve_for_recommendation(rec) for rec in recommendations]
    explanations = explain_recommendations(profile, recommendations, retrieved_contexts, llm)

    return PipelineResult(
        nl_input="[Step 1: hand-crafted profile]",
        extracted_profile=profile,
        final_profile=profile,
        recommendations=recommendations,
        retrieved_contexts=retrieved_contexts,
        explanations=explanations,
        refinement_history=[],
        ambiguous_match=False,
        cache_stats=_cache_stats(llm),
    )
