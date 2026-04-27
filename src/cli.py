from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from src.agents.profile_extractor import ProfileExtractionError
from src.llm.client import CachedLLMClient, GeminiClient, LLMClient, StubLLMClient
from src.pipeline import (
    BuildInputs,
    EmptyBuildInputsError,
    ProfileBuildResult,
    RecommendationResult,
    build_profile,
    recommend,
)
from src.recommender import UserProfile


USAGE = 'Usage: python -m src.cli "<natural-language request>"'


def _build_client() -> LLMClient:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key:
        return CachedLLMClient(GeminiClient(api_key=api_key))
    print(
        "[stub mode - set GEMINI_API_KEY in .env for real explanations]",
        file=sys.stderr,
    )
    return StubLLMClient([])


def _format_profile_block(profile: UserProfile, header: str) -> list[str]:
    return [
        header,
        f"  favorite_genre:       {profile.favorite_genre}",
        f"  favorite_mood:        {profile.favorite_mood}",
        f"  target_energy:        {profile.target_energy}",
        f"  target_tempo_bpm:     {profile.target_tempo_bpm}",
        f"  target_valence:       {profile.target_valence}",
        f"  target_danceability:  {profile.target_danceability}",
        f"  target_acousticness:  {profile.target_acousticness}",
    ]


def _format_refinement_summary(build: ProfileBuildResult) -> str:
    parts = [f"iter {step.iter_index}: {step.verdict}" for step in build.refinement_history]
    summary = "Refinement summary: " + " -> ".join(parts) if parts else "Refinement summary: (none)"
    if build.ambiguous_match:
        summary += "  [!] ambiguous match"
    return summary


def _render(
    nl_input: str, build: ProfileBuildResult, rec: RecommendationResult
) -> str:
    out: list[str] = []
    out.append(f"Input: {nl_input}")
    out.append("")
    out.extend(_format_profile_block(build.extracted_profile, "Extracted profile:"))
    if build.candidate_profile != build.extracted_profile:
        out.append("")
        out.extend(
            _format_profile_block(
                build.candidate_profile, "Candidate profile (after critic refinement):"
            )
        )
    out.append("")
    if build.extractor_warnings:
        out.append("Extractor warnings:")
        for warning in build.extractor_warnings:
            out.append(f"  - {warning}")
        out.append("")
    out.append(_format_refinement_summary(build))
    out.append("")
    for index, (scored, expl) in enumerate(
        zip(rec.recommendations, rec.explanations), start=1
    ):
        out.append(
            f"{index}. {scored.song.title} - {scored.song.artist}  (score {scored.score:.2f})"
        )
        if expl.text is not None:
            out.append(f"   {expl.text}")
        else:
            out.append(f"   [mechanical reasons only - {expl.fallback_reason}]")
        for reason in scored.reasons:
            out.append(f"   - {reason}")
        out.append("")
    if rec.cache_stats is not None:
        out.append(
            f"cache: {rec.cache_stats['hits']} hits, {rec.cache_stats['misses']} misses"
        )
    return "\n".join(out)


def main() -> None:
    load_dotenv()

    if len(sys.argv) != 2 or sys.argv[1] in {"-h", "--help"}:
        print(USAGE, file=sys.stderr)
        sys.exit(2)

    nl_input = sys.argv[1]
    client = _build_client()

    try:
        build = build_profile(BuildInputs(description=nl_input), client)
        rec = recommend(build.candidate_profile, client)
    except EmptyBuildInputsError as exc:
        print(f"Empty input - {exc}", file=sys.stderr)
        sys.exit(2)
    except ProfileExtractionError as exc:
        print("Could not parse the request - try rephrasing.", file=sys.stderr)
        print(f"  reason: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        # Catches LLM errors (network, quota, empty-stub IndexError offline)
        # and any other pipeline failure. Online failures should be rare;
        # offline stub mode hits this path by design.
        print("Pipeline failed before producing recommendations.", file=sys.stderr)
        print(f"  {exc.__class__.__name__}: {exc}", file=sys.stderr)
        sys.exit(1)

    print(_render(nl_input, build, rec))


if __name__ == "__main__":
    main()
