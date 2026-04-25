from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from src.agents.profile_extractor import ProfileExtractionError
from src.llm.client import CachedLLMClient, GeminiClient, LLMClient, StubLLMClient
from src.pipeline import PipelineResult, run_pipeline


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


def _format_profile_block(result: PipelineResult) -> list[str]:
    p = result.extracted_profile
    return [
        "Extracted profile:",
        f"  favorite_genre:       {p.favorite_genre}",
        f"  favorite_mood:        {p.favorite_mood}",
        f"  target_energy:        {p.target_energy}",
        f"  target_tempo_bpm:     {p.target_tempo_bpm}",
        f"  target_valence:       {p.target_valence}",
        f"  target_danceability:  {p.target_danceability}",
        f"  target_acousticness:  {p.target_acousticness}",
    ]


def _format_refinement_summary(result: PipelineResult) -> str:
    parts = [f"iter {step.iter_index}: {step.verdict}" for step in result.refinement_history]
    summary = "Refinement summary: " + " -> ".join(parts) if parts else "Refinement summary: (none)"
    if result.ambiguous_match:
        summary += "  [!] ambiguous match"
    return summary


def _render(result: PipelineResult) -> str:
    out: list[str] = []
    out.append(f"Input: {result.nl_input}")
    out.append("")
    out.extend(_format_profile_block(result))
    out.append("")
    out.append(_format_refinement_summary(result))
    out.append("")
    for index, (rec, expl) in enumerate(
        zip(result.recommendations, result.explanations), start=1
    ):
        out.append(
            f"{index}. {rec.song.title} - {rec.song.artist}  (score {rec.score:.2f})"
        )
        if expl.text is not None:
            out.append(f"   {expl.text}")
        else:
            out.append(f"   [mechanical reasons only - {expl.fallback_reason}]")
        for reason in rec.reasons:
            out.append(f"   - {reason}")
        out.append("")
    if result.cache_stats is not None:
        out.append(
            f"cache: {result.cache_stats['hits']} hits, {result.cache_stats['misses']} misses"
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
        result = run_pipeline(nl_input, llm=client)
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

    print(_render(result))


if __name__ == "__main__":
    main()
