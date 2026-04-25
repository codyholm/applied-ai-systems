from __future__ import annotations

import os
import sys
from typing import Any

from dotenv import load_dotenv

from src.llm.client import CachedLLMClient, GeminiClient, LLMClient, StubLLMClient
from src.pipeline import PipelineResult, run_pipeline
from src.recommender import UserProfile


BUILTIN_PROFILES: dict[str, dict[str, Any]] = {
    "high_energy_pop": {
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "target_energy": 0.85,
        "target_tempo_bpm": 124.0,
        "target_acousticness": 0.15,
        "target_valence": 0.85,
        "target_danceability": 0.85,
    },
    "chill_lofi": {
        "favorite_genre": "lofi",
        "favorite_mood": "chill",
        "target_energy": 0.35,
        "target_tempo_bpm": 78.0,
        "target_acousticness": 0.80,
        "target_valence": 0.55,
        "target_danceability": 0.50,
    },
    "deep_intense_rock": {
        "favorite_genre": "rock",
        "favorite_mood": "intense",
        "target_energy": 0.90,
        "target_tempo_bpm": 145.0,
        "target_acousticness": 0.10,
        "target_valence": 0.45,
        "target_danceability": 0.65,
    },
    "chill_rock": {
        "favorite_genre": "rock",
        "favorite_mood": "chill",
        "target_energy": 0.25,
        "target_tempo_bpm": 75.0,
        "target_acousticness": 0.85,
        "target_valence": 0.50,
        "target_danceability": 0.30,
    },
    "boundary_maximalist": {
        "favorite_genre": "electronic",
        "favorite_mood": "intense",
        "target_energy": 1.0,
        "target_tempo_bpm": 200.0,
        "target_acousticness": 0.0,
        "target_valence": 1.0,
        "target_danceability": 1.0,
    },
}


USAGE = (
    "Usage: python -m src.cli <profile_name>\n"
    f"  profile_name in {sorted(BUILTIN_PROFILES)}"
)


def _build_client() -> LLMClient:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key:
        return CachedLLMClient(GeminiClient(api_key=api_key))
    print(
        "[stub mode - set GEMINI_API_KEY in .env for real explanations]",
        file=sys.stderr,
    )
    return StubLLMClient([])


def _render(profile_name: str, result: PipelineResult) -> str:
    out: list[str] = []
    out.append(f"Profile: {profile_name}")
    out.append("-" * (len(profile_name) + 9))
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

    profile_name = sys.argv[1]
    if profile_name not in BUILTIN_PROFILES:
        print(f"Unknown profile: {profile_name!r}", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        sys.exit(2)

    profile = UserProfile.from_dict(BUILTIN_PROFILES[profile_name])
    client = _build_client()
    result = run_pipeline(profile, llm=client)
    print(_render(profile_name, result))


if __name__ == "__main__":
    main()
