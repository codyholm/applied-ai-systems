"""Reliability eval harness for the two-pipeline architecture.

Two evaluations under one --confirm-gated invocation:

- run_build_eval(client) — runs each BUILD_CASE through build_profile()
  and asserts the candidate profile lands in the target preset's
  neighborhood (assert_build_neighborhood). Surfaces extractor warnings
  and the ambiguous flag per case. No LLM self-critique here —
  faithfulness is hand-coded.
- run_recommend_eval(client) — runs recommend() against each preset
  directly, applies assert_recommend_structural, then asks the LLM
  self-critique whether the top-5 reasonably reflects the profile.

Quota guardrail (D21): without --confirm, the harness counts predictable
uncached calls (best-effort upper bound) and refuses to proceed if any
would be issued. With --confirm, it pays whatever cost is needed.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from src.eval.assertions import (
    assert_build_neighborhood,
    assert_recommend_structural,
)
from src.eval.cases import BUILD_CASES, RECOMMEND_CASES, BuildCase
from src.llm.client import (
    DEFAULT_CACHE_DIR,
    CachedLLMClient,
    GeminiClient,
    LLMClient,
)
from src.llm.parsing import strip_json_fences
from src.llm.prompts import EVAL_SELF_CRITIQUE_PROMPT
from src.pipeline import (
    BuildInputs,
    ProfileBuildResult,
    RecommendationResult,
    build_profile,
    recommend,
)
from src.profiles import PRESET_PROFILES, load_preset
from src.recommender import UserProfile


log = logging.getLogger(__name__)

EVAL_RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "eval_results"


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BuildEvalResult:
    case: BuildCase
    result: ProfileBuildResult
    passed: bool
    failures: list[str]


@dataclass
class RecommendEvalResult:
    preset_name: str
    result: RecommendationResult
    structural_pass: bool
    structural_failures: list[str]
    self_critique_score: float
    self_critique_pass: bool
    self_critique_reason: str


# ---------------------------------------------------------------------------
# Self-critique (recommend-eval only)
# ---------------------------------------------------------------------------


def _format_profile_block(profile: UserProfile) -> str:
    return (
        f"  favorite_genre:       {profile.favorite_genre}\n"
        f"  favorite_mood:        {profile.favorite_mood}\n"
        f"  target_energy:        {profile.target_energy}\n"
        f"  target_tempo_bpm:     {profile.target_tempo_bpm}\n"
        f"  target_valence:       {profile.target_valence}\n"
        f"  target_danceability:  {profile.target_danceability}\n"
        f"  target_acousticness:  {profile.target_acousticness}"
    )


def _format_top5_block(rec: RecommendationResult) -> str:
    lines = []
    for r in rec.recommendations:
        lines.append(
            f"  {r.song.id} | {r.song.title} | {r.song.artist} | "
            f"{r.song.genre} | {r.song.mood} | {r.score:.2f}"
        )
    return "\n".join(lines)


def _self_critique(
    profile: UserProfile, rec: RecommendationResult, llm: LLMClient
) -> tuple[float, bool, str]:
    prompt = EVAL_SELF_CRITIQUE_PROMPT.format(
        profile_block=_format_profile_block(profile),
        top5_block=_format_top5_block(rec),
    )
    try:
        raw = llm.generate(prompt)
    except Exception as exc:
        log.warning("self-critique LLM call raised %s; failing open", exc.__class__.__name__)
        return 0.5, False, f"llm_error: {exc.__class__.__name__}"

    try:
        payload = json.loads(strip_json_fences(raw))
    except json.JSONDecodeError as exc:
        log.warning("self-critique JSON parse failed: %s", exc)
        return 0.5, False, "self_critique_parse_failure"

    if not isinstance(payload, dict):
        return 0.5, False, "self_critique_parse_failure"

    score_raw = payload.get("score")
    pass_raw = payload.get("pass")
    reason = payload.get("reason", "")

    try:
        score = max(0.0, min(1.0, float(score_raw)))
    except (TypeError, ValueError):
        return 0.5, False, "self_critique_parse_failure"

    pass_ = bool(pass_raw) if isinstance(pass_raw, bool) else (score >= 0.6)
    if not isinstance(reason, str):
        reason = str(reason)

    return score, pass_, reason


# ---------------------------------------------------------------------------
# Quota audit
# ---------------------------------------------------------------------------


def _hash_prompt_key(model: str, prompt: str) -> str:
    payload = json.dumps([model, None, prompt, 0.2, 1024], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _audit_uncached_calls(
    build_cases: list[BuildCase], recommend_cases: list[str], model: str
) -> int:
    """Best-effort upper bound on uncached calls.

    For each BuildCase: check the predictable extractor prompt. The
    critic prompt depends on the extractor's output, so a +1 conservative
    upper bound per build case is added.
    For each recommend case: explainer + self-critique prompts depend on
    the catalog ranking, so a +2 conservative upper bound per case.
    """
    from src.agents.profile_extractor import _build_prompt as build_extractor_prompt

    misses = 0
    for case in build_cases:
        ext_prompt = build_extractor_prompt(case.inputs)
        ext_key = _hash_prompt_key(model, ext_prompt)
        if not (DEFAULT_CACHE_DIR / f"{ext_key}.json").exists():
            misses += 1
        misses += 1  # critic upper bound
    for _preset in recommend_cases:
        misses += 2  # explainer + self-critique upper bound
    return misses


# ---------------------------------------------------------------------------
# build-eval
# ---------------------------------------------------------------------------


def run_build_eval(client: LLMClient) -> list[BuildEvalResult]:
    """Run all BUILD_CASES through build_profile and assert neighborhood."""
    results: list[BuildEvalResult] = []
    for case in BUILD_CASES:
        log.info("running build case %s", case.name)
        try:
            build_result = build_profile(case.inputs, client)
        except Exception as exc:
            log.exception("build pipeline failed for case %s", case.name)
            print(f"[build {case.name}] pipeline error: {exc}", file=sys.stderr)
            continue
        passed, failures = assert_build_neighborhood(case, build_result)
        results.append(
            BuildEvalResult(
                case=case, result=build_result, passed=passed, failures=failures
            )
        )
    return results


# ---------------------------------------------------------------------------
# recommend-eval
# ---------------------------------------------------------------------------


def run_recommend_eval(client: LLMClient) -> list[RecommendEvalResult]:
    """Run recommend() against each preset and apply structural + self-critique."""
    results: list[RecommendEvalResult] = []
    for preset_name in RECOMMEND_CASES:
        log.info("running recommend case %s", preset_name)
        try:
            profile = load_preset(preset_name)
            rec_result = recommend(profile, client)
        except Exception as exc:
            log.exception("recommend pipeline failed for preset %s", preset_name)
            print(f"[recommend {preset_name}] pipeline error: {exc}", file=sys.stderr)
            continue
        struct_pass, struct_failures = assert_recommend_structural(preset_name, rec_result)
        score, crit_pass, crit_reason = _self_critique(profile, rec_result, client)
        results.append(
            RecommendEvalResult(
                preset_name=preset_name,
                result=rec_result,
                structural_pass=struct_pass,
                structural_failures=struct_failures,
                self_critique_score=score,
                self_critique_pass=crit_pass,
                self_critique_reason=crit_reason,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Scorecards
# ---------------------------------------------------------------------------


def _print_build_scorecard(results: list[BuildEvalResult]) -> None:
    print("BUILD scorecard")
    print("=" * 78)
    header = (
        f"{'name':<32} | {'cat':<8} | pass | iters | ambig | warns"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        ambig = "yes" if r.result.ambiguous_match else "no"
        iters = len(r.result.refinement_history)
        warns = len(r.result.extractor_warnings)
        print(
            f"{r.case.name:<32} | {r.case.category:<8} | "
            f"{status:<4} | {iters:>5} | {ambig:>5} | {warns:>5}"
        )
    print()
    baseline = [r for r in results if r.case.category == "baseline"]
    baseline_pass = sum(1 for r in baseline if r.passed)
    total_pass = sum(1 for r in results if r.passed)
    print(
        f"Build summary: baseline {baseline_pass}/{len(baseline)}; "
        f"overall {total_pass}/{len(results)}"
    )
    # Per-case failure detail
    failed = [r for r in results if not r.passed]
    if failed:
        print()
        print("Failures (build):")
        for r in failed:
            print(f"  {r.case.name}:")
            for failure in r.failures:
                print(f"    - {failure}")


def _print_recommend_scorecard(results: list[RecommendEvalResult]) -> None:
    print("RECOMMEND scorecard")
    print("=" * 78)
    header = (
        f"{'preset':<24} | struct | crit_score | crit_pass"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        struct = "PASS" if r.structural_pass else "FAIL"
        crit_pass = "yes" if r.self_critique_pass else "no"
        print(
            f"{r.preset_name:<24} | {struct:<6} | "
            f"{r.self_critique_score:>9.2f}  | {crit_pass}"
        )
    print()
    struct_pass = sum(1 for r in results if r.structural_pass)
    crit_total_pass = sum(1 for r in results if r.self_critique_pass)
    print(
        f"Recommend summary: structural {struct_pass}/{len(results)}; "
        f"self-critique {crit_total_pass}/{len(results)} "
        f"({100 * crit_total_pass / max(len(results), 1):.0f}%)"
    )
    failed = [r for r in results if not r.structural_pass]
    if failed:
        print()
        print("Failures (recommend structural):")
        for r in failed:
            print(f"  {r.preset_name}:")
            for failure in r.structural_failures:
                print(f"    - {failure}")


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _serialise_build_result(result: ProfileBuildResult) -> dict:
    return {
        "inputs": dataclasses.asdict(result.inputs),
        "extracted_profile": dataclasses.asdict(result.extracted_profile),
        "candidate_profile": dataclasses.asdict(result.candidate_profile),
        "refinement_history": [
            {
                "iter_index": s.iter_index,
                "verdict": s.verdict,
                "candidate_after_iter": s.candidate_after_iter,
                "adjustments_applied": s.adjustments_applied,
                "reason": s.reason,
            }
            for s in result.refinement_history
        ],
        "ambiguous_match": result.ambiguous_match,
        "extractor_warnings": list(result.extractor_warnings),
        "cache_stats": result.cache_stats,
    }


def _serialise_rec_result(result: RecommendationResult) -> dict:
    return {
        "profile": dataclasses.asdict(result.profile),
        "recommendations": [
            {
                "song_id": r.song.id,
                "title": r.song.title,
                "artist": r.song.artist,
                "genre": r.song.genre,
                "mood": r.song.mood,
                "score": r.score,
                "reasons": list(r.reasons),
            }
            for r in result.recommendations
        ],
        "explanations": [
            {
                "song_id": e.song_id,
                "text": e.text,
                "cited_snippets": list(e.cited_snippets),
                "fallback_reason": e.fallback_reason,
            }
            for e in result.explanations
        ],
        "cache_stats": result.cache_stats,
    }


def _serialise_build_eval(b: BuildEvalResult) -> dict:
    return {
        "case": {
            "name": b.case.name,
            "category": b.case.category,
            "expected_profile": dataclasses.asdict(b.case.expected_profile),
            "tempo_tolerance_bpm": b.case.tempo_tolerance_bpm,
            "numeric_tolerance": b.case.numeric_tolerance,
            "inputs": dataclasses.asdict(b.case.inputs),
        },
        "result": _serialise_build_result(b.result),
        "passed": b.passed,
        "failures": list(b.failures),
    }


def _serialise_recommend_eval(r: RecommendEvalResult) -> dict:
    return {
        "preset_name": r.preset_name,
        "result": _serialise_rec_result(r.result),
        "structural_pass": r.structural_pass,
        "structural_failures": list(r.structural_failures),
        "self_critique_score": r.self_critique_score,
        "self_critique_pass": r.self_critique_pass,
        "self_critique_reason": r.self_critique_reason,
    }


def _write_artifact(
    build_results: list[BuildEvalResult],
    recommend_results: list[RecommendEvalResult],
) -> Path:
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    path = EVAL_RESULTS_DIR / f"{timestamp}.json"
    payload = {
        "timestamp": timestamp,
        "build_results": [_serialise_build_eval(b) for b in build_results],
        "recommend_results": [_serialise_recommend_eval(r) for r in recommend_results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_harness(
    *, confirm: bool = False
) -> tuple[list[BuildEvalResult], list[RecommendEvalResult]]:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        print(
            "Set GEMINI_API_KEY in .env to run the eval harness.",
            file=sys.stderr,
        )
        sys.exit(2)

    inner = GeminiClient(api_key=api_key)
    client = CachedLLMClient(inner)

    if not confirm:
        misses = _audit_uncached_calls(BUILD_CASES, RECOMMEND_CASES, client.MODEL)
        if misses > 0:
            print(
                f"would issue ~{misses} uncached calls (upper bound); "
                "pass --confirm to proceed"
            )
            return [], []

    build_results = run_build_eval(client)
    recommend_results = run_recommend_eval(client)

    _print_build_scorecard(build_results)
    print()
    _print_recommend_scorecard(recommend_results)

    artifact_path = _write_artifact(build_results, recommend_results)
    print(f"\nArtifact written to: {artifact_path}")
    return build_results, recommend_results


def main() -> None:
    confirm = "--confirm" in sys.argv[1:]
    if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        print("Usage: python -m src.eval.harness [--confirm]")
        sys.exit(0)
    run_harness(confirm=confirm)


if __name__ == "__main__":
    main()
