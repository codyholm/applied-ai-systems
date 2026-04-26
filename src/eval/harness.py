"""Reliability eval harness.

Runs all 10 EvalCases through the full pipeline, applies hand-coded
structural assertions, asks the LLM for a self-critique score, and
prints a scorecard plus writes a JSON artifact to eval_results/.

Quota guardrail (D21): without --confirm, the harness counts uncached
calls (best-effort upper bound) and refuses to proceed if any would be
issued. With --confirm, the harness pays whatever cost is needed.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from src.eval.assertions import assert_structural
from src.eval.cases import ALL_CASES, EvalCase
from src.llm.client import (
    DEFAULT_CACHE_DIR,
    CachedLLMClient,
    GeminiClient,
    LLMClient,
)
from src.llm.parsing import strip_json_fences
from src.llm.prompts import EVAL_SELF_CRITIQUE_PROMPT
from src.pipeline import PipelineResult, run_pipeline


log = logging.getLogger(__name__)

EVAL_RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "eval_results"


@dataclass
class EvalResult:
    case: EvalCase
    pipeline_result: PipelineResult
    structural_pass: bool
    structural_failures: list[str]
    self_critique_score: float
    self_critique_pass: bool
    self_critique_reason: str


def _format_top5_block(result: PipelineResult) -> str:
    lines = []
    for rec in result.recommendations:
        lines.append(
            f"  {rec.song.id} | {rec.song.title} | {rec.song.artist} | "
            f"{rec.song.genre} | {rec.song.mood} | {rec.score:.2f}"
        )
    return "\n".join(lines)


def _self_critique(
    case: EvalCase, result: PipelineResult, llm: LLMClient
) -> tuple[float, bool, str]:
    prompt = EVAL_SELF_CRITIQUE_PROMPT.format(
        nl_input=case.nl_input,
        top5_block=_format_top5_block(result),
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


def _hash_prompt_key(model: str, prompt: str) -> str:
    """Replicate CachedLLMClient._key for the default temperature/max_output.

    We cannot perfectly predict the run's cache pattern (critic and explainer
    prompts depend on already-generated content), so this is a best-effort
    audit of the predictable extractor and self-critique calls.
    """
    payload = json.dumps([model, None, prompt, 0.2, 1024], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _audit_uncached_calls(cases: list[EvalCase], model: str) -> int:
    """Best-effort audit: count predictable cache misses.

    Currently audits the 10 self-critique calls (predictable from case
    nl_input alone). Extractor calls are also predictable but use the
    PROFILE_EXTRACTOR_PROMPT.format() output — replicate that here.
    Critic and explainer calls are NOT predictable at audit time.
    """
    from src.agents.profile_extractor import _build_prompt as build_extractor_prompt

    misses = 0
    for case in cases:
        # Predictable: extractor prompt for this case.
        ext_prompt = build_extractor_prompt(case.nl_input)
        ext_key = _hash_prompt_key(model, ext_prompt)
        if not (DEFAULT_CACHE_DIR / f"{ext_key}.json").exists():
            misses += 1
        # Self-critique prompt: depends on pipeline result, which we don't
        # have at audit time. Conservative upper bound: assume a miss per case.
        # (When the pipeline runs and produces results, the actual prompt is
        # built and may or may not hit cache.)
        misses += 1
    return misses


def _serialise_pipeline_result(result: PipelineResult) -> dict:
    return {
        "nl_input": result.nl_input,
        "extracted_profile": dataclasses.asdict(result.extracted_profile),
        "final_profile": dataclasses.asdict(result.final_profile),
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
        "refinement_history": [
            {
                "iter_index": s.iter_index,
                "verdict": s.verdict,
                "top5_song_ids": list(s.top5_song_ids),
                "adjustments_applied": s.adjustments_applied,
                "reason": s.reason,
            }
            for s in result.refinement_history
        ],
        "ambiguous_match": result.ambiguous_match,
        "cache_stats": result.cache_stats,
        "extractor_warnings": list(result.extractor_warnings),
    }


def _serialise_eval_result(eval_result: EvalResult) -> dict:
    return {
        "case": dataclasses.asdict(eval_result.case),
        "pipeline_result": _serialise_pipeline_result(eval_result.pipeline_result),
        "structural_pass": eval_result.structural_pass,
        "structural_failures": list(eval_result.structural_failures),
        "self_critique_score": eval_result.self_critique_score,
        "self_critique_pass": eval_result.self_critique_pass,
        "self_critique_reason": eval_result.self_critique_reason,
    }


def _print_scorecard(results: list[EvalResult]) -> None:
    print("Eval scorecard")
    print("=" * 78)
    header = (
        f"{'name':<22} | {'cat':<8} | struct | crit_score | crit_pass | iters | ambig"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        struct = "PASS" if r.structural_pass else "FAIL"
        crit_pass = "yes" if r.self_critique_pass else "no"
        ambig = "yes" if r.pipeline_result.ambiguous_match else "no"
        iters = len(r.pipeline_result.refinement_history)
        print(
            f"{r.case.name:<22} | {r.case.category:<8} | "
            f"{struct:<6} | {r.self_critique_score:>9.2f}  | "
            f"{crit_pass:<9} | {iters:>5} | {ambig:>5}"
        )
    print()
    baseline = [r for r in results if r.case.category == "baseline"]
    baseline_pass = sum(1 for r in baseline if r.structural_pass)
    crit_total_pass = sum(1 for r in results if r.self_critique_pass)
    print(
        f"Summary: structural {baseline_pass}/{len(baseline)} baseline; "
        f"self-critique {crit_total_pass}/{len(results)} "
        f"({100 * crit_total_pass / len(results):.0f}%)"
    )


def _write_artifact(results: list[EvalResult]) -> Path:
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    path = EVAL_RESULTS_DIR / f"{timestamp}.json"
    payload = {
        "timestamp": timestamp,
        "results": [_serialise_eval_result(r) for r in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def run_harness(*, confirm: bool = False) -> list[EvalResult]:
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
    cases = ALL_CASES

    if not confirm:
        misses = _audit_uncached_calls(cases, client.MODEL)
        if misses > 0:
            print(
                f"would issue ~{misses} uncached calls (upper bound); "
                "pass --confirm to proceed"
            )
            return []

    results: list[EvalResult] = []
    for case in cases:
        log.info("running case %s", case.name)
        try:
            pipeline_result = run_pipeline(case.nl_input, llm=client)
        except Exception as exc:
            log.exception("pipeline failed for case %s", case.name)
            print(f"[case {case.name}] pipeline error: {exc}", file=sys.stderr)
            continue
        structural_pass, structural_failures = assert_structural(case, pipeline_result)
        score, crit_pass, crit_reason = _self_critique(case, pipeline_result, client)
        results.append(
            EvalResult(
                case=case,
                pipeline_result=pipeline_result,
                structural_pass=structural_pass,
                structural_failures=structural_failures,
                self_critique_score=score,
                self_critique_pass=crit_pass,
                self_critique_reason=crit_reason,
            )
        )

    _print_scorecard(results)
    artifact_path = _write_artifact(results)
    print(f"\nArtifact written to: {artifact_path}")
    return results


def main() -> None:
    confirm = "--confirm" in sys.argv[1:]
    if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        print("Usage: python -m src.eval.harness [--confirm]")
        sys.exit(0)
    run_harness(confirm=confirm)


if __name__ == "__main__":
    main()
