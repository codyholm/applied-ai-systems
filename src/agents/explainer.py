from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from src.kb.retriever import RetrievedContext
from src.llm.client import LLMClient
from src.llm.parsing import strip_json_fences
from src.llm.prompts import EXPLAINER_PROMPT
from src.recommender import ScoredRecommendation, UserProfile


log = logging.getLogger(__name__)


@dataclass
class Explanation:
    song_id: int
    text: str | None
    cited_snippets: list[str]
    fallback_reason: str | None


def _format_profile(profile: UserProfile) -> str:
    return (
        f"  favorite_genre: {profile.favorite_genre}\n"
        f"  favorite_mood:  {profile.favorite_mood}\n"
        f"  target_energy:        {profile.target_energy}\n"
        f"  target_tempo_bpm:     {profile.target_tempo_bpm}\n"
        f"  target_valence:       {profile.target_valence}\n"
        f"  target_danceability:  {profile.target_danceability}\n"
        f"  target_acousticness:  {profile.target_acousticness}"
    )


def _format_candidates(
    recs: list[ScoredRecommendation], contexts: list[RetrievedContext]
) -> str:
    blocks: list[str] = []
    for rec, ctx in zip(recs, contexts):
        blocks.append(
            f"--- Candidate (song_id={rec.song.id}) ---\n"
            f"Title: {rec.song.title}\n"
            f"Artist: {rec.song.artist}\n"
            f"Genre: {rec.song.genre}\n"
            f"Mood: {rec.song.mood}\n"
            f"Score: {rec.score:.2f}\n"
            f"Mechanical reasons: {', '.join(rec.reasons)}\n\n"
            f"Genre snippet ({rec.song.genre}):\n{ctx.genre.body}\n\n"
            f"Mood snippet ({rec.song.mood}):\n{ctx.mood.body}\n\n"
            f"Song snippet:\n{ctx.song.body}"
        )
    return "\n\n".join(blocks)


def _all_fallback(
    recs: list[ScoredRecommendation], reason: str
) -> list[Explanation]:
    return [
        Explanation(song_id=rec.song.id, text=None, cited_snippets=[], fallback_reason=reason)
        for rec in recs
    ]


def explain_recommendations(
    profile: UserProfile,
    recs: list[ScoredRecommendation],
    contexts: list[RetrievedContext],
    llm: LLMClient,
) -> list[Explanation]:
    if len(recs) != len(contexts):
        raise ValueError(
            f"explain_recommendations: recs/contexts length mismatch ({len(recs)} vs {len(contexts)})"
        )

    prompt = EXPLAINER_PROMPT.format(
        profile_block=_format_profile(profile),
        candidates_block=_format_candidates(recs, contexts),
    )

    try:
        raw = llm.generate(prompt)
    except Exception as exc:
        log.warning("explainer LLM call raised %s; falling back", exc.__class__.__name__)
        return _all_fallback(recs, "llm_error")

    try:
        payload = json.loads(strip_json_fences(raw))
    except json.JSONDecodeError as exc:
        log.warning("explainer JSON parse failed: %s", exc)
        return _all_fallback(recs, "json_parse_error")

    entries = payload.get("explanations") if isinstance(payload, dict) else None
    if not isinstance(entries, list) or len(entries) != len(recs):
        log.warning("explainer payload shape invalid; got %r", payload)
        return _all_fallback(recs, "shape_invalid")

    expected_ids = {rec.song.id for rec in recs}
    by_id: dict[int, RetrievedContext] = {rec.song.id: ctx for rec, ctx in zip(recs, contexts)}

    results: list[Explanation] = []
    for entry in entries:
        if not isinstance(entry, dict):
            log.warning("explainer entry not dict: %r", entry)
            return _all_fallback(recs, "shape_invalid")
        sid = entry.get("song_id")
        text = entry.get("text")
        cited = entry.get("cited_snippets")
        if sid not in expected_ids:
            log.warning("explainer entry has unknown song_id %r", sid)
            return _all_fallback(recs, "song_id_mismatch")
        if not isinstance(text, str) or not text.strip():
            log.warning("explainer entry text invalid for song_id %r", sid)
            return _all_fallback(recs, "shape_invalid")
        if not isinstance(cited, list) or not cited:
            log.warning("explainer entry cited_snippets invalid for song_id %r", sid)
            return _all_fallback(recs, "shape_invalid")
        ctx = by_id[sid]
        haystack = (ctx.genre.body or "") + "\n" + (ctx.mood.body or "") + "\n" + (ctx.song.body or "")
        for snippet in cited:
            if not isinstance(snippet, str) or not snippet.strip():
                log.warning("explainer entry has empty snippet for song_id %r", sid)
                return _all_fallback(recs, "shape_invalid")
            if snippet not in haystack:
                log.warning("explainer fabricated citation for song_id %r: %r", sid, snippet[:60])
                return _all_fallback(recs, "fabricated_citation")
        results.append(
            Explanation(
                song_id=sid,
                text=text.strip(),
                cited_snippets=list(cited),
                fallback_reason=None,
            )
        )

    rec_order = [rec.song.id for rec in recs]
    by_sid = {expl.song_id: expl for expl in results}
    return [by_sid[sid] for sid in rec_order]
