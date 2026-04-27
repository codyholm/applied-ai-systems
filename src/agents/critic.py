from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from src.llm.client import LLMClient
from src.llm.parsing import strip_json_fences
from src.llm.prompts import CRITIC_PROMPT
from src.recommender import UserProfile


log = logging.getLogger(__name__)


_NUMERIC_KEYS = {
    "target_energy",
    "target_tempo_bpm",
    "target_valence",
    "target_danceability",
    "target_acousticness",
}
_STRING_KEYS = {"favorite_genre", "favorite_mood"}
_ALLOWED_KEYS = _NUMERIC_KEYS | _STRING_KEYS

_NUMERIC_RANGES = {
    "target_energy": (0.0, 1.0),
    "target_tempo_bpm": (40.0, 220.0),
    "target_valence": (0.0, 1.0),
    "target_danceability": (0.0, 1.0),
    "target_acousticness": (0.0, 1.0),
}


@dataclass
class CriticVerdict:
    verdict: str
    adjustments: dict[str, float | str] | None
    reason: str


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


def _clamp_adjustments(raw: dict) -> dict[str, float | str]:
    cleaned: dict[str, float | str] = {}
    for key, value in raw.items():
        if key not in _ALLOWED_KEYS:
            log.warning("critic: dropping disallowed adjustment key %r", key)
            continue
        if key in _NUMERIC_KEYS:
            try:
                lo, hi = _NUMERIC_RANGES[key]
                cleaned[key] = max(lo, min(hi, float(value)))
            except (TypeError, ValueError):
                log.warning("critic: dropping non-numeric value for %r: %r", key, value)
                continue
        else:  # string keys
            if isinstance(value, str) and value:
                cleaned[key] = value
            else:
                log.warning("critic: dropping non-string value for %r: %r", key, value)
    return cleaned


def critique_extraction(
    nl_description: str,
    candidate_profile: UserProfile,
    llm: LLMClient,
) -> CriticVerdict:
    """Check whether a candidate UserProfile faithfully encodes the listener's description.

    Inputs:
      - nl_description: the listener's free-form description of their taste.
      - candidate_profile: the extractor's current proposed UserProfile.
      - llm: an LLMClient.

    Returns a CriticVerdict. verdict='ok' means the profile faithfully
    encodes the description (no changes needed). verdict='refine' means
    one or more fields plainly diverge from the description and the
    `adjustments` dict carries the corrected absolute values for those
    fields.

    Failure modes (LLM error, malformed JSON, invalid verdict, refine
    without adjustments) all degrade to 'ok' so the build pipeline never
    blocks on a misbehaving critic.
    """
    prompt = CRITIC_PROMPT.format(
        nl_description=nl_description,
        profile_block=_format_profile(candidate_profile),
    )

    try:
        raw = llm.generate(prompt)
    except Exception as exc:
        log.warning("critic: LLM call raised %s; failing open to ok", exc.__class__.__name__)
        return CriticVerdict(verdict="ok", adjustments=None, reason="llm_error")

    try:
        payload = json.loads(strip_json_fences(raw))
    except json.JSONDecodeError as exc:
        log.warning("critic: JSON parse failed (%s); failing open to ok", exc)
        return CriticVerdict(verdict="ok", adjustments=None, reason="parse_failure")

    if not isinstance(payload, dict):
        log.warning("critic: payload not dict: %r; failing open to ok", payload)
        return CriticVerdict(verdict="ok", adjustments=None, reason="parse_failure")

    verdict = payload.get("verdict")
    if verdict not in {"ok", "refine"}:
        log.warning("critic: invalid verdict %r; failing open to ok", verdict)
        return CriticVerdict(verdict="ok", adjustments=None, reason="parse_failure")

    reason = payload.get("reason", "")
    if not isinstance(reason, str):
        reason = ""

    if verdict == "ok":
        return CriticVerdict(verdict="ok", adjustments=None, reason=reason or "ok")

    raw_adj = payload.get("adjustments")
    if not isinstance(raw_adj, dict) or not raw_adj:
        log.warning("critic: refine without valid adjustments; failing open to ok")
        return CriticVerdict(verdict="ok", adjustments=None, reason="parse_failure")

    cleaned = _clamp_adjustments(raw_adj)
    if not cleaned:
        log.warning("critic: refine had no valid adjustments after cleaning; failing open")
        return CriticVerdict(verdict="ok", adjustments=None, reason="parse_failure")

    return CriticVerdict(verdict="refine", adjustments=cleaned, reason=reason or "refine")
