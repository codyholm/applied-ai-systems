from __future__ import annotations

import json
import logging

from src.llm.client import LLMClient
from src.llm.parsing import strip_json_fences
from src.llm.prompts import PROFILE_EXTRACTOR_PROMPT
from src.recommender import UserProfile, load_songs


log = logging.getLogger(__name__)


class ProfileExtractionError(Exception):
    """Raised when the LLM fails to produce a parseable profile after one retry."""


_ALLOWED_GENRES: list[str] = sorted({s.genre for s in load_songs()})
_ALLOWED_MOODS: list[str] = sorted({s.mood for s in load_songs()})


_REQUIRED_KEYS = (
    "favorite_genre",
    "favorite_mood",
    "target_energy",
    "target_tempo_bpm",
    "target_valence",
    "target_danceability",
    "target_acousticness",
)

_RETRY_PREAMBLE = (
    "Your previous response failed JSON parsing. Return only valid JSON "
    "conforming to the schema below.\n\n"
)


def _build_prompt(nl_input: str) -> str:
    return PROFILE_EXTRACTOR_PROMPT.format(
        nl_input=nl_input,
        allowed_genres=", ".join(_ALLOWED_GENRES),
        allowed_moods=", ".join(_ALLOWED_MOODS),
    )


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _resolve_allowed(value: str, allowed: list[str]) -> str:
    if not isinstance(value, str):
        log.warning("profile_extractor: non-string value %r; using fallback", value)
        return allowed[0]
    for candidate in allowed:
        if candidate.lower() == value.lower():
            return candidate
    log.warning(
        "profile_extractor: value %r not in allowed list %r; using fallback %r",
        value,
        allowed,
        allowed[0],
    )
    return allowed[0]


def _parse_payload(payload: dict) -> UserProfile:
    missing = [k for k in _REQUIRED_KEYS if k not in payload]
    if missing:
        raise ProfileExtractionError(f"missing keys in extracted profile: {missing}")
    resolved = {
        "favorite_genre": _resolve_allowed(payload["favorite_genre"], _ALLOWED_GENRES),
        "favorite_mood": _resolve_allowed(payload["favorite_mood"], _ALLOWED_MOODS),
        "target_energy": _clamp(float(payload["target_energy"]), 0.0, 1.0),
        "target_tempo_bpm": _clamp(float(payload["target_tempo_bpm"]), 40.0, 220.0),
        "target_valence": _clamp(float(payload["target_valence"]), 0.0, 1.0),
        "target_danceability": _clamp(float(payload["target_danceability"]), 0.0, 1.0),
        "target_acousticness": _clamp(float(payload["target_acousticness"]), 0.0, 1.0),
    }
    return UserProfile.from_dict(resolved)


def extract_profile(nl_input: str, llm: LLMClient) -> UserProfile:
    prompt = _build_prompt(nl_input)
    raw = llm.generate(prompt)
    try:
        payload = json.loads(strip_json_fences(raw))
    except json.JSONDecodeError as exc:
        log.warning("profile_extractor: first parse failed (%s); retrying once", exc)
        raw = llm.generate(_RETRY_PREAMBLE + prompt)
        try:
            payload = json.loads(strip_json_fences(raw))
        except json.JSONDecodeError as exc2:
            raise ProfileExtractionError(
                f"profile JSON failed to parse after retry: {exc2}"
            ) from exc2

    if not isinstance(payload, dict):
        raise ProfileExtractionError(f"profile payload is not a JSON object: {payload!r}")

    return _parse_payload(payload)
