from __future__ import annotations

import functools
import json
import logging
from typing import TYPE_CHECKING

from src.llm.client import LLMClient
from src.llm.parsing import strip_json_fences
from src.llm.prompts import PROFILE_EXTRACTOR_PROMPT
from src.recommender import UserProfile, load_songs

if TYPE_CHECKING:
    from src.pipeline import BuildInputs


log = logging.getLogger(__name__)


class ProfileExtractionError(Exception):
    """Raised when the LLM fails to produce a parseable profile after one retry."""


@functools.lru_cache(maxsize=1)
def _allowed_genres() -> list[str]:
    return sorted({s.genre for s in load_songs()})


@functools.lru_cache(maxsize=1)
def _allowed_moods() -> list[str]:
    return sorted({s.mood for s in load_songs()})


_REQUIRED_KEYS = (
    "favorite_genre",
    "favorite_mood",
    "target_energy",
    "target_tempo_bpm",
    "target_valence",
    "target_danceability",
    "target_acousticness",
    "avoid_genres",
)

_RETRY_PREAMBLE = (
    "Your previous response failed JSON parsing. Return only valid JSON "
    "conforming to the schema below.\n\n"
)

_INPUT_LABELS: tuple[tuple[str, str], ...] = (
    ("activity",    "Activity (what they're doing or want this music for)"),
    ("feeling",     "Feeling (how they want this music to make them feel)"),
    ("movement",    "Movement (in the mood to move, sit still, or in between)"),
    ("instruments", "Instruments (acoustic, electronic, or a mix)"),
    ("genres",      "Genres (specifically wanted or to avoid)"),
    ("description", "Description (free-form mood description)"),
)


def _format_inputs_bundle(inputs: BuildInputs) -> str:
    lines: list[str] = []
    for attr, label in _INPUT_LABELS:
        value = getattr(inputs, attr, None)
        if value and value.strip():
            lines.append(f"  {label}: {value.strip()}")
    if not lines:
        return "  (no inputs provided)"
    return "\n".join(lines)


def _format_starting_from(profile: UserProfile | None) -> str:
    if profile is None:
        return ""
    avoid_display = ", ".join(profile.avoid_genres) if profile.avoid_genres else "(none)"
    return (
        "An existing profile is provided as a seed; treat it as the starting\n"
        "point and only modify fields the new inputs clearly change.\n"
        "Existing profile:\n"
        f"  favorite_genre:       {profile.favorite_genre}\n"
        f"  favorite_mood:        {profile.favorite_mood}\n"
        f"  target_energy:        {profile.target_energy}\n"
        f"  target_tempo_bpm:     {profile.target_tempo_bpm}\n"
        f"  target_valence:       {profile.target_valence}\n"
        f"  target_danceability:  {profile.target_danceability}\n"
        f"  target_acousticness:  {profile.target_acousticness}\n"
        f"  avoid_genres:         {avoid_display}\n\n"
    )


def _build_prompt(
    inputs: BuildInputs, starting_from: UserProfile | None = None
) -> str:
    return PROFILE_EXTRACTOR_PROMPT.format(
        inputs_block=_format_inputs_bundle(inputs),
        starting_from_block=_format_starting_from(starting_from),
        allowed_genres=", ".join(_allowed_genres()),
        allowed_moods=", ".join(_allowed_moods()),
    )


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _resolve_allowed(
    value: str, allowed: list[str], field_name: str, warnings: list[str]
) -> str:
    if not isinstance(value, str):
        msg = f"{field_name}: non-string value {value!r}; mapped to fallback {allowed[0]!r}"
        log.warning("profile_extractor: %s", msg)
        warnings.append(msg)
        return allowed[0]
    for candidate in allowed:
        if candidate.lower() == value.lower():
            return candidate
    msg = (
        f"{field_name}: {value!r} not in allowed list; mapped to fallback {allowed[0]!r}"
    )
    log.warning("profile_extractor: %s", msg)
    warnings.append(msg)
    return allowed[0]


def _resolve_allowed_list(
    value, allowed: list[str], field_name: str, warnings: list[str]
) -> list[str]:
    """Validate a list-typed categorical field.

    Drops non-list inputs and entries not present in `allowed`, with a
    warning per drop. Survivors are lowercased and deduplicated while
    preserving first-seen order.
    """
    if not isinstance(value, list):
        msg = f"{field_name}: non-list value {value!r}; treating as empty list"
        log.warning("profile_extractor: %s", msg)
        warnings.append(msg)
        return []
    allowed_lower = {a.lower() for a in allowed}
    seen: set[str] = set()
    result: list[str] = []
    for entry in value:
        if not isinstance(entry, str):
            msg = f"{field_name}: dropping non-string entry {entry!r}"
            log.warning("profile_extractor: %s", msg)
            warnings.append(msg)
            continue
        normalized = entry.strip().lower()
        if not normalized:
            continue
        if normalized not in allowed_lower:
            msg = f"{field_name}: dropping unknown entry {entry!r} (not in allowed list)"
            log.warning("profile_extractor: %s", msg)
            warnings.append(msg)
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _parse_payload(payload: dict) -> tuple[UserProfile, list[str]]:
    missing = [k for k in _REQUIRED_KEYS if k not in payload]
    if missing:
        raise ProfileExtractionError(f"missing keys in extracted profile: {missing}")
    warnings: list[str] = []
    favorite_genre = _resolve_allowed(
        payload["favorite_genre"], _allowed_genres(), "favorite_genre", warnings
    )
    avoid_genres = _resolve_allowed_list(
        payload["avoid_genres"], _allowed_genres(), "avoid_genres", warnings
    )
    # Contradiction guard: if the LLM put favorite_genre into avoid_genres,
    # drop it from the avoid list (favorite wins; the prompt forbids this
    # but defensive coding is cheap here).
    if favorite_genre.lower() in avoid_genres:
        msg = (
            f"avoid_genres: dropping {favorite_genre!r} because it is the "
            f"favorite_genre"
        )
        log.warning("profile_extractor: %s", msg)
        warnings.append(msg)
        avoid_genres = [g for g in avoid_genres if g != favorite_genre.lower()]
    resolved = {
        "favorite_genre": favorite_genre,
        "favorite_mood": _resolve_allowed(
            payload["favorite_mood"], _allowed_moods(), "favorite_mood", warnings
        ),
        "target_energy": _clamp(float(payload["target_energy"]), 0.0, 1.0),
        "target_tempo_bpm": _clamp(float(payload["target_tempo_bpm"]), 40.0, 220.0),
        "target_valence": _clamp(float(payload["target_valence"]), 0.0, 1.0),
        "target_danceability": _clamp(float(payload["target_danceability"]), 0.0, 1.0),
        "target_acousticness": _clamp(float(payload["target_acousticness"]), 0.0, 1.0),
        "avoid_genres": avoid_genres,
    }
    return UserProfile.from_dict(resolved), warnings


def extract_profile(
    inputs: BuildInputs,
    llm: LLMClient,
    *,
    starting_from: UserProfile | None = None,
) -> tuple[UserProfile, list[str]]:
    """Extract a UserProfile from a labeled BuildInputs bundle.

    Builds a labeled prompt that includes only the non-blank fields from
    `inputs`, plus an optional starting-profile seed when `starting_from`
    is provided (re-describe-to-update flow). Returns the profile plus a
    list of warning strings noting any fallback substitutions made
    (e.g. unknown genre mapped to default).
    """
    prompt = _build_prompt(inputs, starting_from=starting_from)
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
