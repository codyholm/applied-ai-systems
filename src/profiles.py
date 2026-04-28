"""Profile store for the music recommender.

Profiles are 7-dimension UserProfile dataclasses that the recommender uses
to rank the catalog. They come from two sources:

- PRESET_PROFILES: five hand-authored profiles from Module 3, kept in code
  as immutable demo/test fixtures. Accessed via load_preset() or by direct
  dict lookup. They are NEVER copied into the on-disk user profile store.
- User profiles saved to applied-ai-systems/profiles/<slug>.json. These
  persist across sessions and can be created, listed, loaded, and deleted
  via this module's public API.

The on-disk JSON format wraps the UserProfile fields with a small envelope
recording the human-readable name and an ISO-8601 created-at timestamp so
the store can list profiles in creation order without reading the file
bodies.

Slug normalization (used both for preset keys and filename derivation):
lowercase, whitespace -> hyphen, drop everything that is not alphanumeric
or hyphen.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

from src.recommender import UserProfile


_USER_PROFILE_FIELDS: frozenset[str] = frozenset(
    f.name for f in dataclasses.fields(UserProfile)
)


PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"


class ProfileNotFoundError(Exception):
    """Raised when load/delete is attempted on a profile that does not exist."""


class ProfileExistsError(Exception):
    """Raised when save with overwrite=False finds an existing profile."""


PRESET_PROFILES: dict[str, UserProfile] = {
    "high_energy_pop": UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.85,
        target_tempo_bpm=124.0,
        target_acousticness=0.15,
        target_valence=0.85,
        target_danceability=0.85,
        avoid_genres=[],
    ),
    "chill_lofi": UserProfile(
        favorite_genre="lofi",
        favorite_mood="chill",
        target_energy=0.35,
        target_tempo_bpm=78.0,
        target_acousticness=0.80,
        target_valence=0.55,
        target_danceability=0.50,
        avoid_genres=[],
    ),
    "deep_intense_rock": UserProfile(
        favorite_genre="rock",
        favorite_mood="intense",
        target_energy=0.90,
        target_tempo_bpm=145.0,
        target_acousticness=0.10,
        target_valence=0.45,
        target_danceability=0.65,
        avoid_genres=[],
    ),
    "chill_rock": UserProfile(
        favorite_genre="rock",
        favorite_mood="moody",
        target_energy=0.55,
        target_tempo_bpm=95.0,
        target_acousticness=0.35,
        target_valence=0.40,
        target_danceability=0.50,
        avoid_genres=[],
    ),
    "boundary_maximalist": UserProfile(
        favorite_genre="electronic",
        favorite_mood="intense",
        target_energy=1.0,
        target_tempo_bpm=200.0,
        target_acousticness=0.0,
        target_valence=1.0,
        target_danceability=1.0,
        avoid_genres=[],
    ),
}


PRESET_DISPLAY_NAMES: dict[str, str] = {
    "high_energy_pop": "High-Energy Pop",
    "chill_lofi": "Chill Lofi",
    "deep_intense_rock": "Deep Intense Rock",
    "chill_rock": "Chill Rock",
    "boundary_maximalist": "Boundary Maximalist",
}


def slugify(name: str) -> str:
    """Normalize a human-readable profile name to a filesystem-safe slug.

    Lowercases, collapses whitespace runs to single hyphens, drops any
    character that is not alphanumeric or hyphen. Empty result raises
    ValueError so callers do not accidentally write to profiles/.json.
    """
    lowered = name.strip().lower()
    hyphenated = re.sub(r"\s+", "-", lowered)
    cleaned = re.sub(r"[^a-z0-9-]", "", hyphenated)
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    if not cleaned:
        raise ValueError(f"Profile name {name!r} produces an empty slug.")
    return cleaned


def _profile_path(name: str) -> Path:
    return PROFILES_DIR / f"{slugify(name)}.json"


def save_profile(
    name: str, profile: UserProfile, *, overwrite: bool = True
) -> Path:
    """Save a UserProfile to disk under PROFILES_DIR.

    The on-disk file is named slugify(name).json and contains the human
    name, an ISO-8601 UTC created-at timestamp, and the profile's seven
    dataclass fields. With overwrite=False, raises ProfileExistsError if
    a profile with the same slug already exists; otherwise clobbers.
    """
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    path = _profile_path(name)
    if path.exists() and not overwrite:
        raise ProfileExistsError(
            f"Profile {slugify(name)!r} already exists at {path}"
        )
    payload = {
        "name": name,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "profile": dataclasses.asdict(profile),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_profile(name: str) -> UserProfile:
    """Load a saved UserProfile by name (or by slug)."""
    path = _profile_path(name)
    if not path.exists():
        raise ProfileNotFoundError(f"No saved profile named {name!r} (looked at {path})")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return UserProfile.from_dict(payload["profile"])


def delete_profile(name: str) -> None:
    """Remove a saved profile by name (or slug)."""
    path = _profile_path(name)
    if not path.exists():
        raise ProfileNotFoundError(f"No saved profile named {name!r} (looked at {path})")
    path.unlink()


def list_profiles() -> list[tuple[str, dt.datetime]]:
    """Enumerate saved profiles, newest first.

    Returns a list of (name, created_at) tuples. The name is the original
    human-readable string from the on-disk envelope, not the slug. Files
    that fail to parse are silently skipped; the on-disk store is treated
    as best-effort, and a corrupt file should not block the rest of the
    listing.
    """
    if not PROFILES_DIR.exists():
        return []
    items: list[tuple[str, dt.datetime]] = []
    for path in PROFILES_DIR.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            name = payload["name"]
            created = dt.datetime.fromisoformat(payload["created_at"])
            items.append((name, created))
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    items.sort(key=lambda entry: entry[1], reverse=True)
    return items


def load_preset(name: str) -> UserProfile:
    """Look up a preset profile by its key (e.g. 'chill_lofi')."""
    if name not in PRESET_PROFILES:
        raise ProfileNotFoundError(
            f"No preset named {name!r}; available: {sorted(PRESET_PROFILES)}"
        )
    return PRESET_PROFILES[name]


def edit_profile_fields(name: str, **updates: Any) -> UserProfile:
    """Apply field-level updates to a saved profile and write it back.

    Loads the existing user profile via load_profile(name), applies any
    updates whose keys match the seven UserProfile dataclass fields,
    re-saves with overwrite=True, and returns the updated profile. The
    save preserves the original human-readable name and refreshes the
    created_at timestamp.

    Raises ValueError (with a directive message) if `name` matches a key
    in PRESET_PROFILES — presets are immutable per D33; the caller is
    pointed to `build --from-preset NAME --save NEW_NAME`.

    Raises ProfileNotFoundError if the profile does not exist on disk.
    Unknown update keys raise ValueError listing the allowed fields so
    typos do not silently no-op.
    """
    if name in PRESET_PROFILES:
        raise ValueError(
            f"{name!r} is a preset and cannot be edited in place; "
            f"use `build --from-preset {name} --save NEW_NAME` to derive "
            f"a new user profile from it."
        )

    unknown = set(updates) - _USER_PROFILE_FIELDS
    if unknown:
        raise ValueError(
            f"Unknown profile field(s): {sorted(unknown)}; "
            f"allowed fields are {sorted(_USER_PROFILE_FIELDS)}."
        )

    existing = load_profile(name)
    updated = dataclasses.replace(existing, **updates)
    save_profile(name, updated, overwrite=True)
    return updated
