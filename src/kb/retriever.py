from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.recommender import ScoredRecommendation


KB_ROOT = Path(__file__).resolve().parent / "docs"


class KBLookupError(Exception):
    """Raised when a required KB doc is missing."""


@dataclass
class KBDoc:
    path: Path
    frontmatter: dict
    body: str


@dataclass
class RetrievedContext:
    genre: KBDoc
    mood: KBDoc
    song: KBDoc


def load_doc(path: Path) -> KBDoc:
    if not path.exists():
        raise KBLookupError(f"KB doc not found: {path}")
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---"):
        raise KBLookupError(f"KB doc missing YAML frontmatter: {path}")
    parts = raw.split("---", 2)
    if len(parts) < 3:
        raise KBLookupError(f"KB doc has malformed frontmatter delimiters: {path}")
    frontmatter = yaml.safe_load(parts[1]) or {}
    body = parts[2].lstrip("\n")
    return KBDoc(path=path, frontmatter=frontmatter, body=body)


def retrieve_for_recommendation(rec: ScoredRecommendation) -> RetrievedContext:
    song = rec.song
    genre_path = KB_ROOT / "genres" / f"{song.genre}.md"
    mood_path = KB_ROOT / "moods" / f"{song.mood}.md"
    song_path = KB_ROOT / "songs" / f"{song.id}.md"
    return RetrievedContext(
        genre=load_doc(genre_path),
        mood=load_doc(mood_path),
        song=load_doc(song_path),
    )
