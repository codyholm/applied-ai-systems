from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

GENRE_WEIGHT = 1.5
MOOD_WEIGHT = 2.0
ENERGY_WEIGHT = 2.5
TEMPO_WEIGHT = 2.0
ACOUSTIC_WEIGHT = 1.5
VALENCE_WEIGHT = 2.0
DANCEABILITY_WEIGHT = 1.5
TEMPO_RANGE = 92.0

DEFAULT_SONGS_CSV = Path(__file__).resolve().parent.parent / "data" / "songs.csv"


@dataclass
class Song:
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "Song":
        return cls(
            id=int(row["id"]),
            title=row["title"],
            artist=row["artist"],
            genre=row["genre"],
            mood=row["mood"],
            energy=float(row["energy"]),
            tempo_bpm=float(row["tempo_bpm"]),
            valence=float(row["valence"]),
            danceability=float(row["danceability"]),
            acousticness=float(row["acousticness"]),
        )


@dataclass
class UserProfile:
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    target_tempo_bpm: float
    target_valence: float
    target_danceability: float
    target_acousticness: float

    @classmethod
    def from_dict(cls, d: dict) -> "UserProfile":
        # Plain type casts only. Clamping/validation is the Profile Extractor's
        # job in Step 2; keeping this boundary thin on purpose.
        return cls(
            favorite_genre=str(d["favorite_genre"]),
            favorite_mood=str(d["favorite_mood"]),
            target_energy=float(d["target_energy"]),
            target_tempo_bpm=float(d["target_tempo_bpm"]),
            target_valence=float(d["target_valence"]),
            target_danceability=float(d["target_danceability"]),
            target_acousticness=float(d["target_acousticness"]),
        )


@dataclass
class ScoredRecommendation:
    song: Song
    score: float
    reasons: list[str]


def load_songs(csv_path: Path | str | None = None) -> list[Song]:
    """Load songs from a CSV file into a list of Song dataclasses."""
    path = Path(csv_path) if csv_path is not None else DEFAULT_SONGS_CSV
    with open(path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return [Song.from_csv_row(row) for row in reader]


def _clamp_similarity(value: float) -> float:
    """Clamp a similarity score to the 0.0 to 1.0 range."""
    return max(0.0, min(value, 1.0))


def score_song(user: UserProfile, song: Song) -> tuple[float, list[str]]:
    """Score one song against a user's preferences and explain why."""
    score = 0.0
    reasons: list[str] = []

    if user.favorite_genre == song.genre:
        score += GENRE_WEIGHT
        reasons.append(f"genre match (+{GENRE_WEIGHT:.1f})")

    if user.favorite_mood == song.mood:
        score += MOOD_WEIGHT
        reasons.append(f"mood match (+{MOOD_WEIGHT:.1f})")

    energy_similarity = _clamp_similarity(1 - abs(song.energy - user.target_energy))
    energy_points = ENERGY_WEIGHT * energy_similarity
    score += energy_points
    reasons.append(f"energy similarity (+{energy_points:.2f})")

    tempo_similarity = _clamp_similarity(
        1 - min(abs(song.tempo_bpm - user.target_tempo_bpm) / TEMPO_RANGE, 1)
    )
    tempo_points = TEMPO_WEIGHT * tempo_similarity
    score += tempo_points
    reasons.append(f"tempo similarity (+{tempo_points:.2f})")

    acoustic_similarity = _clamp_similarity(
        1 - abs(song.acousticness - user.target_acousticness)
    )
    acoustic_points = ACOUSTIC_WEIGHT * acoustic_similarity
    score += acoustic_points
    reasons.append(f"acousticness similarity (+{acoustic_points:.2f})")

    valence_similarity = _clamp_similarity(
        1 - abs(song.valence - user.target_valence)
    )
    valence_points = VALENCE_WEIGHT * valence_similarity
    score += valence_points
    reasons.append(f"valence similarity (+{valence_points:.2f})")

    danceability_similarity = _clamp_similarity(
        1 - abs(song.danceability - user.target_danceability)
    )
    danceability_points = DANCEABILITY_WEIGHT * danceability_similarity
    score += danceability_points
    reasons.append(f"danceability similarity (+{danceability_points:.2f})")

    return score, reasons


def recommend_songs(
    user: UserProfile, songs: list[Song], k: int = 5
) -> list[ScoredRecommendation]:
    """Rank all songs for a user and return the top-k recommendations."""
    scored: list[ScoredRecommendation] = []
    for song in songs:
        score, reasons = score_song(user, song)
        scored.append(ScoredRecommendation(song=song, score=score, reasons=reasons))

    scored.sort(key=lambda rec: rec.score, reverse=True)
    return scored[:k]
