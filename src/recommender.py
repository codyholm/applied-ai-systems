import csv
from typing import Any, List, Dict, Tuple, Optional
from dataclasses import dataclass

GENRE_WEIGHT = 1.5
MOOD_WEIGHT = 2.0
ENERGY_WEIGHT = 2.5
TEMPO_WEIGHT = 2.0
ACOUSTIC_WEIGHT = 1.5
TEMPO_RANGE = 92.0

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
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

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        # TODO: Implement recommendation logic
        return self.songs[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        # TODO: Implement explanation logic
        return "Explanation placeholder"

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    songs: List[Dict] = []

    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            row["id"] = int(row["id"])
            row["energy"] = float(row["energy"])
            row["tempo_bpm"] = float(row["tempo_bpm"])
            row["valence"] = float(row["valence"])
            row["danceability"] = float(row["danceability"])
            row["acousticness"] = float(row["acousticness"])
            songs.append(row)

    print(f"Loaded songs: {len(songs)}")
    return songs


def _clamp_similarity(value: float) -> float:
    return max(0.0, min(value, 1.0))


def _get_user_pref(user_prefs: Dict[str, Any], *keys: str) -> Optional[Any]:
    for key in keys:
        if key in user_prefs and user_prefs[key] is not None:
            return user_prefs[key]
    return None


def score_song(user_prefs: Dict[str, Any], song: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    Scores one song against a user's preferences and returns reasons.
    """
    score = 0.0
    reasons: List[str] = []

    favorite_genre = _get_user_pref(user_prefs, "favorite_genre", "genre")
    favorite_mood = _get_user_pref(user_prefs, "favorite_mood", "mood")
    target_energy = _get_user_pref(user_prefs, "target_energy", "energy")
    target_tempo = _get_user_pref(user_prefs, "target_tempo_bpm", "tempo_bpm")
    target_acousticness = _get_user_pref(
        user_prefs, "target_acousticness", "acousticness"
    )

    if favorite_genre == song["genre"]:
        score += GENRE_WEIGHT
        reasons.append(f"genre match (+{GENRE_WEIGHT:.1f})")

    if favorite_mood == song["mood"]:
        score += MOOD_WEIGHT
        reasons.append(f"mood match (+{MOOD_WEIGHT:.1f})")

    if target_energy is not None:
        energy_similarity = _clamp_similarity(1 - abs(song["energy"] - float(target_energy)))
        energy_points = ENERGY_WEIGHT * energy_similarity
        score += energy_points
        reasons.append(f"energy similarity (+{energy_points:.2f})")

    if target_tempo is not None:
        tempo_similarity = _clamp_similarity(
            1 - min(abs(song["tempo_bpm"] - float(target_tempo)) / TEMPO_RANGE, 1)
        )
        tempo_points = TEMPO_WEIGHT * tempo_similarity
        score += tempo_points
        reasons.append(f"tempo similarity (+{tempo_points:.2f})")

    if target_acousticness is None and "likes_acoustic" in user_prefs:
        target_acousticness = 1.0 if user_prefs["likes_acoustic"] else 0.0

    if target_acousticness is not None:
        acoustic_similarity = _clamp_similarity(
            1 - abs(song["acousticness"] - float(target_acousticness))
        )
        acoustic_points = ACOUSTIC_WEIGHT * acoustic_similarity
        score += acoustic_points
        reasons.append(f"acousticness similarity (+{acoustic_points:.2f})")

    return score, reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    """
    # TODO: Implement scoring and ranking logic
    # Expected return format: (song_dict, score, explanation)
    return []
