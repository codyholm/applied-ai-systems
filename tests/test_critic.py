from __future__ import annotations

import json

from src.agents.critic import critique
from src.llm.client import StubLLMClient
from src.recommender import ScoredRecommendation, Song, UserProfile


PROFILE = UserProfile(
    favorite_genre="lofi",
    favorite_mood="chill",
    target_energy=0.4,
    target_tempo_bpm=78.0,
    target_valence=0.55,
    target_danceability=0.5,
    target_acousticness=0.8,
)


def _rec(id_, title="Track", genre="lofi", mood="chill", score=10.0):
    song = Song(
        id=id_,
        title=title,
        artist="Tester",
        genre=genre,
        mood=mood,
        energy=0.4,
        tempo_bpm=80.0,
        valence=0.55,
        danceability=0.5,
        acousticness=0.78,
    )
    return ScoredRecommendation(song=song, score=score, reasons=["genre match (+1.5)"])


TOP5 = [_rec(i + 1) for i in range(5)]


def test_critic_ok_verdict():
    response = json.dumps({"verdict": "ok", "adjustments": None, "reason": "matches intent"})
    llm = StubLLMClient([response])

    result = critique("anything", PROFILE, TOP5, llm)

    assert result.verdict == "ok"
    assert result.adjustments is None
    assert result.reason == "matches intent"


def test_critic_refine_with_adjustments():
    response = json.dumps(
        {
            "verdict": "refine",
            "adjustments": {"target_energy": 0.3, "target_tempo_bpm": 75.0},
            "reason": "current top-5 too high-energy",
        }
    )
    llm = StubLLMClient([response])

    result = critique("calmer please", PROFILE, TOP5, llm)

    assert result.verdict == "refine"
    assert result.adjustments == {"target_energy": 0.3, "target_tempo_bpm": 75.0}
    assert "high-energy" in result.reason


def test_critic_parse_failure_degrades_to_ok():
    llm = StubLLMClient(["this is not json"])

    result = critique("anything", PROFILE, TOP5, llm)

    assert result.verdict == "ok"
    assert result.adjustments is None
    assert result.reason == "parse_failure"
