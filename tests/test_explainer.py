from __future__ import annotations

import json
from pathlib import Path

from src.agents.explainer import explain_recommendations
from src.kb.retriever import KBDoc, RetrievedContext
from src.llm.client import StubLLMClient
from src.recommender import ScoredRecommendation, Song, UserProfile


PROFILE = UserProfile(
    favorite_genre="lofi",
    favorite_mood="chill",
    target_energy=0.4,
    target_tempo_bpm=78.0,
    target_valence=0.6,
    target_danceability=0.6,
    target_acousticness=0.7,
)


def _song(id_, **overrides):
    defaults = dict(
        id=id_,
        title=f"Track {id_}",
        artist="Tester",
        genre="lofi",
        mood="chill",
        energy=0.4,
        tempo_bpm=80.0,
        valence=0.6,
        danceability=0.6,
        acousticness=0.7,
    )
    defaults.update(overrides)
    return Song(**defaults)


def _rec(id_, **overrides):
    return ScoredRecommendation(song=_song(id_, **overrides), score=10.0, reasons=["genre match (+1.5)"])


def _ctx(genre_body: str, mood_body: str, song_body: str) -> RetrievedContext:
    return RetrievedContext(
        genre=KBDoc(path=Path("genre.md"), frontmatter={}, body=genre_body),
        mood=KBDoc(path=Path("mood.md"), frontmatter={}, body=mood_body),
        song=KBDoc(path=Path("song.md"), frontmatter={}, body=song_body),
    )


def _build_inputs(n: int = 5):
    recs = [_rec(i + 1) for i in range(n)]
    contexts = [
        _ctx(
            genre_body=f"Genre {i + 1} fact: lofi sits between 60 and 90 BPM.",
            mood_body=f"Mood {i + 1} fact: chill sits at energy 0.20 to 0.55.",
            song_body=f"Song {i + 1} fact: track id {i + 1} is built for headphones.",
        )
        for i in range(n)
    ]
    return recs, contexts


def test_explain_happy_path():
    recs, contexts = _build_inputs()
    response = json.dumps(
        {
            "explanations": [
                {
                    "song_id": rec.song.id,
                    "text": f"This track works because lofi sits between 60 and 90 BPM and the chill mood matches.",
                    "cited_snippets": [
                        "lofi sits between 60 and 90 BPM",
                        f"track id {rec.song.id} is built for headphones",
                    ],
                }
                for rec in recs
            ]
        }
    )
    llm = StubLLMClient([response])

    explanations = explain_recommendations(PROFILE, recs, contexts, llm)

    assert len(explanations) == 5
    for rec, expl in zip(recs, explanations):
        assert expl.song_id == rec.song.id
        assert expl.text and "lofi sits between" in expl.text
        assert expl.fallback_reason is None
        assert len(expl.cited_snippets) >= 1


def test_explain_falls_back_on_malformed_json():
    recs, contexts = _build_inputs()
    llm = StubLLMClient(["this is not json"])

    explanations = explain_recommendations(PROFILE, recs, contexts, llm)

    assert len(explanations) == 5
    for expl in explanations:
        assert expl.text is None
        assert expl.cited_snippets == []
        assert expl.fallback_reason == "json_parse_error"


def test_explain_falls_back_on_fabricated_citation():
    recs, contexts = _build_inputs()
    response = json.dumps(
        {
            "explanations": [
                {
                    "song_id": rec.song.id,
                    "text": "Plausible-sounding paragraph.",
                    "cited_snippets": ["this exact string does not appear in any retrieved doc"],
                }
                for rec in recs
            ]
        }
    )
    llm = StubLLMClient([response])

    explanations = explain_recommendations(PROFILE, recs, contexts, llm)

    assert len(explanations) == 5
    for expl in explanations:
        assert expl.text is None
        assert expl.fallback_reason == "fabricated_citation"
