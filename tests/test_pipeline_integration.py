from __future__ import annotations

import json

from src.kb import retriever as retriever_module
from src.llm.client import StubLLMClient
from src.pipeline import run_pipeline
from src.recommender import Song, UserProfile


def _write_doc(path, body, **frontmatter):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["---"]
    for key, value in frontmatter.items():
        lines.append(f"{key}: {value}")
    lines.append("---")
    lines.append("")
    lines.append(body)
    path.write_text("\n".join(lines), encoding="utf-8")


def test_run_pipeline_step1_happy_path(tmp_path, monkeypatch):
    songs = [
        Song(
            id=1,
            title="Library Hours",
            artist="Tester",
            genre="lofi",
            mood="chill",
            energy=0.40,
            tempo_bpm=78.0,
            valence=0.60,
            danceability=0.60,
            acousticness=0.78,
        ),
        Song(
            id=2,
            title="Tape Loop",
            artist="Tester",
            genre="lofi",
            mood="chill",
            energy=0.42,
            tempo_bpm=80.0,
            valence=0.58,
            danceability=0.62,
            acousticness=0.75,
        ),
        Song(
            id=3,
            title="Quiet Window",
            artist="Tester",
            genre="lofi",
            mood="chill",
            energy=0.38,
            tempo_bpm=76.0,
            valence=0.62,
            danceability=0.58,
            acousticness=0.80,
        ),
    ]

    _write_doc(
        tmp_path / "genres" / "lofi.md",
        "Lofi sits between 60 and 90 BPM and leans on warm vinyl-style filters.",
        type="genre",
        name="lofi",
    )
    _write_doc(
        tmp_path / "moods" / "chill.md",
        "Chill describes a low-energy listening state with mid valence.",
        type="mood",
        name="chill",
    )
    for song in songs:
        _write_doc(
            tmp_path / "songs" / f"{song.id}.md",
            f"Song {song.id} fact: track is built for headphones.",
            type="song",
            id=song.id,
        )

    monkeypatch.setattr(retriever_module, "KB_ROOT", tmp_path)

    profile = UserProfile(
        favorite_genre="lofi",
        favorite_mood="chill",
        target_energy=0.40,
        target_tempo_bpm=78.0,
        target_valence=0.60,
        target_danceability=0.60,
        target_acousticness=0.78,
    )

    response = json.dumps(
        {
            "explanations": [
                {
                    "song_id": song_id,
                    "text": "Fits because lofi sits between 60 and 90 BPM.",
                    "cited_snippets": [
                        "Lofi sits between 60 and 90 BPM",
                        f"Song {song_id} fact: track is built for headphones",
                    ],
                }
                for song_id in (1, 2, 3)
            ]
        }
    )
    llm = StubLLMClient([response])

    result = run_pipeline(profile, llm=llm, songs=songs, k=3)

    assert len(result.recommendations) == 3
    assert len(result.retrieved_contexts) == 3
    assert len(result.explanations) == 3
    for expl in result.explanations:
        assert expl.text is not None
        assert expl.fallback_reason is None
        assert any("Lofi sits" in s for s in expl.cited_snippets)
    assert result.ambiguous_match is False
    assert result.refinement_history == []
    assert result.extracted_profile is profile
    assert result.final_profile is profile
    assert result.cache_stats is None  # plain stub, not wrapped in CachedLLMClient
