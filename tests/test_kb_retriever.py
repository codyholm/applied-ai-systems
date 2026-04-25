from __future__ import annotations

import pytest

from src.kb import retriever as retriever_module
from src.kb.retriever import KBLookupError, load_doc, retrieve_for_recommendation
from src.recommender import ScoredRecommendation, Song


def _write_doc(path, body, **frontmatter):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["---"]
    for key, value in frontmatter.items():
        lines.append(f"{key}: {value}")
    lines.append("---")
    lines.append("")
    lines.append(body)
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_rec(**overrides) -> ScoredRecommendation:
    defaults = dict(
        id=42,
        title="Test Track",
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
    song = Song(**defaults)
    return ScoredRecommendation(song=song, score=10.0, reasons=["genre match (+1.5)"])


def test_load_doc_parses_frontmatter_and_body(tmp_path):
    path = tmp_path / "sample.md"
    _write_doc(path, "Plain body text here.", type="genre", name="lofi")

    doc = load_doc(path)

    assert doc.path == path
    assert doc.frontmatter == {"type": "genre", "name": "lofi"}
    assert "Plain body text here." in doc.body


def test_retrieve_for_recommendation_returns_three_docs(tmp_path, monkeypatch):
    _write_doc(tmp_path / "genres" / "lofi.md", "Lofi genre body.", type="genre", name="lofi")
    _write_doc(tmp_path / "moods" / "chill.md", "Chill mood body.", type="mood", name="chill")
    _write_doc(tmp_path / "songs" / "42.md", "Track 42 body.", type="song", id=42)
    monkeypatch.setattr(retriever_module, "KB_ROOT", tmp_path)

    context = retrieve_for_recommendation(_make_rec())

    assert "Lofi genre body." in context.genre.body
    assert "Chill mood body." in context.mood.body
    assert "Track 42 body." in context.song.body


def test_retrieve_raises_on_missing_doc(tmp_path, monkeypatch):
    _write_doc(tmp_path / "genres" / "lofi.md", "Lofi genre body.", type="genre", name="lofi")
    _write_doc(tmp_path / "moods" / "chill.md", "Chill mood body.", type="mood", name="chill")
    # Intentionally do NOT write the song doc.
    monkeypatch.setattr(retriever_module, "KB_ROOT", tmp_path)

    with pytest.raises(KBLookupError, match="42.md"):
        retrieve_for_recommendation(_make_rec())
