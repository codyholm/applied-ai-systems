from __future__ import annotations

import json

from src.kb import retriever as retriever_module
from src.llm.client import StubLLMClient
from src.pipeline import run_pipeline
from src.recommender import Song


def _write_doc(path, body, **frontmatter):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["---"]
    for key, value in frontmatter.items():
        lines.append(f"{key}: {value}")
    lines.append("---")
    lines.append("")
    lines.append(body)
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_fixture_kb(tmp_path, songs):
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


def _fixture_songs():
    return [
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


_EXTRACTOR_JSON = json.dumps(
    {
        "favorite_genre": "lofi",
        "favorite_mood": "chill",
        "target_energy": 0.40,
        "target_tempo_bpm": 78.0,
        "target_valence": 0.60,
        "target_danceability": 0.60,
        "target_acousticness": 0.78,
    }
)


def _explainer_json(song_ids):
    return json.dumps(
        {
            "explanations": [
                {
                    "song_id": sid,
                    "text": "Fits because lofi sits between 60 and 90 BPM.",
                    "cited_snippets": [
                        "Lofi sits between 60 and 90 BPM",
                        f"Song {sid} fact: track is built for headphones",
                    ],
                }
                for sid in song_ids
            ]
        }
    )


def test_run_pipeline_nl_happy_path(tmp_path, monkeypatch):
    songs = _fixture_songs()
    _build_fixture_kb(tmp_path, songs)
    monkeypatch.setattr(retriever_module, "KB_ROOT", tmp_path)

    critic_ok = json.dumps({"verdict": "ok", "adjustments": None, "reason": "matches"})
    llm = StubLLMClient([_EXTRACTOR_JSON, critic_ok, _explainer_json([1, 2, 3])])

    result = run_pipeline("chill lofi for studying", llm=llm, songs=songs, k=3)

    assert len(result.recommendations) == 3
    assert len(result.explanations) == 3
    for expl in result.explanations:
        assert expl.text is not None
        assert expl.fallback_reason is None
    assert result.refinement_history[0].verdict == "ok"
    assert result.ambiguous_match is False
    assert result.extracted_profile == result.final_profile


def test_pipeline_critic_ok_on_iter_0(tmp_path, monkeypatch):
    songs = _fixture_songs()
    _build_fixture_kb(tmp_path, songs)
    monkeypatch.setattr(retriever_module, "KB_ROOT", tmp_path)

    critic_ok = json.dumps({"verdict": "ok", "adjustments": None, "reason": "matches"})
    llm = StubLLMClient([_EXTRACTOR_JSON, critic_ok, _explainer_json([1, 2, 3])])

    result = run_pipeline("anything", llm=llm, songs=songs, k=3)

    assert len(result.refinement_history) == 1
    assert result.refinement_history[0].iter_index == 0
    assert result.refinement_history[0].verdict == "ok"
    assert result.ambiguous_match is False


def test_pipeline_critic_refines_then_oks(tmp_path, monkeypatch):
    songs = _fixture_songs()
    _build_fixture_kb(tmp_path, songs)
    monkeypatch.setattr(retriever_module, "KB_ROOT", tmp_path)

    critic_refine = json.dumps(
        {
            "verdict": "refine",
            "adjustments": {"target_tempo_bpm": 70.0},
            "reason": "slow it down",
        }
    )
    critic_ok = json.dumps({"verdict": "ok", "adjustments": None, "reason": "matches now"})
    llm = StubLLMClient(
        [_EXTRACTOR_JSON, critic_refine, critic_ok, _explainer_json([1, 2, 3])]
    )

    result = run_pipeline("slower please", llm=llm, songs=songs, k=3)

    assert len(result.refinement_history) == 2
    assert result.refinement_history[0].verdict == "refine"
    assert result.refinement_history[1].verdict == "ok"
    assert result.ambiguous_match is False
    assert result.final_profile.target_tempo_bpm == 70.0
    assert result.extracted_profile.target_tempo_bpm == 78.0
    # Extracted untouched, final reflects adjustment.
    assert result.final_profile != result.extracted_profile


def test_pipeline_ambiguous_match(tmp_path, monkeypatch):
    songs = _fixture_songs()
    _build_fixture_kb(tmp_path, songs)
    monkeypatch.setattr(retriever_module, "KB_ROOT", tmp_path)

    critic_refine_a = json.dumps(
        {
            "verdict": "refine",
            "adjustments": {"target_energy": 0.3},
            "reason": "still too high",
        }
    )
    critic_refine_b = json.dumps(
        {
            "verdict": "refine",
            "adjustments": {"target_energy": 0.2},
            "reason": "even lower",
        }
    )
    llm = StubLLMClient(
        [_EXTRACTOR_JSON, critic_refine_a, critic_refine_b, _explainer_json([1, 2, 3])]
    )

    result = run_pipeline("calmer calmer calmer", llm=llm, songs=songs, k=3)

    assert len(result.refinement_history) == 2
    assert result.refinement_history[0].verdict == "refine"
    assert result.refinement_history[1].verdict == "refine"
    assert result.ambiguous_match is True
