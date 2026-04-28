from src.recommender import (
    ScoredRecommendation,
    Song,
    UserProfile,
    load_songs,
    recommend_songs,
    score_song,
)


# --- Fixtures ----------------------------------------------------------------


def make_pop_song(**overrides) -> Song:
    defaults = dict(
        id=1,
        title="Test Pop Track",
        artist="Test Artist",
        genre="pop",
        mood="happy",
        energy=0.85,
        tempo_bpm=120.0,
        valence=0.9,
        danceability=0.85,
        acousticness=0.2,
    )
    defaults.update(overrides)
    return Song(**defaults)


def make_lofi_song(**overrides) -> Song:
    defaults = dict(
        id=2,
        title="Chill Lofi Loop",
        artist="Test Artist",
        genre="lofi",
        mood="chill",
        energy=0.4,
        tempo_bpm=80.0,
        valence=0.6,
        danceability=0.5,
        acousticness=0.85,
    )
    defaults.update(overrides)
    return Song(**defaults)


def make_pop_profile(**overrides) -> UserProfile:
    defaults = dict(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.85,
        target_tempo_bpm=120.0,
        target_valence=0.9,
        target_danceability=0.85,
        target_acousticness=0.2,
    )
    defaults.update(overrides)
    return UserProfile(**defaults)


# --- load_songs --------------------------------------------------------------


def test_load_songs_returns_list_of_song_dataclasses():
    songs = load_songs()
    assert len(songs) > 0
    assert all(isinstance(s, Song) for s in songs)


def test_load_songs_uses_default_csv_when_no_arg(tmp_path, monkeypatch):
    # Switching cwd must not break load_songs() — the path is anchored to the
    # package, not the working directory.
    monkeypatch.chdir(tmp_path)
    songs = load_songs()
    assert len(songs) > 0


# --- Song.from_csv_row --------------------------------------------------------


def test_song_from_csv_row_casts_string_fields_to_correct_types():
    row = {
        "id": "42",
        "title": "Some Track",
        "artist": "Some Artist",
        "genre": "rock",
        "mood": "intense",
        "energy": "0.9",
        "tempo_bpm": "145",
        "valence": "0.5",
        "danceability": "0.7",
        "acousticness": "0.1",
    }
    song = Song.from_csv_row(row)
    assert song.id == 42
    assert isinstance(song.id, int)
    assert song.energy == 0.9
    assert isinstance(song.energy, float)
    assert song.tempo_bpm == 145.0
    assert isinstance(song.tempo_bpm, float)
    assert song.title == "Some Track"


# --- UserProfile.from_dict ---------------------------------------------------


def test_user_profile_from_dict_builds_instance_with_type_casts():
    profile = UserProfile.from_dict(
        {
            "favorite_genre": "pop",
            "favorite_mood": "happy",
            "target_energy": "0.85",
            "target_tempo_bpm": 124,
            "target_valence": 0.85,
            "target_danceability": 0.85,
            "target_acousticness": 0.15,
        }
    )
    assert isinstance(profile, UserProfile)
    assert profile.favorite_genre == "pop"
    assert profile.target_energy == 0.85
    assert isinstance(profile.target_energy, float)
    assert profile.target_tempo_bpm == 124.0
    assert isinstance(profile.target_tempo_bpm, float)


def test_user_profile_from_dict_omits_avoid_genres_defaults_to_empty_list():
    # Back-compat: profiles persisted before avoid_genres existed must still
    # load. Missing key -> empty list, not a KeyError.
    profile = UserProfile.from_dict(
        {
            "favorite_genre": "pop",
            "favorite_mood": "happy",
            "target_energy": 0.85,
            "target_tempo_bpm": 124.0,
            "target_valence": 0.85,
            "target_danceability": 0.85,
            "target_acousticness": 0.15,
        }
    )
    assert profile.avoid_genres == []


def test_user_profile_from_dict_lowercases_avoid_genres():
    profile = UserProfile.from_dict(
        {
            "favorite_genre": "pop",
            "favorite_mood": "happy",
            "target_energy": 0.85,
            "target_tempo_bpm": 124.0,
            "target_valence": 0.85,
            "target_danceability": 0.85,
            "target_acousticness": 0.15,
            "avoid_genres": ["Country", "METAL"],
        }
    )
    assert profile.avoid_genres == ["country", "metal"]


# --- score_song --------------------------------------------------------------


def test_score_song_returns_score_and_reasons_list():
    score, reasons = score_song(make_pop_profile(), make_pop_song())
    assert isinstance(score, float)
    assert isinstance(reasons, list)
    assert all(isinstance(r, str) for r in reasons)
    assert len(reasons) > 0


def test_score_song_higher_when_genre_and_mood_match():
    pop_profile = make_pop_profile()
    pop_song = make_pop_song()
    lofi_song = make_lofi_song()

    pop_score, _ = score_song(pop_profile, pop_song)
    lofi_score, _ = score_song(pop_profile, lofi_song)

    assert pop_score > lofi_score


def test_score_song_zeros_when_genre_in_avoid_list():
    profile = make_pop_profile(avoid_genres=["lofi"])
    score, reasons = score_song(profile, make_lofi_song())
    assert score == 0.0
    assert reasons == ["avoided genre: lofi"]


def test_score_song_avoid_is_case_insensitive():
    profile = make_pop_profile(avoid_genres=["LoFi"])
    score, reasons = score_song(profile, make_lofi_song())
    assert score == 0.0
    assert reasons == ["avoided genre: lofi"]


def test_score_song_unaffected_when_avoid_genres_empty():
    # Regression: ensure the short-circuit doesn't change scoring math when
    # avoid_genres is empty (the default).
    profile = make_pop_profile(avoid_genres=[])
    score, reasons = score_song(profile, make_pop_song())
    assert score > 0.0
    assert "avoided" not in " ".join(reasons)


# --- recommend_songs ---------------------------------------------------------


def test_recommend_songs_returns_sorted_scored_recommendations():
    songs = [make_pop_song(), make_lofi_song()]
    results = recommend_songs(make_pop_profile(), songs, k=2)

    assert len(results) == 2
    assert all(isinstance(r, ScoredRecommendation) for r in results)
    assert results[0].score >= results[1].score
    assert results[0].song.genre == "pop"


def test_recommend_songs_respects_k_parameter():
    songs = [make_pop_song(id=i) for i in range(5)]
    results = recommend_songs(make_pop_profile(), songs, k=3)
    assert len(results) == 3
