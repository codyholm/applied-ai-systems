"""Tests for the profile store and preset registry."""
from __future__ import annotations

import json

import pytest

from src import profiles as profiles_module
from src.profiles import (
    PRESET_PROFILES,
    ProfileExistsError,
    ProfileNotFoundError,
    delete_profile,
    edit_profile_fields,
    list_profiles,
    load_preset,
    load_profile,
    save_profile,
    slugify,
)
from src.recommender import UserProfile


@pytest.fixture
def tmp_profiles_dir(tmp_path, monkeypatch):
    """Redirect PROFILES_DIR to a tmp path for the test scope."""
    target = tmp_path / "profiles"
    monkeypatch.setattr(profiles_module, "PROFILES_DIR", target)
    return target


def _sample_profile() -> UserProfile:
    return UserProfile(
        favorite_genre="lofi",
        favorite_mood="chill",
        target_energy=0.4,
        target_tempo_bpm=78.0,
        target_valence=0.6,
        target_danceability=0.5,
        target_acousticness=0.8,
    )


def test_slugify_normalizes_whitespace_and_case_and_punctuation():
    assert slugify("Calm Acoustic") == "calm-acoustic"
    assert slugify("  Study Mode!  ") == "study-mode"
    assert slugify("Mix #2 (work)") == "mix-2-work"


def test_slugify_rejects_empty_or_punctuation_only_input():
    with pytest.raises(ValueError):
        slugify("!!!")


def test_save_and_load_round_trip(tmp_profiles_dir):
    original = _sample_profile()
    path = save_profile("Calm Acoustic", original)

    assert path.parent == tmp_profiles_dir
    assert path.name == "calm-acoustic.json"

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["name"] == "Calm Acoustic"
    assert "created_at" in payload
    assert payload["profile"]["favorite_genre"] == "lofi"

    reloaded = load_profile("Calm Acoustic")
    assert reloaded == original


def test_save_with_overwrite_false_raises_on_collision(tmp_profiles_dir):
    save_profile("study", _sample_profile())
    with pytest.raises(ProfileExistsError):
        save_profile("study", _sample_profile(), overwrite=False)


def test_save_with_overwrite_true_clobbers(tmp_profiles_dir):
    save_profile("study", _sample_profile())
    new_profile = UserProfile(
        favorite_genre="ambient",
        favorite_mood="relaxed",
        target_energy=0.2,
        target_tempo_bpm=60.0,
        target_valence=0.5,
        target_danceability=0.3,
        target_acousticness=0.95,
    )
    save_profile("study", new_profile, overwrite=True)
    assert load_profile("study") == new_profile


def test_list_profiles_returns_newest_first(tmp_profiles_dir):
    save_profile("alpha", _sample_profile())
    save_profile("beta", _sample_profile())
    listed = list_profiles()
    names = [name for name, _ in listed]
    assert "alpha" in names and "beta" in names
    # Newest first by created_at
    assert listed[0][1] >= listed[-1][1]


def test_list_profiles_skips_corrupt_files(tmp_profiles_dir):
    save_profile("good", _sample_profile())
    (tmp_profiles_dir / "bad.json").write_text("not json", encoding="utf-8")
    names = [name for name, _ in list_profiles()]
    assert names == ["good"]


def test_delete_profile_removes_file(tmp_profiles_dir):
    save_profile("temp", _sample_profile())
    delete_profile("temp")
    with pytest.raises(ProfileNotFoundError):
        load_profile("temp")


def test_load_missing_profile_raises(tmp_profiles_dir):
    with pytest.raises(ProfileNotFoundError):
        load_profile("never-saved")


def test_delete_missing_profile_raises(tmp_profiles_dir):
    with pytest.raises(ProfileNotFoundError):
        delete_profile("never-saved")


def test_preset_profiles_has_five_module3_entries():
    assert set(PRESET_PROFILES) == {
        "high_energy_pop",
        "chill_lofi",
        "deep_intense_rock",
        "chill_rock",
        "boundary_maximalist",
    }


def test_load_preset_returns_module3_user_profile():
    profile = load_preset("chill_lofi")
    assert profile.favorite_genre == "lofi"
    assert profile.favorite_mood == "chill"
    assert profile.target_tempo_bpm == 78.0


def test_load_preset_unknown_raises():
    with pytest.raises(ProfileNotFoundError):
        load_preset("not-a-preset")


def test_presets_match_main_module_literals():
    """Sanity: the preset constants must equal the literals in src/main.py.

    src/main.py is the legacy 5-profile demo; if these drift, the eval
    harness and the demo print different things.
    """
    from src.main import user_profiles as main_user_profiles

    main_dict = {name: profile for name, profile in main_user_profiles}
    # Module 3 main.py uses display names; map to our slug keys via PRESET_DISPLAY_NAMES.
    from src.profiles import PRESET_DISPLAY_NAMES

    for slug, display in PRESET_DISPLAY_NAMES.items():
        assert main_dict[display] == PRESET_PROFILES[slug], (
            f"Preset '{slug}' diverges from src/main.py '{display}' literal"
        )


def test_edit_profile_fields_applies_whitelisted_updates(tmp_profiles_dir):
    save_profile("study mode", _sample_profile())

    updated = edit_profile_fields(
        "study mode",
        target_energy=0.55,
        favorite_mood="happy",
    )

    assert updated.target_energy == 0.55
    assert updated.favorite_mood == "happy"
    # Other fields untouched.
    assert updated.favorite_genre == "lofi"
    assert updated.target_tempo_bpm == 78.0

    # Persisted to disk and re-loadable with the same values.
    reloaded = load_profile("study mode")
    assert reloaded == updated


def test_edit_profile_fields_rejects_preset_name():
    with pytest.raises(ValueError, match="preset"):
        edit_profile_fields("chill_lofi", target_energy=0.1)


def test_edit_profile_fields_rejects_unknown_field(tmp_profiles_dir):
    save_profile("study mode", _sample_profile())

    with pytest.raises(ValueError, match="Unknown profile field"):
        edit_profile_fields("study mode", target_loudness=0.5)


def test_edit_profile_fields_raises_when_profile_missing(tmp_profiles_dir):
    with pytest.raises(ProfileNotFoundError):
        edit_profile_fields("never-saved", target_energy=0.5)
