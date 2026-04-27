"""Streamlit UI for the applied AI music recommender (Phase 6).

Three top-level tabs cover the full workflow without needing the CLI:

- Recommend: pick a saved profile or preset and get top-k grounded
  recommendations.
- Build profile: answer the five guided questions plus an optional
  free-form description; optionally seed from an existing profile
  or preset; optionally save the result under a name.
- Manage profiles: view, edit, or delete saved profiles; view presets
  and use any of them as a build seed.

The page is GEMINI_API_KEY-aware: with a key, build_profile and
recommend issue real LLM calls (cached via CachedLLMClient). Without a
key, the page renders a stub-mode warning and most actions fail with a
clear message — extraction and grounded explanations both need an LLM.
"""

from __future__ import annotations

import functools
import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from src.agents.profile_extractor import ProfileExtractionError
from src.llm.client import CachedLLMClient, GeminiClient, LLMClient, StubLLMClient
from src.pipeline import (
    BuildInputs,
    EmptyBuildInputsError,
    ProfileBuildResult,
    RecommendationResult,
    build_profile,
    recommend,
)
from src.profiles import (
    PRESET_DISPLAY_NAMES,
    PRESET_PROFILES,
    ProfileExistsError,
    ProfileNotFoundError,
    delete_profile,
    edit_profile_fields,
    list_profiles,
    load_preset,
    load_profile,
    save_profile,
)
from src.recommender import UserProfile, load_songs


load_dotenv()


# ---------------------------------------------------------------------------
# Client construction (cached for the session)
# ---------------------------------------------------------------------------


@st.cache_resource
def _get_client() -> LLMClient:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if key:
        return CachedLLMClient(GeminiClient(api_key=key))
    return StubLLMClient([])


def _is_online(client: LLMClient) -> bool:
    return isinstance(client, CachedLLMClient)


# ---------------------------------------------------------------------------
# Catalog-derived option lists (for Edit selectboxes)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _allowed_genres() -> list[str]:
    return sorted({s.genre for s in load_songs()})


@functools.lru_cache(maxsize=1)
def _allowed_moods() -> list[str]:
    return sorted({s.mood for s in load_songs()})


# ---------------------------------------------------------------------------
# Profile picker option helpers (used in Recommend + Build seed pickers)
# ---------------------------------------------------------------------------


def _profile_picker_options(*, include_none: bool = False) -> list[tuple[str, str, str]]:
    """Return [(label, kind, name), ...] for a unified profile picker.

    `kind` is "saved" or "preset". `label` is what the user sees.
    """
    options: list[tuple[str, str, str]] = []
    if include_none:
        options.append(("(none)", "none", ""))
    for name, _created in list_profiles():
        options.append((f"Saved: {name}", "saved", name))
    for slug, display in PRESET_DISPLAY_NAMES.items():
        options.append((f"Preset: {display}", "preset", slug))
    return options


def _resolve_picked(kind: str, name: str) -> UserProfile | None:
    if kind == "saved":
        return load_profile(name)
    if kind == "preset":
        return load_preset(name)
    return None


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _profile_row_dict(profile: UserProfile) -> dict[str, list[Any]]:
    return {
        "favorite_genre": [profile.favorite_genre],
        "favorite_mood": [profile.favorite_mood],
        "target_energy": [round(profile.target_energy, 3)],
        "target_tempo_bpm": [round(profile.target_tempo_bpm, 1)],
        "target_valence": [round(profile.target_valence, 3)],
        "target_danceability": [round(profile.target_danceability, 3)],
        "target_acousticness": [round(profile.target_acousticness, 3)],
    }


def _render_profile_table(profile: UserProfile) -> None:
    st.dataframe(
        _profile_row_dict(profile), use_container_width=True, hide_index=True
    )


def _render_recommendation_card(scored, expl) -> None:
    with st.container(border=True):
        cols = st.columns([4, 1])
        cols[0].subheader(scored.song.title, anchor=False)
        cols[0].markdown(
            f"_{scored.song.artist}_  -  `{scored.song.genre}` / `{scored.song.mood}`"
        )
        cols[1].metric("score", f"{scored.score:.2f}")

        if expl.text:
            st.write(expl.text)
        else:
            st.markdown(
                f"_Mechanical reasons only "
                f"(fallback: `{expl.fallback_reason}`)._"
            )

        with st.expander("Mechanical reasons"):
            for reason in scored.reasons:
                st.markdown(f"- {reason}")


def _render_recommendations(rec: RecommendationResult, label: str) -> None:
    st.subheader(f"Top {len(rec.recommendations)} for {label}", anchor=False)
    for scored, expl in zip(rec.recommendations, rec.explanations):
        _render_recommendation_card(scored, expl)


def _render_build_debug(build: ProfileBuildResult) -> None:
    if build.extractor_warnings:
        st.warning("Extractor warnings:")
        for warning in build.extractor_warnings:
            st.markdown(f"- {warning}")

    st.markdown("**Extracted profile (before refinement):**")
    _render_profile_table(build.extracted_profile)

    if build.candidate_profile != build.extracted_profile:
        st.markdown("**Candidate profile (after critic refinement):**")
        _render_profile_table(build.candidate_profile)

    st.markdown("**Refinement history:**")
    if build.refinement_history:
        rows = [
            {
                "iter": s.iter_index,
                "verdict": s.verdict,
                "adjustments": str(s.adjustments_applied or {}),
                "reason": s.reason,
            }
            for s in build.refinement_history
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.write("(none)")

    if build.ambiguous_match:
        st.warning("Ambiguous match: critic disagreed through every iteration.")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _render_sidebar(client: LLMClient) -> bool:
    st.sidebar.markdown("## Status")
    if _is_online(client):
        st.sidebar.success("Connected to Gemini")
    else:
        st.sidebar.warning(
            "Stub mode — set `GEMINI_API_KEY` in `.env` to enable building "
            "profiles and grounded explanations."
        )
    if isinstance(client, CachedLLMClient):
        st.sidebar.markdown(
            f"**Cache:** {client.hits} hits / {client.misses} misses (this session)"
        )
    st.sidebar.divider()
    st.sidebar.markdown(
        "_The recommender is built around two pipelines: **build a profile**"
        " from natural-language inputs, then **recommend** songs against any"
        " saved profile or preset._"
    )
    st.sidebar.divider()
    return st.sidebar.checkbox("Show build debug pane", value=False, key="show_debug")


# ---------------------------------------------------------------------------
# Tab 1 — Recommend
# ---------------------------------------------------------------------------


def _render_recommend_tab(client: LLMClient, show_debug: bool) -> None:
    st.markdown(
        "Select a profile and get the top-k songs ranked for it, with "
        "explanations grounded in the catalog knowledge base."
    )

    options = _profile_picker_options()
    if not options:
        st.info("No profiles found yet. Build one in the **Build profile** tab.")
        return

    labels = [label for label, _kind, _name in options]
    # Consume any pending cross-tab pre-selection BEFORE the selectbox is
    # instantiated. Streamlit forbids mutating a widget's session_state key
    # after the widget is created, so cross-tab handoffs use a separate
    # `pending_recommend_label` slot that we transfer here.
    pending = st.session_state.pop("pending_recommend_label", None)
    if pending in labels:
        st.session_state["recommend_picker_label"] = pending
    # Reset stored selection if it points to a profile that no longer exists.
    if st.session_state.get("recommend_picker_label") not in labels:
        st.session_state["recommend_picker_label"] = labels[0]
    picked_label = st.selectbox("Profile", labels, key="recommend_picker_label")
    picked_index = labels.index(picked_label)
    _label, kind, name = options[picked_index]
    profile = _resolve_picked(kind, name)
    assert profile is not None  # picker excludes "none"

    with st.expander("Show profile fields", expanded=False):
        _render_profile_table(profile)

    cols = st.columns([1, 3])
    k = cols[0].slider("How many", min_value=1, max_value=10, value=5, key="recommend_k")
    recommend_clicked = cols[1].button("Recommend", type="primary", use_container_width=True)

    if recommend_clicked:
        with st.spinner("Ranking the catalog and writing explanations..."):
            try:
                rec_result = recommend(profile, client, k=k)
                st.session_state["last_rec"] = rec_result
                st.session_state["last_rec_label"] = picked_label
                st.session_state.pop("last_rec_error", None)
            except Exception as exc:
                st.session_state["last_rec_error"] = (
                    f"Recommend failed: {exc.__class__.__name__}: {exc}"
                )
                st.session_state.pop("last_rec", None)

    if "last_rec_error" in st.session_state:
        st.error(st.session_state["last_rec_error"])

    last_rec = st.session_state.get("last_rec")
    last_label = st.session_state.get("last_rec_label", "")
    if last_rec is not None:
        _render_recommendations(last_rec, last_label)
        if last_rec.cache_stats and show_debug:
            st.caption(
                f"recommend cache: {last_rec.cache_stats['hits']} hits, "
                f"{last_rec.cache_stats['misses']} misses"
            )


# ---------------------------------------------------------------------------
# Tab 2 — Build profile
# ---------------------------------------------------------------------------


_QUESTIONS = (
    ("activity",
     "1. What are you usually doing when you listen, or what do you want this music for?"),
    ("feeling",
     "2. How do you want this music to make you feel?"),
    ("movement",
     "3. Are you in the mood to move, sit still, or somewhere in between?"),
    ("instruments",
     "4. Are you drawn to acoustic instruments, electronic/synthesized sounds, or a mix?"),
    ("genres",
     "5. Any genres you specifically want, or any you want to avoid?"),
)


def _render_build_tab(client: LLMClient, show_debug: bool) -> None:
    st.markdown(
        "Answer any combination of the five questions plus the free-form "
        "description (at least one field must be non-blank). The system "
        "will translate your answers into a 7-dimension listener profile "
        "and ask a critic to verify faithfulness."
    )

    seed_options = _profile_picker_options(include_none=True)
    seed_labels = [label for label, _kind, _name in seed_options]
    # Same pending-slot pattern as the recommend picker: cross-tab handoffs
    # write to `pending_build_seed_label` and we transfer here, before the
    # seed selectbox is instantiated.
    pending_seed = st.session_state.pop("pending_build_seed_label", None)
    if pending_seed in seed_labels:
        st.session_state["build_seed_label"] = pending_seed
    # Reset stored seed if it points to a profile that no longer exists.
    if st.session_state.get("build_seed_label") not in seed_labels:
        st.session_state["build_seed_label"] = seed_labels[0]
    seed_label = st.selectbox(
        "Seed from an existing profile or preset (optional)",
        seed_labels,
        key="build_seed_label",
        help=(
            "When set, the existing profile is shown to the LLM as a "
            "starting point — you only need to describe what's changing."
        ),
    )
    seed_kind = next(k for label, k, _ in seed_options if label == seed_label)
    seed_name = next(n for label, _, n in seed_options if label == seed_label)
    seed_profile = _resolve_picked(seed_kind, seed_name) if seed_kind != "none" else None

    with st.form("build_form"):
        answers: dict[str, str] = {}
        for field, question in _QUESTIONS:
            answers[field] = st.text_input(question, key=f"build_q_{field}")
        description = st.text_area(
            "Or describe your mood / what you want in your own words",
            key="build_description",
            height=80,
        )

        cols = st.columns([2, 1])
        save_name = cols[0].text_input(
            "Save under name (optional)",
            key="build_save_name",
            help="Leave blank to discard the candidate after this session.",
        )
        no_overwrite = cols[1].checkbox(
            "Refuse overwrite",
            value=False,
            key="build_no_overwrite",
            help="With save: refuse if a profile of this name already exists.",
        )

        submitted = st.form_submit_button("Build profile", type="primary")

    if submitted:
        inputs = BuildInputs(
            activity=answers["activity"] or None,
            feeling=answers["feeling"] or None,
            movement=answers["movement"] or None,
            instruments=answers["instruments"] or None,
            genres=answers["genres"] or None,
            description=description or None,
        )
        if not inputs.has_minimum():
            st.error(
                "Provide at least one non-blank answer or fill the description."
            )
            return

        with st.spinner("Extracting profile and asking the critic..."):
            try:
                build_result = build_profile(inputs, client, starting_from=seed_profile)
                st.session_state["last_build"] = build_result
                st.session_state.pop("last_build_error", None)
            except EmptyBuildInputsError as exc:
                st.session_state["last_build_error"] = str(exc)
                st.session_state.pop("last_build", None)
                return
            except ProfileExtractionError as exc:
                st.session_state["last_build_error"] = (
                    f"Could not parse a profile from these inputs - try rephrasing.\n\n{exc}"
                )
                st.session_state.pop("last_build", None)
                return
            except Exception as exc:
                st.session_state["last_build_error"] = (
                    f"Build failed: {exc.__class__.__name__}: {exc}"
                )
                st.session_state.pop("last_build", None)
                return

        # Optional save
        if save_name.strip():
            try:
                save_profile(
                    save_name,
                    build_result.candidate_profile,
                    overwrite=not no_overwrite,
                )
                st.session_state["last_build_save_status"] = (
                    f"Saved as **{save_name}**."
                )
            except ProfileExistsError as exc:
                st.session_state["last_build_save_status"] = (
                    f"Not saved: {exc}. Uncheck 'Refuse overwrite' to clobber."
                )
            except Exception as exc:
                st.session_state["last_build_save_status"] = (
                    f"Save failed: {exc.__class__.__name__}: {exc}"
                )
        else:
            st.session_state.pop("last_build_save_status", None)

    if "last_build_error" in st.session_state:
        st.error(st.session_state["last_build_error"])

    last_build = st.session_state.get("last_build")
    if last_build is not None:
        st.divider()
        st.markdown("### Candidate profile")
        _render_profile_table(last_build.candidate_profile)

        save_status = st.session_state.get("last_build_save_status")
        if save_status:
            if save_status.startswith("Saved"):
                st.success(save_status)
            else:
                st.warning(save_status)

        if show_debug:
            with st.expander("Build details", expanded=False):
                _render_build_debug(last_build)
                if last_build.cache_stats:
                    st.caption(
                        f"build cache: {last_build.cache_stats['hits']} hits, "
                        f"{last_build.cache_stats['misses']} misses"
                    )


# ---------------------------------------------------------------------------
# Tab 3 — Manage profiles
# ---------------------------------------------------------------------------


def _render_edit_form(name: str, profile: UserProfile) -> None:
    """Direct-field-edit form for a saved profile (no LLM call)."""
    with st.form(f"edit_form_{name}"):
        new_genre = st.selectbox(
            "favorite_genre",
            _allowed_genres(),
            index=_allowed_genres().index(profile.favorite_genre)
            if profile.favorite_genre in _allowed_genres()
            else 0,
            key=f"edit_{name}_genre",
        )
        new_mood = st.selectbox(
            "favorite_mood",
            _allowed_moods(),
            index=_allowed_moods().index(profile.favorite_mood)
            if profile.favorite_mood in _allowed_moods()
            else 0,
            key=f"edit_{name}_mood",
        )
        new_energy = st.slider(
            "target_energy", 0.0, 1.0, value=float(profile.target_energy), step=0.05,
            key=f"edit_{name}_energy",
        )
        new_tempo = st.slider(
            "target_tempo_bpm", 40.0, 220.0, value=float(profile.target_tempo_bpm), step=1.0,
            key=f"edit_{name}_tempo",
        )
        new_valence = st.slider(
            "target_valence", 0.0, 1.0, value=float(profile.target_valence), step=0.05,
            key=f"edit_{name}_valence",
        )
        new_dance = st.slider(
            "target_danceability", 0.0, 1.0, value=float(profile.target_danceability), step=0.05,
            key=f"edit_{name}_dance",
        )
        new_acoustic = st.slider(
            "target_acousticness", 0.0, 1.0, value=float(profile.target_acousticness), step=0.05,
            key=f"edit_{name}_acoustic",
        )

        cols = st.columns(2)
        save_clicked = cols[0].form_submit_button("Save changes", type="primary")
        cancel_clicked = cols[1].form_submit_button("Cancel")

    if save_clicked:
        try:
            edit_profile_fields(
                name,
                favorite_genre=new_genre,
                favorite_mood=new_mood,
                target_energy=new_energy,
                target_tempo_bpm=new_tempo,
                target_valence=new_valence,
                target_danceability=new_dance,
                target_acousticness=new_acoustic,
            )
            st.session_state.pop("editing_profile", None)
            st.success(f"Saved changes to **{name}**.")
            st.rerun()
        except (ValueError, ProfileNotFoundError) as exc:
            st.error(f"Edit failed: {exc}")
    elif cancel_clicked:
        st.session_state.pop("editing_profile", None)
        st.rerun()


def _render_saved_profile_card(name: str, created) -> None:
    with st.container(border=True):
        cols = st.columns([3, 1])
        cols[0].subheader(name, anchor=False)
        cols[0].caption(f"created {created.isoformat()}")

        try:
            profile = load_profile(name)
        except ProfileNotFoundError as exc:
            cols[0].error(f"Could not load: {exc}")
            return

        with st.expander("Show fields", expanded=False):
            _render_profile_table(profile)

        editing = st.session_state.get("editing_profile") == name
        pending_delete = st.session_state.get("pending_delete") == name

        action_cols = st.columns(3)
        if action_cols[0].button(
            "Edit" if not editing else "Close edit",
            key=f"edit_btn_{name}",
            use_container_width=True,
        ):
            if editing:
                st.session_state.pop("editing_profile", None)
            else:
                st.session_state["editing_profile"] = name
                st.session_state.pop("pending_delete", None)
            st.rerun()

        if action_cols[1].button(
            "Delete", key=f"delete_btn_{name}", use_container_width=True
        ):
            st.session_state["pending_delete"] = name
            st.session_state.pop("editing_profile", None)
            st.rerun()

        if action_cols[2].button(
            "Use for recommend", key=f"use_btn_{name}", use_container_width=True
        ):
            st.session_state["pending_recommend_label"] = f"Saved: {name}"
            st.session_state["pending_recommend_announce"] = name
            st.rerun()

        if editing:
            st.divider()
            _render_edit_form(name, profile)

        if pending_delete:
            st.divider()
            st.warning(f"Delete profile **{name}**? This cannot be undone.")
            confirm_cols = st.columns(2)
            if confirm_cols[0].button(
                "Confirm delete", key=f"confirm_delete_{name}", type="primary"
            ):
                try:
                    delete_profile(name)
                    st.session_state.pop("pending_delete", None)
                    st.success(f"Deleted **{name}**.")
                    st.rerun()
                except ProfileNotFoundError as exc:
                    st.error(f"Delete failed: {exc}")
            if confirm_cols[1].button("Cancel", key=f"cancel_delete_{name}"):
                st.session_state.pop("pending_delete", None)
                st.rerun()


def _render_preset_card(slug: str) -> None:
    profile = PRESET_PROFILES[slug]
    display = PRESET_DISPLAY_NAMES.get(slug, slug)

    with st.container(border=True):
        cols = st.columns([3, 1])
        cols[0].subheader(display, anchor=False)
        cols[0].caption(f"preset slug: `{slug}` (read-only)")

        with st.expander("Show fields", expanded=False):
            _render_profile_table(profile)

        action_cols = st.columns(2)
        if action_cols[0].button(
            "Use for recommend", key=f"preset_use_{slug}", use_container_width=True
        ):
            st.session_state["pending_recommend_label"] = f"Preset: {display}"
            st.session_state["pending_recommend_announce"] = display
            st.rerun()

        if action_cols[1].button(
            "Use as build seed", key=f"preset_seed_{slug}", use_container_width=True
        ):
            st.session_state["pending_build_seed_label"] = f"Preset: {display}"
            st.session_state["pending_build_seed_announce"] = display
            st.rerun()


def _render_manage_tab() -> None:
    st.markdown(
        "Saved profiles persist across sessions in `applied-ai-systems/profiles/`. "
        "Presets are immutable — derive new profiles from them via *Use as build seed*."
    )

    # Surface confirmation messages from cross-tab pre-selection (the
    # actual selection happens via pending_* slots on the receiving tab).
    rec_announce = st.session_state.pop("pending_recommend_announce", None)
    if rec_announce:
        st.success(
            f"Selected **{rec_announce}** for recommend. Switch to the "
            f"**Recommend** tab and click *Recommend*."
        )
    seed_announce = st.session_state.pop("pending_build_seed_announce", None)
    if seed_announce:
        st.success(
            f"Set **{seed_announce}** as the seed for the next build. "
            f"Switch to the **Build profile** tab to fill in your changes."
        )

    saved = list_profiles()
    st.markdown("### Saved profiles")
    if not saved:
        st.info("No saved profiles yet. Build one in the **Build profile** tab.")
    else:
        for name, created in saved:
            _render_saved_profile_card(name, created)

    st.divider()
    st.markdown("### Presets")
    for slug in PRESET_DISPLAY_NAMES:
        _render_preset_card(slug)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Music Recommender", layout="wide")

    client = _get_client()
    show_debug = _render_sidebar(client)

    st.title("Music Recommender")
    st.caption(
        "AI-assisted profile creation + deterministic content-based scoring + "
        "RAG-grounded explanations against a 30-track catalog."
    )

    rec_tab, build_tab, manage_tab = st.tabs(
        ["Recommend", "Build profile", "Manage profiles"]
    )

    with rec_tab:
        _render_recommend_tab(client, show_debug)
    with build_tab:
        _render_build_tab(client, show_debug)
    with manage_tab:
        _render_manage_tab()


main()
