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

import dataclasses
import functools
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from src.agents.profile_extractor import ProfileExtractionError
from src.eval.assertions import assert_build_neighborhood, assert_recommend_structural
from src.eval.cases import BUILD_CASES, RECOMMEND_CASES, BuildCase
from src.eval.harness import (
    EVAL_RESULTS_DIR,
    BuildEvalResult,
    RecommendEvalResult,
    _audit_uncached_calls,
    run_build_eval,
    run_recommend_eval,
)
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
    avoid_display = ", ".join(profile.avoid_genres) if profile.avoid_genres else "(none)"
    return {
        "favorite_genre": [profile.favorite_genre],
        "favorite_mood": [profile.favorite_mood],
        "target_energy": [round(profile.target_energy, 3)],
        "target_tempo_bpm": [round(profile.target_tempo_bpm, 1)],
        "target_valence": [round(profile.target_valence, 3)],
        "target_danceability": [round(profile.target_danceability, 3)],
        "target_acousticness": [round(profile.target_acousticness, 3)],
        "avoid_genres": [avoid_display],
    }


def _render_profile_table(profile: UserProfile) -> None:
    st.dataframe(
        _profile_row_dict(profile), use_container_width=True, hide_index=True
    )


def _render_recommendation_card(scored, expl, ctx) -> None:
    with st.container(border=True):
        cols = st.columns([4, 1])
        cols[0].subheader(scored.song.title, anchor=False)
        cols[0].markdown(
            f"_{scored.song.artist}_  -  `{scored.song.genre}` / `{scored.song.mood}`"
        )
        cols[1].metric("score", f"{scored.score:.2f}")

        # Quote-check badge — shows whether the explanation is grounded.
        if expl.text:
            quote_count = len(expl.cited_snippets)
            st.success(
                f"✓ Fact-checked · {quote_count} quote"
                f"{'s' if quote_count != 1 else ''} verified against song notes"
            )
            st.write(expl.text)
        else:
            st.warning(
                "Couldn't verify quotes — showing the score breakdown "
                "below instead."
            )

        with st.expander("Why this song scored where it did"):
            for reason in scored.reasons:
                st.markdown(f"- {reason}")

        if expl.cited_snippets:
            with st.expander("Verified quotes"):
                for snippet in expl.cited_snippets:
                    st.markdown(f"> {snippet}")

        with st.expander("Song notes SongFinder read"):
            st.caption(
                "SongFinder pulled three notes for this pick — one about "
                "the genre, one about the mood, and one about the song "
                "itself. Quotes in the explanation above are checked "
                "against these."
            )
            st.markdown(
                f"**About the genre** · `{ctx.genre.path.name}`"
            )
            st.text(ctx.genre.body[:600] + ("..." if len(ctx.genre.body) > 600 else ""))
            st.markdown(
                f"**About the mood** · `{ctx.mood.path.name}`"
            )
            st.text(ctx.mood.body[:600] + ("..." if len(ctx.mood.body) > 600 else ""))
            st.markdown(
                f"**About the song** · `{ctx.song.path.name}`"
            )
            st.text(ctx.song.body[:600] + ("..." if len(ctx.song.body) > 600 else ""))


def _render_recommendations(rec: RecommendationResult, label: str) -> None:
    grounded = sum(1 for e in rec.explanations if e.text)
    fallback = len(rec.explanations) - grounded
    st.subheader(f"Top {len(rec.recommendations)} for {label}", anchor=False)
    cols = st.columns(3)
    cols[0].metric("Songs picked", len(rec.recommendations))
    cols[1].metric("Quote-checked", grounded)
    cols[2].metric("Score-only", fallback)
    for scored, expl, ctx in zip(
        rec.recommendations, rec.explanations, rec.retrieved_contexts
    ):
        _render_recommendation_card(scored, expl, ctx)


def _render_build_debug(build: ProfileBuildResult) -> None:
    if build.extractor_warnings:
        st.warning("Heads up:")
        for warning in build.extractor_warnings:
            st.markdown(f"- {warning}")

    st.markdown("**First read of your answers:**")
    _render_profile_table(build.extracted_profile)

    if build.candidate_profile != build.extracted_profile:
        st.markdown("**Final profile (after self-check adjustments):**")
        _render_profile_table(build.candidate_profile)

    st.markdown("**Self-check steps:**")
    if build.refinement_history:
        rows = [
            {
                "round": s.iter_index + 1,
                "verdict": "looks good" if s.verdict == "ok" else "needs adjustment",
                "adjustments": str(s.adjustments_applied or {}),
                "why": s.reason,
            }
            for s in build.refinement_history
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.write("(none)")

    if build.ambiguous_match:
        st.warning(
            "Heads up: the self-check wasn't fully convinced about this "
            "profile. Saved the best read but flagged it for your review."
        )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _render_sidebar(client: LLMClient) -> bool:
    st.sidebar.markdown("## Status")
    if _is_online(client):
        st.sidebar.success("Connected to Gemini")
    else:
        st.sidebar.warning(
            "Demo mode — set `GEMINI_API_KEY` in `.env` to enable building "
            "Vibe Profiles and grounded song picks."
        )
    if isinstance(client, CachedLLMClient):
        st.sidebar.markdown(
            f"**API calls this session:** "
            f"{client.hits} cached / {client.misses} new"
        )
    st.sidebar.divider()
    st.sidebar.markdown(
        "### How this works\n"
        "**1. Tell SongFinder your vibe.** Answer a few questions or just "
        "describe what you want. SongFinder builds a saved Vibe Profile "
        "from your answers and double-checks itself.\n\n"
        "**2. SongFinder picks songs.** Pick any saved Vibe Profile or "
        "starter preset and get recommendations. Each pick comes with a "
        "short explanation grounded in real song notes.\n\n"
        "**Reliability tab** tests both halves to make sure they actually "
        "work."
    )
    st.sidebar.divider()
    return st.sidebar.checkbox(
        "Show details",
        value=True,
        key="show_debug",
        help=(
            "Show how SongFinder built each profile or pick — its "
            "intermediate steps, the song notes it used, and quote checks."
        ),
    )


# ---------------------------------------------------------------------------
# Tab 1 — Recommend
# ---------------------------------------------------------------------------


def _render_recommend_tab(client: LLMClient, show_debug: bool) -> None:
    st.markdown(
        "Pick a Vibe Profile and let SongFinder pick songs for you. "
        "Each pick comes with the why behind the choice."
    )

    options = _profile_picker_options()
    if not options:
        st.info(
            "No Vibe Profiles found yet. Build one in the "
            "**Build Vibe Profile** tab."
        )
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
    picked_label = st.selectbox(
        "Vibe Profile", labels, key="recommend_picker_label"
    )
    picked_index = labels.index(picked_label)
    _label, kind, name = options[picked_index]
    profile = _resolve_picked(kind, name)
    assert profile is not None  # picker excludes "none"

    with st.expander("Show profile fields", expanded=False):
        _render_profile_table(profile)

    cols = st.columns([1, 3])
    k = cols[0].slider(
        "How many picks", min_value=1, max_value=10, value=5,
        key="recommend_k"
    )
    recommend_clicked = cols[1].button(
        "Find songs", type="primary", use_container_width=True
    )

    if recommend_clicked:
        with st.status(
            "SongFinder is picking songs...", expanded=True
        ) as status:
            status.write(
                "**Scoring the catalog** — checking all 30 tracks against "
                "your Vibe Profile..."
            )
            try:
                rec_result = recommend(profile, client, k=k)
            except Exception as exc:
                status.update(label="Couldn't get recommendations", state="error")
                st.session_state["last_rec_error"] = (
                    f"Something went wrong: {exc.__class__.__name__}: {exc}"
                )
                st.session_state.pop("last_rec", None)
                rec_result = None

            if rec_result is not None:
                top = rec_result.recommendations[0]
                status.write(
                    f"✓ Top {k} picked. "
                    f"#1 is **{top.song.title}** by *{top.song.artist}* "
                    f"(score {top.score:.2f})."
                )
                doc_count = len(rec_result.retrieved_contexts) * 3
                status.write(
                    f"**Pulling song notes** — {doc_count} short notes "
                    f"({k} × genre + mood + per-song)."
                )
                grounded = sum(1 for e in rec_result.explanations if e.text)
                fallback = len(rec_result.explanations) - grounded
                total_citations = sum(
                    len(e.cited_snippets) for e in rec_result.explanations
                )
                status.write(
                    f"**Writing the why** — one call to Gemini that "
                    f"writes a short explanation for each pick."
                )
                status.write(
                    f"**Quote-checking** — verified {total_citations} "
                    f"quote{'s' if total_citations != 1 else ''} against "
                    f"the song notes. {grounded}/{k} explanations passed; "
                    f"{fallback}/{k} fell back to score breakdown only."
                )
                status.update(label="Done — see your picks below", state="complete")
                st.session_state["last_rec"] = rec_result
                st.session_state["last_rec_label"] = picked_label
                st.session_state.pop("last_rec_error", None)

    if "last_rec_error" in st.session_state:
        st.error(st.session_state["last_rec_error"])

    last_rec = st.session_state.get("last_rec")
    last_label = st.session_state.get("last_rec_label", "")
    if last_rec is not None:
        _render_recommendations(last_rec, last_label)
        if last_rec.cache_stats and show_debug:
            st.caption(
                f"API calls: {last_rec.cache_stats['hits']} cached, "
                f"{last_rec.cache_stats['misses']} new"
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
        "Tell SongFinder about your vibe — answer any of the questions "
        "below or just describe what you want in your own words. "
        "SongFinder builds a Vibe Profile from your answers and "
        "double-checks itself before saving."
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
        "Start from an existing profile or preset (optional)",
        seed_labels,
        key="build_seed_label",
        help=(
            "Pick one and SongFinder uses it as a starting point — you "
            "only need to describe what you want to change."
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
            "Or just describe your vibe in your own words",
            key="build_description",
            height=80,
        )

        cols = st.columns([2, 1])
        save_name = cols[0].text_input(
            "Save as (optional)",
            key="build_save_name",
            help=(
                "Give it a name to save it. Leave blank to just preview "
                "without saving."
            ),
        )
        no_overwrite = cols[1].checkbox(
            "Don't overwrite",
            value=False,
            key="build_no_overwrite",
            help=(
                "If a Vibe Profile with this name already exists, "
                "refuse to save instead of replacing it."
            ),
        )

        submitted = st.form_submit_button(
            "Build my Vibe Profile", type="primary"
        )

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

        with st.status(
            "SongFinder is building your Vibe Profile...", expanded=True
        ) as status:
            status.write(
                "**Reading your answers** — making sure there's something "
                "to work with. ✓"
            )
            status.write(
                "**Pulling out a profile** — Gemini reads what you wrote "
                "and turns it into a 7-feature Vibe Profile."
            )
            try:
                build_result = build_profile(
                    inputs, client, starting_from=seed_profile
                )
            except EmptyBuildInputsError as exc:
                status.update(label="Need at least one answer", state="error")
                st.session_state["last_build_error"] = str(exc)
                st.session_state.pop("last_build", None)
                return
            except ProfileExtractionError as exc:
                status.update(
                    label="Couldn't make sense of these answers",
                    state="error",
                )
                st.session_state["last_build_error"] = (
                    f"Couldn't pull a profile from these inputs — "
                    f"try rephrasing.\n\n{exc}"
                )
                st.session_state.pop("last_build", None)
                return
            except Exception as exc:
                status.update(label="Something went wrong", state="error")
                st.session_state["last_build_error"] = (
                    f"Something went wrong: {exc.__class__.__name__}: {exc}"
                )
                st.session_state.pop("last_build", None)
                return

            extracted = build_result.extracted_profile
            status.write(
                f"✓ First read: `{extracted.favorite_genre}` / "
                f"`{extracted.favorite_mood}`, "
                f"energy={extracted.target_energy:.2f}, "
                f"tempo={extracted.target_tempo_bpm:.0f} BPM."
            )
            if build_result.extractor_warnings:
                status.write(
                    f"⚠ Heads up: "
                    f"{', '.join(build_result.extractor_warnings)}"
                )

            for step in build_result.refinement_history:
                if step.verdict == "ok":
                    status.write(
                        f"**Double-check {step.iter_index + 1}** — "
                        f"profile matches what you said. ✓"
                    )
                else:
                    adj = step.adjustments_applied or {}
                    status.write(
                        f"**Double-check {step.iter_index + 1}** — "
                        f"adjusting: "
                        f"{', '.join(f'{k}={v}' for k, v in adj.items()) or '(none)'}"
                    )
            if build_result.ambiguous_match:
                status.write(
                    "⚠ The double-check wasn't fully convinced. Keeping "
                    "the best read but flagging it for your review."
                )

            status.update(label="Vibe Profile ready", state="complete")
            st.session_state["last_build"] = build_result
            st.session_state.pop("last_build_error", None)

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
                    f"Not saved: {exc}. Uncheck 'Don't overwrite' to replace it."
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

        with st.expander(
            "How SongFinder built this profile",
            expanded=show_debug,
        ):
            _render_build_debug(last_build)
            if last_build.cache_stats:
                st.caption(
                    f"API calls: {last_build.cache_stats['hits']} cached, "
                    f"{last_build.cache_stats['misses']} new"
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
        new_avoid = st.multiselect(
            "avoid_genres",
            options=_allowed_genres(),
            default=[g for g in profile.avoid_genres if g in _allowed_genres()],
            key=f"edit_{name}_avoid",
            help="SongFinder will hard-skip songs in these genres.",
        )

        cols = st.columns(2)
        save_clicked = cols[0].form_submit_button("Save changes", type="primary")
        cancel_clicked = cols[1].form_submit_button("Cancel")

    if save_clicked:
        if new_genre in new_avoid:
            st.error("Cannot avoid your favorite genre. Remove it from one or the other.")
            return
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
                avoid_genres=list(new_avoid),
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
            "Find songs with this", key=f"use_btn_{name}", use_container_width=True
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
            "Find songs with this", key=f"preset_use_{slug}", use_container_width=True
        ):
            st.session_state["pending_recommend_label"] = f"Preset: {display}"
            st.session_state["pending_recommend_announce"] = display
            st.rerun()

        if action_cols[1].button(
            "Use as starting point", key=f"preset_seed_{slug}", use_container_width=True
        ):
            st.session_state["pending_build_seed_label"] = f"Preset: {display}"
            st.session_state["pending_build_seed_announce"] = display
            st.rerun()


def _render_manage_tab() -> None:
    st.markdown(
        "Your saved Vibe Profiles live here. The starter presets are "
        "read-only — use one as a starting point if you want to build a "
        "tweaked version."
    )

    # Surface confirmation messages from cross-tab pre-selection (the
    # actual selection happens via pending_* slots on the receiving tab).
    rec_announce = st.session_state.pop("pending_recommend_announce", None)
    if rec_announce:
        st.success(
            f"Picked **{rec_announce}** for song recommendations. "
            f"Head to the **Find Songs** tab and hit *Recommend*."
        )
    seed_announce = st.session_state.pop("pending_build_seed_announce", None)
    if seed_announce:
        st.success(
            f"Using **{seed_announce}** as the starting point for your "
            f"next build. Head to the **Build Vibe Profile** tab and "
            f"describe what you want to change."
        )

    saved = list_profiles()
    st.markdown("### Your saved Vibe Profiles")
    if not saved:
        st.info(
            "No saved Vibe Profiles yet. Build one in the "
            "**Build Vibe Profile** tab."
        )
    else:
        for name, created in saved:
            _render_saved_profile_card(name, created)

    st.divider()
    st.markdown("### Starter presets (read-only)")
    for slug in PRESET_DISPLAY_NAMES:
        _render_preset_card(slug)


# ---------------------------------------------------------------------------
# Tab 4 — Eval & Tests
# ---------------------------------------------------------------------------


def _build_eval_to_dict(b: BuildEvalResult) -> dict[str, Any]:
    """Convert a live BuildEvalResult to the same JSON shape the harness
    writes to disk, so one renderer handles stored + live results.
    """
    return {
        "case": {
            "name": b.case.name,
            "category": b.case.category,
            "expected_profile": dataclasses.asdict(b.case.expected_profile),
            "tempo_tolerance_bpm": b.case.tempo_tolerance_bpm,
            "numeric_tolerance": b.case.numeric_tolerance,
            "inputs": dataclasses.asdict(b.case.inputs),
        },
        "result": {
            "inputs": dataclasses.asdict(b.result.inputs),
            "extracted_profile": dataclasses.asdict(b.result.extracted_profile),
            "candidate_profile": dataclasses.asdict(b.result.candidate_profile),
            "refinement_history": [
                {
                    "iter_index": s.iter_index,
                    "verdict": s.verdict,
                    "candidate_after_iter": s.candidate_after_iter,
                    "adjustments_applied": s.adjustments_applied,
                    "reason": s.reason,
                }
                for s in b.result.refinement_history
            ],
            "ambiguous_match": b.result.ambiguous_match,
            "extractor_warnings": list(b.result.extractor_warnings),
            "cache_stats": b.result.cache_stats,
        },
        "passed": b.passed,
        "failures": list(b.failures),
    }


def _recommend_eval_to_dict(r: RecommendEvalResult) -> dict[str, Any]:
    return {
        "preset_name": r.preset_name,
        "result": {
            "profile": dataclasses.asdict(r.result.profile),
            "recommendations": [
                {
                    "song_id": rec.song.id,
                    "title": rec.song.title,
                    "artist": rec.song.artist,
                    "genre": rec.song.genre,
                    "mood": rec.song.mood,
                    "score": rec.score,
                    "reasons": list(rec.reasons),
                }
                for rec in r.result.recommendations
            ],
            "explanations": [
                {
                    "song_id": e.song_id,
                    "text": e.text,
                    "cited_snippets": list(e.cited_snippets),
                    "fallback_reason": e.fallback_reason,
                }
                for e in r.result.explanations
            ],
            "cache_stats": r.result.cache_stats,
        },
        "structural_pass": r.structural_pass,
        "structural_failures": list(r.structural_failures),
        "self_critique_score": r.self_critique_score,
        "self_critique_pass": r.self_critique_pass,
        "self_critique_reason": r.self_critique_reason,
    }


def _render_build_scorecard_row(b: dict[str, Any]) -> None:
    """Per-row expander content for one BUILD case in a scorecard."""
    case = b["case"]
    result = b["result"]

    inputs = case["inputs"]
    non_blank = {k: v for k, v in inputs.items() if v}
    if non_blank:
        st.markdown("**The listener's answers:**")
        st.json(non_blank)

    expected = case.get("expected_profile")
    if expected:
        st.markdown("**What we expected SongFinder to produce:**")
        st.dataframe(
            {k: [v] for k, v in expected.items()},
            hide_index=True,
            use_container_width=True,
        )

    cols = st.columns(2)
    cols[0].markdown("**SongFinder's first read:**")
    cols[0].dataframe(
        {k: [v] for k, v in result["extracted_profile"].items()},
        hide_index=True,
        use_container_width=True,
    )
    cols[1].markdown("**Final profile (after self-checks):**")
    cols[1].dataframe(
        {k: [v] for k, v in result["candidate_profile"].items()},
        hide_index=True,
        use_container_width=True,
    )

    history = result.get("refinement_history", [])
    if history:
        st.markdown("**Self-check rounds:**")
        rows = [
            {
                "round": s["iter_index"] + 1,
                "verdict":
                    "looks good" if s["verdict"] == "ok" else "needs adjustment",
                "adjustments": str(s.get("adjustments_applied") or {}),
                "why": s.get("reason", "")[:120],
            }
            for s in history
        ]
        st.dataframe(rows, hide_index=True, use_container_width=True)

    if result.get("ambiguous_match"):
        st.warning(
            "The self-check wasn't fully convinced — flagged for review."
        )
    warnings = result.get("extractor_warnings", [])
    if warnings:
        st.warning("Heads up:")
        for w in warnings:
            st.markdown(f"- {w}")

    failures = b.get("failures", [])
    if failures:
        st.error("How the result missed the expected profile:")
        for f in failures:
            st.markdown(f"- {f}")


def _render_recommend_scorecard_row(r: dict[str, Any]) -> None:
    """Per-row expander content for one RECOMMEND case in a scorecard."""
    profile = r["result"]["profile"]
    st.markdown(
        f"**Preset profile:** `{profile['favorite_genre']}` / "
        f"`{profile['favorite_mood']}`, "
        f"energy={profile['target_energy']:.2f}, "
        f"tempo={profile['target_tempo_bpm']:.0f} BPM"
    )

    cols = st.columns(2)
    cols[0].metric(
        "Self-review score", f"{r['self_critique_score']:.2f}"
    )
    cols[1].metric(
        "Self-review verdict",
        "passed" if r["self_critique_pass"] else "failed",
    )
    if r.get("self_critique_reason"):
        st.markdown(
            f"**Why Gemini gave this score:** {r['self_critique_reason']}"
        )

    recommendations = r["result"]["recommendations"]
    explanations = {e["song_id"]: e for e in r["result"]["explanations"]}
    st.markdown("**Top 5 picks:**")
    for rec in recommendations:
        expl = explanations.get(rec["song_id"], {})
        text = expl.get("text")
        cited = expl.get("cited_snippets") or []
        with st.container(border=True):
            cols = st.columns([4, 1])
            cols[0].markdown(
                f"**{rec['title']}** — *{rec['artist']}* · "
                f"`{rec['genre']}` / `{rec['mood']}`"
            )
            cols[1].markdown(f"score **{rec['score']:.2f}**")
            if text:
                st.caption(
                    f"✓ Fact-checked · {len(cited)} quote"
                    f"{'s' if len(cited) != 1 else ''} verified"
                )
                st.write(text)
            else:
                st.caption(
                    "Couldn't verify quotes — score breakdown only."
                )

    failures = r.get("structural_failures", [])
    if failures:
        st.error("How the picks missed the structural check:")
        for f in failures:
            st.markdown(f"- {f}")


def _render_eval_scorecards(payload: dict[str, Any]) -> None:
    """Render both scorecards from a JSON-shape payload (stored or live)."""
    timestamp = payload.get("timestamp")

    # Backward compat: artifacts from before the Step 5 pipeline split used
    # a single "results" key. The current harness writes "build_results" +
    # "recommend_results". Detect and flag clearly.
    if (
        "results" in payload
        and "build_results" not in payload
        and "recommend_results" not in payload
    ):
        st.warning(
            "This past run is from an older version of SongFinder and "
            "doesn't split builder tests and recommender tests "
            "separately. Use **Run a fresh test** below to produce "
            "results in the current format."
        )
        if timestamp:
            st.caption(f"Legacy run timestamp: `{timestamp}`")
        return

    build_results = payload.get("build_results", [])
    recommend_results = payload.get("recommend_results", [])

    build_pass = sum(1 for b in build_results if b.get("passed"))
    rec_struct_pass = sum(
        1 for r in recommend_results if r.get("structural_pass")
    )
    rec_crit_pass = sum(
        1 for r in recommend_results if r.get("self_critique_pass")
    )

    if timestamp:
        st.markdown(f"**When this ran:** `{timestamp}`")
    cols = st.columns(3)
    cols[0].metric(
        "Profiles built correctly",
        f"{build_pass}/{len(build_results)}",
    )
    cols[1].metric(
        "Picks pass structure check",
        f"{rec_struct_pass}/{len(recommend_results)}",
    )
    cols[2].metric(
        "Picks pass self-review",
        f"{rec_crit_pass}/{len(recommend_results)}",
    )

    st.markdown("##### Vibe Profile Builder results")
    if build_results:
        build_rows = [
            {
                "listener": b["case"]["name"],
                "type": b["case"]["category"],
                "result": "passed" if b["passed"] else "failed",
                "self-checks": len(b["result"].get("refinement_history", [])),
                "flagged": (
                    "yes" if b["result"].get("ambiguous_match") else "no"
                ),
                "warnings": len(b["result"].get("extractor_warnings", [])),
                "expected genre": (
                    b["case"].get("expected_profile", {}).get(
                        "favorite_genre", ""
                    )
                ),
                "expected mood": (
                    b["case"].get("expected_profile", {}).get(
                        "favorite_mood", ""
                    )
                ),
            }
            for b in build_results
        ]
        st.dataframe(
            build_rows, hide_index=True, use_container_width=True
        )
        for b in build_results:
            label = (
                f"{'✓' if b['passed'] else '✗'} {b['case']['name']} "
                f"({b['case']['category']})"
            )
            with st.expander(label, expanded=False):
                _render_build_scorecard_row(b)
    else:
        st.info("No Vibe Profile Builder results in this run.")

    st.markdown("##### SongFinder results")
    if recommend_results:
        rec_rows = [
            {
                "preset": r["preset_name"],
                "structure check":
                    "passed" if r["structural_pass"] else "failed",
                "self-review score":
                    f"{r['self_critique_score']:.2f}",
                "self-review":
                    "passed" if r["self_critique_pass"] else "failed",
            }
            for r in recommend_results
        ]
        st.dataframe(
            rec_rows, hide_index=True, use_container_width=True
        )
        for r in recommend_results:
            label = (
                f"{'✓' if r['structural_pass'] else '✗'} {r['preset_name']} "
                f"· self-review {r['self_critique_score']:.2f}"
            )
            with st.expander(label, expanded=False):
                _render_recommend_scorecard_row(r)
    else:
        st.info("No SongFinder results in this run.")


def _render_eval_tab(client: LLMClient, show_debug: bool) -> None:
    st.markdown(
        "This page shows whether SongFinder is actually working. Two test "
        "suites run the live system end-to-end:"
    )
    st.markdown(
        "- **Vibe Profile Builder tests** — 5 fresh listener personas. "
        "Each one has a hand-written ground-truth profile, and the test "
        "is whether SongFinder can build a similar profile from the "
        "listener's natural-language answers.\n"
        "- **SongFinder tests** — 5 starter presets. For each one, the "
        "test checks both a hard-coded structural rule (e.g. \"top 5 "
        "should contain at least 1 rock track\") *and* asks Gemini to "
        "rate its own picks against the profile."
    )
    st.markdown(
        "Below you can preview cost, view past test runs, run a fresh "
        "test, or run the fast offline code tests."
    )

    # ----- Section 1: Cost audit -----
    st.divider()
    st.markdown("### Preview the cost first")
    st.caption(
        "Estimates how many API calls a fresh test run would make. "
        "This check itself costs nothing."
    )
    if st.button("Preview cost", key="eval_cost_btn"):
        misses = _audit_uncached_calls(
            BUILD_CASES, RECOMMEND_CASES, GeminiClient.MODEL
        )
        if misses == 0:
            st.success(
                "All test calls are already cached — a fresh run would "
                "make 0 API calls."
            )
        else:
            st.info(
                f"A fresh run would make about {misses} API calls "
                f"(upper bound). Use the live runner below to actually run."
            )

    # ----- Section 2: Stored results viewer -----
    st.divider()
    st.markdown("### Past test runs")
    artifacts = sorted(
        EVAL_RESULTS_DIR.glob("*.json"), reverse=True
    ) if EVAL_RESULTS_DIR.exists() else []

    if not artifacts:
        st.info(
            "No past test runs saved yet. Use the live runner below to "
            "create one."
        )
    else:
        cols = st.columns([4, 1])
        choice = cols[0].selectbox(
            "Pick a past run",
            artifacts,
            format_func=lambda p: p.stem,
            key="eval_artifact_pick",
        )
        if cols[1].button(
            "Load", key="eval_artifact_load", use_container_width=True
        ):
            try:
                payload = json.loads(choice.read_text(encoding="utf-8"))
                st.session_state["eval_loaded_payload"] = payload
                st.session_state["eval_loaded_path"] = choice.name
            except Exception as exc:
                st.error(f"Couldn't load: {exc.__class__.__name__}: {exc}")

        loaded = st.session_state.get("eval_loaded_payload")
        if loaded is not None:
            st.caption(
                f"Showing: `{st.session_state.get('eval_loaded_path', '')}`"
            )
            _render_eval_scorecards(loaded)

    # ----- Section 3: Live runner -----
    st.divider()
    st.markdown("### Run a fresh test")
    if not _is_online(client):
        st.warning(
            "Demo mode — set `GEMINI_API_KEY` in `.env` to run a fresh "
            "test. The cost preview and past runs above work without "
            "an API key."
        )
        st.button(
            "Run fresh test", type="primary", disabled=True,
            key="eval_live_run_disabled",
        )
    else:
        confirmed = st.checkbox(
            "Yes, run real API calls (uses Gemini quota)",
            value=False,
            key="eval_live_confirm",
        )
        run_clicked = st.button(
            "Run fresh test",
            type="primary",
            disabled=not confirmed,
            key="eval_live_run",
        )
        if run_clicked:
            with st.status(
                "Running tests against the live system...", expanded=True
            ) as status:
                status.write(
                    "**Vibe Profile Builder tests** — feeding 5 listener "
                    "personas through SongFinder's profile builder..."
                )
                try:
                    build_results = run_build_eval(client)
                except Exception as exc:
                    status.update(
                        label="Profile builder tests failed", state="error"
                    )
                    st.error(
                        f"Something went wrong: "
                        f"{exc.__class__.__name__}: {exc}"
                    )
                    return
                build_pass = sum(1 for b in build_results if b.passed)
                status.write(
                    f"✓ Profile builder tests done — {build_pass}/"
                    f"{len(build_results)} produced a profile in the "
                    f"expected neighborhood."
                )

                status.write(
                    "**SongFinder tests** — running each of the 5 starter "
                    "presets and checking the picks..."
                )
                try:
                    rec_results = run_recommend_eval(client)
                except Exception as exc:
                    status.update(
                        label="SongFinder tests failed", state="error"
                    )
                    st.error(
                        f"Something went wrong: "
                        f"{exc.__class__.__name__}: {exc}"
                    )
                    return
                struct_pass = sum(
                    1 for r in rec_results if r.structural_pass
                )
                crit_pass = sum(
                    1 for r in rec_results if r.self_critique_pass
                )
                status.write(
                    f"✓ SongFinder tests done — "
                    f"{struct_pass}/{len(rec_results)} passed the "
                    f"structural check, {crit_pass}/{len(rec_results)} "
                    f"passed the self-review."
                )

                status.write("**Building results display...**")
                payload = {
                    "timestamp": "live (this session)",
                    "build_results": [
                        _build_eval_to_dict(b) for b in build_results
                    ],
                    "recommend_results": [
                        _recommend_eval_to_dict(r) for r in rec_results
                    ],
                }
                st.session_state["eval_live_payload"] = payload
                status.update(label="Tests complete", state="complete")

        live_payload = st.session_state.get("eval_live_payload")
        if live_payload is not None:
            st.markdown("#### Live test results (this session)")
            _render_eval_scorecards(live_payload)

    # ----- Section 4: Unit tests -----
    st.divider()
    st.markdown("### Code health check")
    st.caption(
        "65 fast offline tests covering all of SongFinder's parts. No API "
        "calls — uses fake LLM responses. Takes about a second."
    )
    if st.button("Run code tests", key="eval_pytest_btn"):
        repo_root = Path(__file__).resolve().parent
        with st.spinner("Running 65 tests..."):
            try:
                proc = subprocess.run(
                    [".venv/bin/pytest", "-q", "--no-header"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=repo_root,
                )
            except FileNotFoundError:
                st.error(
                    "Couldn't find the test runner at `.venv/bin/pytest`. "
                    "You can run `pytest -q` from the project folder "
                    "manually instead."
                )
                proc = None
            except subprocess.TimeoutExpired:
                st.error("Tests took longer than 120 seconds — gave up.")
                proc = None

        if proc is not None:
            output = (proc.stdout or "") + (proc.stderr or "")
            if proc.returncode == 0:
                st.success("✓ All 65 tests passed")
            else:
                st.error(f"Tests failed (exit code {proc.returncode})")
            st.code(output or "(no output captured)", language="text")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Music Recommender", layout="wide")

    client = _get_client()
    show_debug = _render_sidebar(client)

    st.title("SongFinder 9001")
    st.caption(
        "Tell SongFinder your vibe and get song picks with the why behind "
        "each one. Built on a 30-track demo catalog."
    )

    rec_tab, build_tab, manage_tab, eval_tab = st.tabs(
        ["Find Songs", "Build Vibe Profile", "My Profiles", "Reliability"]
    )

    with rec_tab:
        _render_recommend_tab(client, show_debug)
    with build_tab:
        _render_build_tab(client, show_debug)
    with manage_tab:
        _render_manage_tab()
    with eval_tab:
        _render_eval_tab(client, show_debug)


main()
