"""Streamlit single-page wrapper around build_profile + recommend.

Run with:  streamlit run app.py

Requires GEMINI_API_KEY in .env for grounded explanations. Without a key,
the page renders a stub-mode warning and the pipeline will fail before
producing recommendations (NL extraction needs an LLM).

Per D35 this is a minimal patch on the existing single-textarea UI:
the textarea text is wrapped into BuildInputs(description=...), then
build_profile + recommend are chained. The full UI redesign that
surfaces the five-question form is Phase 6.
"""

from __future__ import annotations

import os

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


SAMPLE_NL = (
    "Quiet, mellow lofi to focus on writing - slow tempo, lots of warmth, "
    "not too cheerful but not sad."
)


load_dotenv()


@st.cache_resource
def _get_client() -> LLMClient:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if key:
        return CachedLLMClient(GeminiClient(api_key=key))
    return StubLLMClient([])


def _is_online(client: LLMClient) -> bool:
    return isinstance(client, CachedLLMClient)


def _render_sidebar(client: LLMClient) -> bool:
    st.sidebar.markdown("## Status")
    if _is_online(client):
        st.sidebar.success("Connected to Gemini")
    else:
        st.sidebar.warning("Stub mode (no GEMINI_API_KEY)")
    if isinstance(client, CachedLLMClient):
        st.sidebar.markdown(
            f"**Cache:** {client.hits} hits / {client.misses} misses (this session)"
        )
    st.sidebar.divider()
    return st.sidebar.checkbox("Show debug pane", value=False, key="show_debug")


def _render_card(scored, expl) -> None:
    with st.container(border=True):
        cols = st.columns([4, 1])
        cols[0].markdown(f"### {scored.song.title}")
        cols[0].markdown(f"_{scored.song.artist}_  -  `{scored.song.genre}` / `{scored.song.mood}`")
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


def _render_results(nl_input: str, rec: RecommendationResult) -> None:
    st.markdown(f"### Top {len(rec.recommendations)} for: _{nl_input}_")
    for scored, expl in zip(rec.recommendations, rec.explanations):
        _render_card(scored, expl)


def _profile_row(profile) -> dict:
    return {
        "favorite_genre": [profile.favorite_genre],
        "favorite_mood": [profile.favorite_mood],
        "target_energy": [profile.target_energy],
        "target_tempo_bpm": [profile.target_tempo_bpm],
        "target_valence": [profile.target_valence],
        "target_danceability": [profile.target_danceability],
        "target_acousticness": [profile.target_acousticness],
    }


def _render_debug(build: ProfileBuildResult, rec: RecommendationResult) -> None:
    st.divider()
    st.markdown("## Debug")

    if build.extractor_warnings:
        st.markdown("### Extractor warnings")
        for warning in build.extractor_warnings:
            st.warning(warning)

    st.markdown("### Extracted profile")
    st.dataframe(_profile_row(build.extracted_profile), use_container_width=True, hide_index=True)

    if build.candidate_profile != build.extracted_profile:
        st.markdown("### Candidate profile (after critic refinement)")
        st.dataframe(_profile_row(build.candidate_profile), use_container_width=True, hide_index=True)

    st.markdown("### Refinement history")
    if build.refinement_history:
        history_rows = [
            {
                "iter": s.iter_index,
                "verdict": s.verdict,
                "adjustments": str(s.adjustments_applied or {}),
                "reason": s.reason,
            }
            for s in build.refinement_history
        ]
        st.dataframe(history_rows, use_container_width=True, hide_index=True)
    else:
        st.write("(none)")

    cols = st.columns(2)
    cols[0].markdown("### Ambiguous match")
    cols[0].markdown("**Yes**" if build.ambiguous_match else "No")
    cols[1].markdown("### Cache stats (this run)")
    if rec.cache_stats:
        cols[1].markdown(
            f"**{rec.cache_stats.get('hits', 0)} hits** / "
            f"**{rec.cache_stats.get('misses', 0)} misses**"
        )
    else:
        cols[1].markdown("(stub mode - no cache)")


def main() -> None:
    st.set_page_config(page_title="Music Recommender", layout="wide")

    client = _get_client()
    show_debug = _render_sidebar(client)

    st.title("Music Recommender")
    st.caption(
        "Type a free-form English request; the system builds a listener "
        "profile from it (LLM extractor + critic), then ranks 30 songs and "
        "produces grounded explanations against the candidate profile."
    )

    nl_input = st.text_area(
        "What are you in the mood for?",
        value=SAMPLE_NL,
        height=80,
        key="nl_input",
    )

    if st.button("Recommend", type="primary"):
        with st.spinner("Running pipeline..."):
            try:
                build = build_profile(BuildInputs(description=nl_input), client)
                rec = recommend(build.candidate_profile, client)
                st.session_state["last_nl_input"] = nl_input
                st.session_state["last_build"] = build
                st.session_state["last_rec"] = rec
                st.session_state.pop("last_error", None)
            except EmptyBuildInputsError as exc:
                st.session_state["last_error"] = f"Empty input - {exc}"
                st.session_state.pop("last_build", None)
                st.session_state.pop("last_rec", None)
            except ProfileExtractionError as exc:
                st.session_state["last_error"] = (
                    f"Could not parse the request - try rephrasing.\n\n{exc}"
                )
                st.session_state.pop("last_build", None)
                st.session_state.pop("last_rec", None)
            except Exception as exc:
                st.session_state["last_error"] = (
                    f"Pipeline failed: {exc.__class__.__name__}: {exc}"
                )
                st.session_state.pop("last_build", None)
                st.session_state.pop("last_rec", None)

    if "last_error" in st.session_state:
        st.error(st.session_state["last_error"])

    last_build = st.session_state.get("last_build")
    last_rec = st.session_state.get("last_rec")
    if last_build is not None and last_rec is not None:
        _render_results(st.session_state.get("last_nl_input", ""), last_rec)
        if show_debug:
            _render_debug(last_build, last_rec)


main()
