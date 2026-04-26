"""Streamlit single-page wrapper around run_pipeline.

Run with:  streamlit run app.py

Requires GEMINI_API_KEY in .env for grounded explanations. Without a key,
the page renders a stub-mode warning and the pipeline will fail before
producing recommendations (NL extraction needs an LLM).
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from src.agents.profile_extractor import ProfileExtractionError
from src.llm.client import CachedLLMClient, GeminiClient, LLMClient, StubLLMClient
from src.pipeline import PipelineResult, run_pipeline


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


def _render_card(rec, expl) -> None:
    with st.container(border=True):
        cols = st.columns([4, 1])
        cols[0].markdown(f"### {rec.song.title}")
        cols[0].markdown(f"_{rec.song.artist}_  -  `{rec.song.genre}` / `{rec.song.mood}`")
        cols[1].metric("score", f"{rec.score:.2f}")

        if expl.text:
            st.write(expl.text)
        else:
            st.markdown(
                f"_Mechanical reasons only "
                f"(fallback: `{expl.fallback_reason}`)._"
            )

        with st.expander("Mechanical reasons"):
            for reason in rec.reasons:
                st.markdown(f"- {reason}")


def _render_results(result: PipelineResult) -> None:
    st.markdown(f"### Top {len(result.recommendations)} for: _{result.nl_input}_")
    for rec, expl in zip(result.recommendations, result.explanations):
        _render_card(rec, expl)


def _render_debug(result: PipelineResult) -> None:
    st.divider()
    st.markdown("## Debug")

    if result.extractor_warnings:
        st.markdown("### Extractor warnings")
        for warning in result.extractor_warnings:
            st.warning(warning)

    st.markdown("### Extracted profile")
    p = result.extracted_profile
    profile_row = {
        "favorite_genre": [p.favorite_genre],
        "favorite_mood": [p.favorite_mood],
        "target_energy": [p.target_energy],
        "target_tempo_bpm": [p.target_tempo_bpm],
        "target_valence": [p.target_valence],
        "target_danceability": [p.target_danceability],
        "target_acousticness": [p.target_acousticness],
    }
    st.dataframe(profile_row, use_container_width=True, hide_index=True)

    if result.final_profile != result.extracted_profile:
        st.markdown("### Final profile (after critic refinement)")
        f = result.final_profile
        final_row = {
            "favorite_genre": [f.favorite_genre],
            "favorite_mood": [f.favorite_mood],
            "target_energy": [f.target_energy],
            "target_tempo_bpm": [f.target_tempo_bpm],
            "target_valence": [f.target_valence],
            "target_danceability": [f.target_danceability],
            "target_acousticness": [f.target_acousticness],
        }
        st.dataframe(final_row, use_container_width=True, hide_index=True)

    st.markdown("### Refinement history")
    if result.refinement_history:
        history_rows = [
            {
                "iter": s.iter_index,
                "verdict": s.verdict,
                "top5_song_ids": ", ".join(str(i) for i in s.top5_song_ids),
                "adjustments": str(s.adjustments_applied or {}),
                "reason": s.reason,
            }
            for s in result.refinement_history
        ]
        st.dataframe(history_rows, use_container_width=True, hide_index=True)
    else:
        st.write("(none)")

    cols = st.columns(2)
    cols[0].markdown("### Ambiguous match")
    cols[0].markdown(
        "**Yes**" if result.ambiguous_match else "No"
    )
    cols[1].markdown("### Cache stats (this run)")
    if result.cache_stats:
        cols[1].markdown(
            f"**{result.cache_stats.get('hits', 0)} hits** / "
            f"**{result.cache_stats.get('misses', 0)} misses**"
        )
    else:
        cols[1].markdown("(stub mode - no cache)")


def main() -> None:
    st.set_page_config(page_title="Music Recommender", layout="wide")

    client = _get_client()
    show_debug = _render_sidebar(client)

    st.title("Music Recommender")
    st.caption(
        "Type a free-form English request; the system extracts a profile, scores "
        "30 songs, asks an LLM critic to validate, and produces grounded explanations."
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
                result = run_pipeline(nl_input, llm=client)
                st.session_state["last_result"] = result
                st.session_state.pop("last_error", None)
            except ProfileExtractionError as exc:
                st.session_state["last_error"] = (
                    f"Could not parse the request - try rephrasing.\n\n{exc}"
                )
                st.session_state.pop("last_result", None)
            except Exception as exc:
                st.session_state["last_error"] = (
                    f"Pipeline failed: {exc.__class__.__name__}: {exc}"
                )
                st.session_state.pop("last_result", None)

    if "last_error" in st.session_state:
        st.error(st.session_state["last_error"])

    last_result = st.session_state.get("last_result")
    if last_result is not None:
        _render_results(last_result)
        if show_debug:
            _render_debug(last_result)


main()
