"""Streamlit entry point for the Walker constellation coverage tool.

Keeps the top-level flow small: build the sidebar form, run the (cached)
analysis, dispatch to tab renderers.
"""

from __future__ import annotations

import streamlit as st

from walkercov.config import RunConfig
from walkercov.coverage.analysis import (
    RunArtifacts,
    compare_f_values,
    run_coverage,
)
from walkercov.ui.sidebar import sidebar_form
from walkercov.ui.tabs import (
    render_compare_f,
    render_constellation,
    render_coverage_maps,
    render_methods,
    render_statistics,
)


st.set_page_config(
    page_title="Walker constellation coverage",
    page_icon=":satellite:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached analysis entry points.  ``RunConfig`` is a frozen dataclass hierarchy
# so Streamlit can hash it reliably.
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _cached_run(config: RunConfig) -> RunArtifacts:
    bar = st.progress(0.0, text="Running coverage analysis")

    def _progress(frac: float, msg: str) -> None:
        bar.progress(min(1.0, frac), text=msg)

    art = run_coverage(config, progress=_progress)
    bar.empty()
    return art


@st.cache_data(show_spinner=False)
def _cached_compare(config: RunConfig) -> dict[int, RunArtifacts]:
    bar = st.progress(0.0, text="Comparing F values")

    def _progress(frac: float, msg: str) -> None:
        bar.progress(min(1.0, frac), text=msg)

    results = compare_f_values(config, progress=_progress)
    bar.empty()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("Walker constellation coverage")
    st.caption(
        "Interactive Walker Delta design and Earth-coverage analysis.  "
        "Vectorized two-body propagation, equal-area ground grid, explicit "
        "geometry."
    )

    config, submitted = sidebar_form()

    # Remember the most recent config across reruns so switching tabs does not
    # discard results.
    if "run_config" not in st.session_state or submitted:
        st.session_state.run_config = config

    config = st.session_state.run_config

    tabs = st.tabs([
        "Constellation",
        "Coverage maps",
        "Statistics",
        "Compare F",
        "Methods",
    ])

    with st.status("Running analysis", expanded=False) as status:
        art = _cached_run(config)
        status.update(label="Analysis complete", state="complete")

    with tabs[0]:
        render_constellation(art)
    with tabs[1]:
        render_coverage_maps(art)
    with tabs[2]:
        render_statistics(art)
    with tabs[3]:
        st.markdown(
            "Runs the current configuration with every valid F = 0..P-1.  "
            "Use this to see how Walker phasing affects coverage, revisit, "
            "and constellation geometry."
        )
        if st.button("Run F comparison", key="run_compare_f"):
            compare = _cached_compare(config)
            st.session_state["compare_cache"] = (config, compare)

        cached = st.session_state.get("compare_cache")
        if cached is not None and cached[0] == config:
            render_compare_f(cached[1])
        else:
            st.info("Click the button to run the comparison for the current configuration.")
    with tabs[4]:
        render_methods()


if __name__ == "__main__":
    main()
