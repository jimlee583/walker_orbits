"""Streamlit tab layouts.

Each ``render_*`` function receives a :class:`RunArtifacts` (and optionally the
F-comparison dict) and writes one tab's worth of UI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ..config import RunConfig
from ..constants import R_EARTH
from ..coverage.analysis import RunArtifacts, comparison_dataframe
from ..coverage.visibility import max_central_angle, satellite_footprint_radius_km
from ..orbit.walker import satellite_table
from ..visualization import charts, globe3d, maps


# ---------------------------------------------------------------------------
# Constellation tab
# ---------------------------------------------------------------------------


def render_constellation(art: RunArtifacts) -> None:
    walker = art.config.walker
    elements = art.elements

    period_min = float(elements.period_s()[0] / 60.0)
    lam_rad = max_central_angle(walker.altitude_km * 1_000.0,
                                 np.deg2rad(art.config.coverage.min_elevation_deg))
    lam_deg = float(np.rad2deg(lam_rad))
    footprint_km = satellite_footprint_radius_km(
        walker.altitude_km * 1_000.0,
        np.deg2rad(art.config.coverage.min_elevation_deg),
    )
    max_pass_s = 2.0 * lam_rad / float(elements.mean_motion_rad_s()[0])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Period", f"{period_min:.2f} min")
    c2.metric("Footprint half-angle", f"{lam_deg:.2f}\u00b0",
               help="Earth-centre half-angle (lambda_max) at the selected minimum elevation.")
    c3.metric("Footprint radius", f"{footprint_km:,.0f} km")
    c4.metric("Max pass duration", f"{max_pass_s / 60.0:.2f} min",
               help="Longest possible pass (satellite directly overhead).")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Satellites per plane", f"{walker.sats_per_plane}")
    c2.metric("RAAN step", f"{walker.raan_step_deg:.2f}\u00b0")
    c3.metric("In-plane step", f"{walker.in_plane_step_deg:.2f}\u00b0")
    c4.metric("Walker phase step", f"{walker.phase_increment_deg:.2f}\u00b0")

    dt = art.config.simulation.time_step_s
    if dt > 0.2 * max_pass_s:
        st.warning(
            f"Time step dt = {dt:.0f} s is more than 20% of the max pass duration "
            f"({max_pass_s:.0f} s).  Access-timing metrics may be noisy; consider "
            "using the Standard or High-resolution preset."
        )

    st.markdown("#### 3D constellation view")
    t_sel_min = st.slider("Epoch [min]", min_value=0.0,
                           max_value=float(art.config.simulation.duration_s / 60.0),
                           value=0.0, step=max(1.0, dt / 60.0))
    fig3d = globe3d.constellation_figure(elements, t_s=t_sel_min * 60.0)
    st.plotly_chart(fig3d, width="stretch")

    st.markdown("#### Ground tracks (one orbital period)")
    st.plotly_chart(maps.ground_tracks_map(art.config, elements),
                     width="stretch")

    st.markdown("#### Satellite table")
    df = satellite_table(walker, elements)
    st.dataframe(df, hide_index=True, width="stretch")
    st.download_button(
        "Download satellite table (CSV)",
        data=df.to_csv(index=False).encode(),
        file_name=f"walker_{walker.total_sats}_{walker.num_planes}_"
                   f"{walker.phasing_f}_satellites.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Coverage maps tab
# ---------------------------------------------------------------------------


def render_coverage_maps(art: RunArtifacts) -> None:
    metrics = list(maps.METRIC_LABELS.keys())
    labels = [maps.METRIC_LABELS[m] for m in metrics]
    choice = st.selectbox("Metric", options=list(range(len(metrics))),
                           format_func=lambda i: labels[i])
    fig = maps.metric_heatmap(art.result, metrics[choice])
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### Instantaneous coverage snapshot")
    dt = art.config.simulation.time_step_s
    t_sel_min = st.slider(
        "Snapshot epoch [min]",
        min_value=0.0, max_value=float(art.config.simulation.duration_s / 60.0),
        value=0.0, step=max(1.0, dt / 60.0), key="snapshot_slider",
    )
    st.plotly_chart(
        maps.instantaneous_coverage_map(art.result, art.config, art.elements,
                                         epoch_s=t_sel_min * 60.0),
        width="stretch",
    )


# ---------------------------------------------------------------------------
# Statistics tab
# ---------------------------------------------------------------------------


def render_statistics(art: RunArtifacts) -> None:
    s = art.result.summary()

    st.markdown("#### Instantaneous coverage (area weighted)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean", f"{100.0 * s['mean_global_coverage']:.2f} %")
    c2.metric("Minimum", f"{100.0 * s['min_global_coverage']:.2f} %")
    c3.metric("Maximum", f"{100.0 * s['max_global_coverage']:.2f} %")

    st.markdown("#### Satellites visible (area weighted)")
    c1, c2 = st.columns(2)
    c1.metric("Mean satellites visible", f"{s['mean_mean_sats_visible']:.3f}")
    c2.metric("Area ever covered", f"{100.0 * s['area_fraction_ever_covered']:.1f} %")

    st.markdown("#### Gaps and revisits (area weighted, finite only)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max coverage gap", _fmt_minutes(s["max_gap_s_global"]))
    c2.metric("Mean response time", _fmt_minutes(s["mean_response_time_s_global"]))
    c3.metric("Mean revisit", _fmt_minutes(s["mean_revisit_s_global"]))
    c4.metric("Max revisit", _fmt_minutes(s["max_revisit_s_global"]))

    st.caption(
        "Revisit interval is the time between successive access starts at a "
        "point.  Gaps that touch the start or end of the simulation window are "
        "censored (they may be longer than observed) and are therefore included "
        "in the max-gap/response-time statistics but not the revisit statistics."
    )

    st.markdown("#### Global coverage versus time")
    st.plotly_chart(charts.coverage_vs_time(art.result),
                     width="stretch")

    st.markdown("#### Metric versus latitude")
    metric = st.selectbox(
        "Metric vs latitude",
        options=list(maps.METRIC_LABELS.keys()),
        format_func=lambda m: maps.METRIC_LABELS[m],
        key="lat_metric",
    )
    st.plotly_chart(charts.metric_vs_latitude(art.result, metric),
                     width="stretch")

    st.markdown("#### Distributions (area weighted)")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.revisit_histogram(art.result),
                         width="stretch")
    with c2:
        st.plotly_chart(charts.max_gap_histogram(art.result),
                         width="stretch")


# ---------------------------------------------------------------------------
# Compare F tab
# ---------------------------------------------------------------------------


def render_compare_f(results: dict[int, RunArtifacts]) -> None:
    df = comparison_dataframe(results)
    st.markdown("#### Summary table")
    st.dataframe(df.style.format("{:.2f}"), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.f_comparison_bar(df, "mean_global_coverage_%",
                                                  title="Mean global coverage [%]"),
                         width="stretch")
        st.plotly_chart(charts.f_comparison_bar(df, "max_gap_min",
                                                  title="Max coverage gap [min]"),
                         width="stretch")
    with c2:
        st.plotly_chart(charts.f_comparison_bar(df, "mean_sats_visible",
                                                  title="Mean satellites visible"),
                         width="stretch")
        st.plotly_chart(charts.f_comparison_bar(df, "mean_response_min",
                                                  title="Mean response time [min]"),
                         width="stretch")

    st.markdown("#### Instantaneous coverage versus time")
    labels = [f"F={f}" for f in results]
    rsets = [a.result for a in results.values()]
    st.plotly_chart(charts.coverage_vs_time_multi(labels, rsets),
                     width="stretch")

    st.markdown("#### Initial plane / slot pattern")
    choice = st.selectbox("Phase diagram for", options=list(results.keys()),
                           format_func=lambda f: f"F = {f}")
    st.plotly_chart(charts.plane_phase_diagram(results[choice]),
                     width="stretch")


# ---------------------------------------------------------------------------
# Methods tab
# ---------------------------------------------------------------------------


def render_methods() -> None:
    md = (Path(__file__).parent / "methods.md").read_text()
    st.markdown(md)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_minutes(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "n/a"
    if seconds < 120.0:
        return f"{seconds:.1f} s"
    return f"{seconds / 60.0:.2f} min"
