"""Line/bar/histogram charts for coverage statistics."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..coverage.analysis import RunArtifacts
from ..coverage.metrics import CoverageResult


# ---------------------------------------------------------------------------
# Coverage vs time
# ---------------------------------------------------------------------------


def coverage_vs_time(result: CoverageResult) -> go.Figure:
    """Area-weighted global instantaneous coverage percentage versus time."""

    t_min = np.arange(result.num_steps) * result.time_step_s / 60.0
    fig = go.Figure(go.Scatter(
        x=t_min,
        y=100.0 * result.global_coverage_fraction,
        mode="lines",
        line=dict(color="#1f77b4"),
        name="Global coverage",
    ))
    fig.add_hline(
        y=100.0 * result.global_coverage_fraction.mean(),
        line=dict(color="orange", dash="dash", width=1),
        annotation_text=f"mean = {100.0 * result.global_coverage_fraction.mean():.1f}%",
        annotation_position="top left",
    )
    fig.update_layout(
        xaxis_title="Time [min]",
        yaxis_title="Instantaneous global coverage [%]",
        yaxis_range=[0, max(100.0, 100.0 * result.global_coverage_fraction.max() * 1.05)],
        margin=dict(l=50, r=10, t=20, b=40),
        height=320,
    )
    return fig


def coverage_vs_time_multi(labels: Iterable[str],
                            results: Iterable[CoverageResult]) -> go.Figure:
    """Overlay coverage-vs-time curves (for F comparison)."""

    fig = go.Figure()
    for label, res in zip(labels, results):
        t_min = np.arange(res.num_steps) * res.time_step_s / 60.0
        fig.add_trace(go.Scatter(
            x=t_min, y=100.0 * res.global_coverage_fraction,
            mode="lines", name=label,
        ))
    fig.update_layout(
        xaxis_title="Time [min]",
        yaxis_title="Instantaneous global coverage [%]",
        margin=dict(l=50, r=10, t=20, b=40),
        height=360,
    )
    return fig


# ---------------------------------------------------------------------------
# Metric vs latitude
# ---------------------------------------------------------------------------


def metric_vs_latitude(result: CoverageResult, metric: str) -> go.Figure:
    """Area-weighted mean of a per-point metric per latitude band."""

    arr = getattr(result, metric)
    if metric.endswith("_s"):
        arr = arr / 60.0
        y_label = f"{metric} [min]"
    elif metric == "percent_time_covered":
        arr = 100.0 * arr
        y_label = "Percent time covered [%]"
    else:
        y_label = metric

    band = result.by_latitude(arr)
    fig = go.Figure(go.Scatter(
        x=result.grid.lat_centres_deg,
        y=band,
        mode="lines",
        line=dict(color="#2ca02c"),
    ))
    fig.update_layout(
        xaxis_title="Latitude [deg]",
        yaxis_title=y_label,
        margin=dict(l=50, r=10, t=20, b=40),
        height=320,
    )
    return fig


# ---------------------------------------------------------------------------
# Histograms (area weighted)
# ---------------------------------------------------------------------------


def _area_weighted_histogram(values: np.ndarray,
                              weights: np.ndarray,
                              bins: int = 40) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(values) & (weights > 0)
    v = values[mask]
    w = weights[mask]
    if v.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0])
    counts, edges = np.histogram(v, bins=bins, weights=w)
    counts = counts / w.sum()
    return edges, counts


def revisit_histogram(result: CoverageResult, bins: int = 40) -> go.Figure:
    edges, counts = _area_weighted_histogram(
        result.mean_revisit_s / 60.0, result.grid.area_weights, bins=bins,
    )
    centres = 0.5 * (edges[:-1] + edges[1:])
    width = np.diff(edges)
    fig = go.Figure(go.Bar(x=centres, y=counts, width=width,
                            marker=dict(color="#9467bd")))
    fig.update_layout(
        xaxis_title="Mean revisit interval [min]",
        yaxis_title="Area fraction",
        margin=dict(l=50, r=10, t=20, b=40),
        height=320,
    )
    return fig


def max_gap_histogram(result: CoverageResult, bins: int = 40) -> go.Figure:
    edges, counts = _area_weighted_histogram(
        result.max_gap_s / 60.0, result.grid.area_weights, bins=bins,
    )
    centres = 0.5 * (edges[:-1] + edges[1:])
    width = np.diff(edges)
    fig = go.Figure(go.Bar(x=centres, y=counts, width=width,
                            marker=dict(color="#d62728")))
    fig.update_layout(
        xaxis_title="Max coverage gap [min]",
        yaxis_title="Area fraction",
        margin=dict(l=50, r=10, t=20, b=40),
        height=320,
    )
    return fig


# ---------------------------------------------------------------------------
# F-comparison
# ---------------------------------------------------------------------------


def f_comparison_bar(df: pd.DataFrame, metric_col: str,
                      *, title: str | None = None) -> go.Figure:
    """Bar chart of one metric from the compare_f summary table."""

    fig = go.Figure(go.Bar(
        x=df.index.astype(str),
        y=df[metric_col],
        marker=dict(color="#17becf"),
        text=df[metric_col].round(2),
        textposition="outside",
    ))
    fig.update_layout(
        title=title or metric_col,
        xaxis_title="F",
        yaxis_title=metric_col,
        margin=dict(l=50, r=10, t=50, b=40),
        height=360,
    )
    return fig


def plane_phase_diagram(artifacts: RunArtifacts) -> go.Figure:
    """Polar plot of each plane's slot positions at t=0.

    This is the clearest visual intuition for what F does: it rotates the
    slot pattern between adjacent planes.
    """

    elements = artifacts.elements
    fig = go.Figure()
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                "#bcbd22", "#17becf"]
    u0 = np.rad2deg(elements.arg_of_latitude0_rad())
    for p in np.unique(elements.plane_id):
        mask = elements.plane_id == p
        fig.add_trace(go.Scatterpolar(
            r=np.full(int(mask.sum()), int(p) + 1),  # radial = plane index
            theta=u0[mask],
            mode="markers",
            marker=dict(color=palette[int(p) % len(palette)], size=10),
            name=f"Plane {int(p)}",
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, tickmode="linear", dtick=1),
            angularaxis=dict(tickmode="linear", dtick=30),
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        height=420,
    )
    return fig
