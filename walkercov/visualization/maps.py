"""2-D latitude/longitude map visualisations."""

from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go

from ..config import RunConfig
from ..coverage.metrics import CoverageResult
from ..orbit.coordinates import eci_to_ecef, greenwich_angle_rad, sub_satellite_latlon_deg
from ..orbit.elements import OrbitalElements
from ..orbit.propagator import CircularTwoBodyPropagator


METRIC_LABELS: dict[str, str] = {
    "percent_time_covered": "Percent of simulation time covered [%]",
    "mean_satellites_visible": "Mean satellites visible",
    "min_satellites_visible": "Minimum satellites visible",
    "max_satellites_visible": "Maximum satellites visible",
    "max_gap_s": "Maximum coverage gap [min]",
    "mean_revisit_s": "Mean revisit interval [min]",
    "max_revisit_s": "Maximum revisit interval [min]",
    "mean_response_time_s": "Mean response time [min]",
}


def _full_sphere_axes() -> dict:
    """Lat/lon axes locked to the whole sphere.

    ``scaleanchor`` keeps one degree of longitude the same length as one
    degree of latitude.  ``constrain="domain"`` makes Plotly letterbox the
    plot instead of stretching the numeric range past ±180° / ±90° to fill
    a wide chart.
    """

    return dict(
        xaxis=dict(
            title="Longitude [deg]",
            range=[-180, 180],
            autorange=False,
            constrain="domain",
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            title="Latitude [deg]",
            range=[-90, 90],
            autorange=False,
            constrain="domain",
        ),
    )


def _extract_metric(result: CoverageResult, metric: str) -> np.ndarray:
    """Return the metric reshaped to (n_lat, n_lon), with sensible unit conversion."""

    arr = getattr(result, metric)
    if metric.endswith("_s"):
        arr = arr / 60.0  # seconds -> minutes
    if metric == "percent_time_covered":
        arr = 100.0 * arr
    return arr.reshape(result.grid.shape)


def metric_heatmap(result: CoverageResult,
                    metric: str,
                    *,
                    colorscale: str = "Viridis",
                    title: Optional[str] = None) -> go.Figure:
    """Return a lat/lon heatmap of ``metric``."""

    data = _extract_metric(result, metric)
    label = METRIC_LABELS.get(metric, metric)

    fig = go.Figure(go.Heatmap(
        x=result.grid.lon_centres_deg,
        y=result.grid.lat_centres_deg,
        z=data,
        colorscale=colorscale,
        colorbar=dict(title=label),
        hovertemplate=(
            "lat=%{y:.1f}\u00b0<br>"
            "lon=%{x:.1f}\u00b0<br>"
            "value=%{z:.3f}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=title or label,
        margin=dict(l=50, r=10, t=50, b=40),
        height=520,
        **_full_sphere_axes(),
    )
    return fig


def instantaneous_coverage_map(result: CoverageResult,
                                config: RunConfig,
                                elements: OrbitalElements,
                                epoch_s: float) -> go.Figure:
    """Snapshot heatmap of "how many satellites are visible now" plus sub-satellite points."""

    prop = CircularTwoBodyPropagator()
    r_eci = prop.propagate(elements, np.array([epoch_s]))
    theta = greenwich_angle_rad(
        np.array([epoch_s]),
        np.deg2rad(config.simulation.greenwich_angle0_deg),
    )
    r_ecef = eci_to_ecef(r_eci, theta)[0]                        # (N_sat, 3)

    # Dot product trick for a snapshot too.
    r_norm = np.linalg.norm(r_ecef, axis=-1, keepdims=True)
    sat_unit = r_ecef / r_norm
    cos_angle = sat_unit @ result.grid.unit_vectors.T            # (N_sat, G)
    from ..coverage.visibility import max_central_angle
    from ..constants import R_EARTH
    altitudes = elements.semi_major_axis_m - R_EARTH
    lam = max_central_angle(altitudes, np.deg2rad(config.coverage.min_elevation_deg))
    counts = (cos_angle >= np.cos(lam)[:, None]).sum(axis=0)      # (G,)
    data = counts.reshape(result.grid.shape)

    sub_lat, sub_lon = sub_satellite_latlon_deg(r_ecef)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=result.grid.lon_centres_deg,
        y=result.grid.lat_centres_deg,
        z=data,
        colorscale="Viridis",
        zmin=0,
        colorbar=dict(title="Satellites visible"),
        hovertemplate="lat=%{y:.1f}\u00b0<br>lon=%{x:.1f}\u00b0<br>n=%{z}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=sub_lon, y=sub_lat,
        mode="markers",
        marker=dict(color="red", size=6, line=dict(color="white", width=1)),
        name="sub-sat",
        hovertemplate="sub-satellite<br>lat=%{y:.1f}\u00b0<br>lon=%{x:.1f}\u00b0<extra></extra>",
    ))
    fig.update_layout(
        title=f"Instantaneous coverage at t = {epoch_s / 60.0:.1f} min",
        margin=dict(l=50, r=10, t=50, b=40),
        height=520,
        **_full_sphere_axes(),
    )
    return fig


def ground_tracks_map(config: RunConfig,
                       elements: OrbitalElements,
                       *,
                       max_tracks: int = 24,
                       samples: int = 400) -> go.Figure:
    """Plot ground tracks over one full orbital period for up to ``max_tracks`` sats."""

    prop = CircularTwoBodyPropagator()
    period = elements.period_s()[0]
    tt = np.linspace(0.0, period, samples)
    r_eci = prop.propagate(elements, tt)
    theta = greenwich_angle_rad(
        tt, np.deg2rad(config.simulation.greenwich_angle0_deg),
    )
    r_ecef = eci_to_ecef(r_eci, theta)                           # (T, N, 3)
    lat_deg, lon_deg = sub_satellite_latlon_deg(r_ecef)           # (T, N)

    # Insert NaN wherever longitude wraps so Plotly breaks the line.  The
    # wrap sits between row k and row k+1; blanking row k+1 is sufficient to
    # stop Plotly drawing the long horizontal segment.
    lon_plot = lon_deg.astype(np.float64, copy=True)
    lat_plot = lat_deg.astype(np.float64)
    dlon = np.diff(lon_deg, axis=0)
    wrap_mask = np.abs(dlon) > 180.0
    tail = lon_plot[1:, :]
    tail[wrap_mask] = np.nan
    _, N = lon_plot.shape

    fig = go.Figure()
    traces_shown = 0
    for n in range(min(N, max_tracks)):
        plane_id = int(elements.plane_id[n])
        colour = _plane_palette(plane_id)
        fig.add_trace(go.Scatter(
            x=lon_plot[:, n], y=lat_plot[:, n],
            mode="lines",
            line=dict(color=colour, width=1),
            name=f"P{plane_id}S{int(elements.slot_id[n])}",
            legendgroup=f"plane-{plane_id}",
            showlegend=(traces_shown < 12),
            hoverinfo="skip",
        ))
        traces_shown += 1
    fig.update_layout(
        title="Ground tracks over one orbital period",
        margin=dict(l=50, r=10, t=50, b=40),
        height=520,
        **_full_sphere_axes(),
    )
    return fig


def _plane_palette(plane_id: int) -> str:
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf",
    ]
    return palette[plane_id % len(palette)]
