"""Three-dimensional constellation view (Earth + orbits + satellites)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import plotly.graph_objects as go

from ..constants import R_EARTH
from ..orbit.elements import OrbitalElements
from ..orbit.propagator import CircularTwoBodyPropagator


# Earth is drawn in kilometres so the axis numbers stay small.
_EARTH_KM = R_EARTH / 1_000.0


def _earth_sphere(n: int = 48) -> go.Surface:
    """Return a solid Earth surface mesh."""

    u = np.linspace(0.0, 2.0 * np.pi, 2 * n + 1)
    v = np.linspace(0.0, np.pi, n + 1)
    uu, vv = np.meshgrid(u, v)
    x = _EARTH_KM * np.cos(uu) * np.sin(vv)
    y = _EARTH_KM * np.sin(uu) * np.sin(vv)
    z = _EARTH_KM * np.cos(vv)

    return go.Surface(
        x=x, y=y, z=z,
        showscale=False,
        opacity=0.6,
        colorscale=[(0.0, "#0b3d64"), (1.0, "#1e6ea1")],
        surfacecolor=np.cos(vv),
        hoverinfo="skip",
        name="Earth",
    )


def _equator_and_meridians() -> list[go.Scatter3d]:
    """Reference curves: equator and prime/anti-meridian."""

    out = []
    theta = np.linspace(0, 2 * np.pi, 181)
    # Equator.
    out.append(go.Scatter3d(
        x=_EARTH_KM * np.cos(theta), y=_EARTH_KM * np.sin(theta),
        z=np.zeros_like(theta),
        mode="lines",
        line=dict(color="#999", width=1),
        hoverinfo="skip", showlegend=False,
    ))
    # Prime meridian.
    phi = np.linspace(-np.pi / 2.0, np.pi / 2.0, 91)
    out.append(go.Scatter3d(
        x=_EARTH_KM * np.cos(phi), y=np.zeros_like(phi),
        z=_EARTH_KM * np.sin(phi),
        mode="lines",
        line=dict(color="#999", width=1),
        hoverinfo="skip", showlegend=False,
    ))
    return out


def _plane_colour(plane_id: int, total_planes: int) -> str:
    """Pick a distinct colour per plane using Plotly's qualitative palette."""

    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
    ]
    return palette[plane_id % len(palette)]


@dataclass
class _Snapshot:
    r_km: np.ndarray      # (N_sat, 3)
    orbits_km: dict[int, np.ndarray]   # plane_id -> (M, 3)


def _snapshot(elements: OrbitalElements, t_s: float,
              orbit_samples: int = 180) -> _Snapshot:
    """Compute satellite positions at ``t_s`` and the full orbit ring per plane."""

    prop = CircularTwoBodyPropagator()
    sat_eci = prop.propagate(elements, np.array([t_s]))[0]           # (N_sat, 3)
    sat_km = sat_eci / 1_000.0

    orbits: dict[int, np.ndarray] = {}
    # One ring per unique plane: sample one orbital period of a representative
    # satellite in that plane.
    for plane_id in np.unique(elements.plane_id):
        idx = int(np.where(elements.plane_id == plane_id)[0][0])
        single = _slice_elements(elements, idx)
        period = elements.period_s()[idx]
        tt = np.linspace(0.0, period, orbit_samples)
        ring_eci = prop.propagate(single, tt)[:, 0, :]
        orbits[int(plane_id)] = ring_eci / 1_000.0
    return _Snapshot(r_km=sat_km, orbits_km=orbits)


def _slice_elements(elements: OrbitalElements, idx: int) -> OrbitalElements:
    """Return a one-satellite element block (reusing dtypes)."""

    def _s(a: np.ndarray) -> np.ndarray:
        return a[idx:idx + 1]

    return OrbitalElements(
        semi_major_axis_m=_s(elements.semi_major_axis_m),
        eccentricity=_s(elements.eccentricity),
        inclination_rad=_s(elements.inclination_rad),
        raan_rad=_s(elements.raan_rad),
        arg_perigee_rad=_s(elements.arg_perigee_rad),
        mean_anomaly0_rad=_s(elements.mean_anomaly0_rad),
        plane_id=_s(elements.plane_id),
        slot_id=_s(elements.slot_id),
        shell_id=_s(elements.shell_id) if elements.shell_id is not None else None,
    )


def constellation_figure(elements: OrbitalElements,
                          t_s: float = 0.0,
                          *,
                          orbit_samples: int = 180) -> go.Figure:
    """Return a 3D Plotly figure of the constellation at a single epoch."""

    snap = _snapshot(elements, t_s, orbit_samples=orbit_samples)
    fig = go.Figure(_earth_sphere())
    for ref in _equator_and_meridians():
        fig.add_trace(ref)

    total_planes = int(elements.plane_id.max()) + 1
    for plane_id, ring in snap.orbits_km.items():
        colour = _plane_colour(plane_id, total_planes)
        fig.add_trace(go.Scatter3d(
            x=ring[:, 0], y=ring[:, 1], z=ring[:, 2],
            mode="lines",
            line=dict(color=colour, width=2),
            name=f"Plane {plane_id}",
            legendgroup=f"plane-{plane_id}",
            hoverinfo="skip",
        ))
        mask = elements.plane_id == plane_id
        pts = snap.r_km[mask]
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(color=colour, size=4, line=dict(color="black", width=0.5)),
            name=f"Plane {plane_id} sats",
            legendgroup=f"plane-{plane_id}",
            showlegend=False,
            hovertemplate="plane %{meta}<extra></extra>",
            meta=int(plane_id),
        ))

    r_max = float(np.linalg.norm(snap.r_km, axis=-1).max() * 1.05)
    axis_range = [-r_max, r_max]
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=axis_range),
            yaxis=dict(visible=False, range=axis_range),
            zaxis=dict(visible=False, range=axis_range),
            aspectmode="cube",
            bgcolor="black",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="black",
        font=dict(color="white"),
    )
    return fig
