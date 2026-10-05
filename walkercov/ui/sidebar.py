"""Streamlit sidebar form that produces a :class:`RunConfig`.

Grouped by purpose; defaults chosen to match the plan.  Presets provide a one-
click way to switch between an interactive-fast run and a high-resolution one.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from ..config import CoverageConfig, RunConfig, SimulationConfig, WalkerConfig


@dataclass(frozen=True)
class _Preset:
    label: str
    lat_step_deg: float
    lon_step_deg: float
    time_step_s: float


_PRESETS: dict[str, _Preset] = {
    "Fast (interactive)": _Preset("Fast", 4.0, 4.0, 120.0),
    "Standard": _Preset("Standard", 2.0, 2.0, 60.0),
    "High resolution": _Preset("High-res", 1.0, 1.0, 30.0),
    "Custom": _Preset("Custom", 2.0, 2.0, 60.0),
}


def sidebar_form() -> tuple[RunConfig, bool]:
    """Render the sidebar form.  Returns ``(config, submitted)``."""

    st.sidebar.markdown("## Walker constellation")
    T = st.sidebar.number_input("Total satellites T", min_value=1, max_value=400,
                                 value=24, step=1)
    P = st.sidebar.number_input("Orbital planes P", min_value=1, max_value=int(T),
                                 value=min(6, int(T)), step=1)
    # Clamp P to a divisor of T.
    if T % P != 0:
        candidates = [p for p in range(1, int(T) + 1) if T % p == 0]
        P = min(candidates, key=lambda p: abs(p - int(P)))
        st.sidebar.info(f"P adjusted to {P} so that T is divisible by P.")
    F = st.sidebar.selectbox("Phasing F", options=list(range(int(P))),
                              index=min(1, int(P) - 1))

    altitude_km = st.sidebar.number_input("Altitude [km]", min_value=100.0,
                                            max_value=40_000.0, value=550.0,
                                            step=10.0)
    inclination_deg = st.sidebar.number_input("Inclination [deg]",
                                                min_value=0.0, max_value=180.0,
                                                value=53.0, step=1.0)

    st.sidebar.markdown("## Coverage definition")
    min_elev = st.sidebar.number_input("Min elevation [deg]", min_value=0.0,
                                        max_value=89.0, value=10.0, step=1.0)
    required_sats = st.sidebar.number_input("Required satellites k",
                                              min_value=1, max_value=10,
                                              value=1, step=1)

    st.sidebar.markdown("## Simulation")
    preset_name = st.sidebar.selectbox("Preset", list(_PRESETS.keys()),
                                        index=1)
    preset = _PRESETS[preset_name]
    duration_hr = st.sidebar.number_input("Duration [hours]", min_value=0.5,
                                            max_value=168.0, value=24.0, step=0.5)
    if preset_name == "Custom":
        time_step_s = st.sidebar.number_input("Time step [s]", min_value=1.0,
                                                max_value=600.0,
                                                value=preset.time_step_s, step=5.0)
        lat_step = st.sidebar.number_input("Lat grid step [deg]",
                                            min_value=0.25, max_value=10.0,
                                            value=preset.lat_step_deg, step=0.25)
        lon_step = st.sidebar.number_input("Lon grid step [deg]",
                                            min_value=0.25, max_value=10.0,
                                            value=preset.lon_step_deg, step=0.25)
    else:
        time_step_s = preset.time_step_s
        lat_step = preset.lat_step_deg
        lon_step = preset.lon_step_deg
        st.sidebar.caption(
            f"dt = {time_step_s:.0f} s, grid = {lat_step:.1f}\u00b0 x {lon_step:.1f}\u00b0"
        )

    greenwich0 = st.sidebar.number_input("Greenwich angle at t=0 [deg]",
                                           min_value=0.0, max_value=360.0,
                                           value=0.0, step=5.0)

    submitted = st.sidebar.button("Run analysis", type="primary",
                                   width="stretch")

    walker = WalkerConfig(
        total_sats=int(T), num_planes=int(P), phasing_f=int(F),
        altitude_km=float(altitude_km), inclination_deg=float(inclination_deg),
    )
    sim = SimulationConfig(
        duration_s=float(duration_hr) * 3600.0,
        time_step_s=float(time_step_s),
        greenwich_angle0_deg=float(greenwich0),
    )
    cov = CoverageConfig(
        lat_step_deg=float(lat_step),
        lon_step_deg=float(lon_step),
        min_elevation_deg=float(min_elev),
        required_sats=int(required_sats),
    )
    return RunConfig(walker=walker, simulation=sim, coverage=cov), submitted
