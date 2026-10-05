"""Analytic end-to-end coverage sanity checks.

These are the "I believe the plots because the numbers match" tests.
"""

from __future__ import annotations

import numpy as np

from walkercov.config import CoverageConfig, RunConfig, SimulationConfig, WalkerConfig
from walkercov.constants import MU_EARTH, OMEGA_EARTH, R_EARTH, SIDEREAL_DAY_S
from walkercov.coverage.analysis import run_coverage
from walkercov.coverage.visibility import max_central_angle


# ---------------------------------------------------------------------------
# Analytic cap fraction for a single satellite
# ---------------------------------------------------------------------------


def _spherical_cap_fraction(lam: float) -> float:
    """Fraction of a sphere covered by a cap of half-angle lam."""

    return 0.5 * (1.0 - np.cos(lam))


def test_single_satellite_instantaneous_global_coverage_matches_cap() -> None:
    walker = WalkerConfig(total_sats=1, num_planes=1, phasing_f=0,
                           altitude_km=550.0, inclination_deg=0.0)
    sim = SimulationConfig(duration_s=1.0, time_step_s=1.0)
    cov = CoverageConfig(lat_step_deg=1.0, lon_step_deg=1.0,
                          min_elevation_deg=10.0, required_sats=1)
    art = run_coverage(RunConfig(walker=walker, simulation=sim, coverage=cov))

    lam = max_central_angle(550_000.0, np.deg2rad(10.0))
    expected = _spherical_cap_fraction(lam)
    # Area-weighted equal-area grid, so a 1 deg grid converges to <0.5% error.
    np.testing.assert_allclose(art.result.global_coverage_fraction[0], expected,
                                rtol=0.01)


def test_point_outside_coverage_latitude_band_is_never_covered() -> None:
    inc_deg = 30.0
    walker = WalkerConfig(total_sats=4, num_planes=1, phasing_f=0,
                           altitude_km=550.0, inclination_deg=inc_deg)
    sim = SimulationConfig(duration_s=6 * 3600.0, time_step_s=120.0)
    cov = CoverageConfig(lat_step_deg=5.0, lon_step_deg=10.0,
                          min_elevation_deg=10.0, required_sats=1)
    art = run_coverage(RunConfig(walker=walker, simulation=sim, coverage=cov))

    lam_deg = np.rad2deg(max_central_angle(550_000.0, np.deg2rad(10.0)))
    forbidden = inc_deg + lam_deg + 1.0  # with 1 deg safety margin
    far_mask = np.abs(np.rad2deg(art.result.grid.lat_rad)) > forbidden
    assert far_mask.any()
    assert not art.result.any_access[far_mask].any()


def test_geo_sub_satellite_point_is_always_covered() -> None:
    a_geo = (MU_EARTH / OMEGA_EARTH ** 2) ** (1.0 / 3.0)
    altitude_km = (a_geo - R_EARTH) / 1_000.0
    walker = WalkerConfig(total_sats=1, num_planes=1, phasing_f=0,
                           altitude_km=altitude_km, inclination_deg=0.0)
    # Half a sidereal day is enough to see the "always covered" property
    # within the equatorial footprint.
    sim = SimulationConfig(duration_s=SIDEREAL_DAY_S / 2.0, time_step_s=600.0,
                            greenwich_angle0_deg=0.0)
    cov = CoverageConfig(lat_step_deg=2.0, lon_step_deg=2.0,
                          min_elevation_deg=5.0, required_sats=1)
    art = run_coverage(RunConfig(walker=walker, simulation=sim, coverage=cov))

    # Sub-satellite point sits at (lat, lon) = (0, 0) in ECEF forever.
    lat_c = art.result.grid.lat_centres_deg
    lon_c = art.result.grid.lon_centres_deg
    i = int(np.argmin(np.abs(lat_c)))
    j = int(np.argmin(np.abs(lon_c)))
    flat_idx = i * len(lon_c) + j
    assert art.result.percent_time_covered[flat_idx] == 1.0


def test_polar_point_sees_polar_satellite_once_per_orbit() -> None:
    # Satellite in a 90 deg orbit: it crosses straight over the pole on every
    # orbit, so the revisit interval at the pole equals the orbital period
    # (within one time step).
    walker = WalkerConfig(total_sats=1, num_planes=1, phasing_f=0,
                           altitude_km=550.0, inclination_deg=90.0)
    sim = SimulationConfig(duration_s=4 * 95.6 * 60.0, time_step_s=20.0)
    cov = CoverageConfig(lat_step_deg=2.0, lon_step_deg=4.0,
                          min_elevation_deg=10.0, required_sats=1)
    art = run_coverage(RunConfig(walker=walker, simulation=sim, coverage=cov))

    # Find the northernmost grid point (cell centre is at 89 deg for a 2 deg grid).
    lat_c = art.result.grid.lat_centres_deg
    i = int(np.argmax(lat_c))
    # Any longitude at that latitude works; just pick the first one.
    flat_idx = i * len(art.result.grid.lon_centres_deg)

    period_s = 2.0 * np.pi * np.sqrt((R_EARTH + 550_000.0) ** 3 / MU_EARTH)
    assert np.isfinite(art.result.mean_revisit_s[flat_idx])
    # One time step tolerance.
    np.testing.assert_allclose(art.result.mean_revisit_s[flat_idx], period_s,
                                atol=sim.time_step_s)


def test_mean_global_coverage_equals_cap_fraction_for_single_sat_long_run() -> None:
    # Average the instantaneous global coverage over many orbits.  For a
    # single satellite it must converge to the cap fraction regardless of
    # where the satellite happens to be at any instant.
    walker = WalkerConfig(total_sats=1, num_planes=1, phasing_f=0,
                           altitude_km=550.0, inclination_deg=53.0)
    period_s = 2.0 * np.pi * np.sqrt((R_EARTH + 550_000.0) ** 3 / MU_EARTH)
    sim = SimulationConfig(duration_s=12 * period_s, time_step_s=30.0)
    cov = CoverageConfig(lat_step_deg=1.0, lon_step_deg=1.0,
                          min_elevation_deg=10.0, required_sats=1)
    art = run_coverage(RunConfig(walker=walker, simulation=sim, coverage=cov))

    lam = max_central_angle(550_000.0, np.deg2rad(10.0))
    expected = _spherical_cap_fraction(lam)
    np.testing.assert_allclose(art.result.global_coverage_fraction.mean(),
                                expected, rtol=0.02)
