"""Tests for the circular two-body propagator."""

from __future__ import annotations

import numpy as np
import pytest

from walkercov.config import WalkerConfig
from walkercov.constants import MU_EARTH, OMEGA_EARTH, R_EARTH, SIDEREAL_DAY_S
from walkercov.orbit.propagator import CircularTwoBodyPropagator
from walkercov.orbit.walker import walker_delta


def test_period_matches_formula_at_550km() -> None:
    cfg = WalkerConfig(total_sats=1, num_planes=1, phasing_f=0,
                        altitude_km=550.0, inclination_deg=53.0)
    elems = walker_delta(cfg)
    a = R_EARTH + 550_000.0
    expected = 2.0 * np.pi * np.sqrt(a ** 3 / MU_EARTH)
    np.testing.assert_allclose(elems.period_s()[0], expected, rtol=0, atol=1e-6)
    # Numerical value is about 95.63 min.
    assert 95.0 * 60.0 < expected < 96.0 * 60.0


def test_geo_period_matches_sidereal_day() -> None:
    # a = (mu / omega_e^2)^(1/3) gives geo semi-major axis.
    a_geo = (MU_EARTH / OMEGA_EARTH ** 2) ** (1.0 / 3.0)
    alt = (a_geo - R_EARTH) / 1_000.0
    cfg = WalkerConfig(total_sats=1, num_planes=1, phasing_f=0,
                        altitude_km=alt, inclination_deg=0.0)
    elems = walker_delta(cfg)
    np.testing.assert_allclose(elems.period_s()[0], SIDEREAL_DAY_S,
                                rtol=1e-6)


def test_circular_norm_is_constant_and_orthogonal_to_normal() -> None:
    cfg = WalkerConfig(total_sats=4, num_planes=2, phasing_f=0,
                        altitude_km=550.0, inclination_deg=53.0)
    elems = walker_delta(cfg)
    prop = CircularTwoBodyPropagator()
    t = np.linspace(0.0, 3.0 * elems.period_s()[0], 97)
    r = prop.propagate(elems, t)

    # Radius is constant.
    radii = np.linalg.norm(r, axis=-1)
    expected = elems.semi_major_axis_m
    for i in range(elems.num_sats):
        np.testing.assert_allclose(radii[:, i], expected[i], rtol=1e-12)

    # Orbit normals are perpendicular to r (and constant).
    normals = np.cross(r[0], r[1])
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    dots = np.einsum("tnc,nc->tn", r, normals)
    np.testing.assert_allclose(dots, 0.0, atol=1e-6)


def test_propagator_rejects_eccentric_orbits() -> None:
    cfg = WalkerConfig(total_sats=1, num_planes=1, phasing_f=0,
                        altitude_km=550.0, inclination_deg=53.0)
    elems = walker_delta(cfg)
    # Hand-mutate eccentricity to a value that should trigger the guard.
    elems.eccentricity[0] = 0.1
    with pytest.raises(ValueError):
        CircularTwoBodyPropagator().propagate(elems, np.array([0.0, 60.0]))
