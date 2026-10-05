"""Tests for the ECI/ECEF rotation and lat/lon utilities."""

from __future__ import annotations

import numpy as np

from walkercov.constants import OMEGA_EARTH, R_EARTH
from walkercov.orbit.coordinates import (
    ecef_to_latlon,
    eci_to_ecef,
    greenwich_angle_rad,
    latlon_to_ecef,
    latlon_to_unit_vector,
)


def test_identity_rotation_at_zero_angle() -> None:
    r_eci = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])[None, :, :]
    theta = np.array([0.0])
    r_ecef = eci_to_ecef(r_eci, theta)
    np.testing.assert_allclose(r_ecef, r_eci)


def test_quarter_sidereal_day_rotation() -> None:
    quarter = (np.pi / 2.0) / OMEGA_EARTH
    r_eci = np.array([[R_EARTH, 0.0, 0.0]])[None, :, :]  # (1, 1, 3)
    theta = greenwich_angle_rad(np.array([quarter]))
    r_ecef = eci_to_ecef(r_eci, theta)
    # +X ECI should rotate to -Y in ECEF for a +Z rotation of +pi/2
    # (eci_to_ecef is R_z(-theta)).
    np.testing.assert_allclose(r_ecef[0, 0], np.array([0.0, -R_EARTH, 0.0]),
                                atol=1e-6)


def test_latlon_roundtrip() -> None:
    rng = np.random.default_rng(0)
    lat = np.deg2rad(rng.uniform(-89.0, 89.0, size=256))
    lon = np.deg2rad(rng.uniform(-179.0, 179.0, size=256))
    r = latlon_to_ecef(lat, lon)
    lat2, lon2 = ecef_to_latlon(r)
    np.testing.assert_allclose(lat2, lat, atol=1e-12)
    np.testing.assert_allclose(lon2, lon, atol=1e-12)


def test_unit_vector_norm() -> None:
    lat = np.deg2rad(np.array([0.0, 30.0, -45.0, 89.0]))
    lon = np.deg2rad(np.array([0.0, 90.0, 180.0, -90.0]))
    u = latlon_to_unit_vector(lat, lon)
    norms = np.linalg.norm(u, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-15)
