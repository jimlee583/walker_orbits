"""Tests for the visibility geometry primitives."""

from __future__ import annotations

import numpy as np

from walkercov.constants import R_EARTH
from walkercov.coverage.visibility import (
    elevation_angle,
    max_central_angle,
    satellite_footprint_radius_km,
)


def test_zenith_satellite_has_90deg_elevation() -> None:
    ground = np.array([R_EARTH, 0.0, 0.0])
    sat = np.array([R_EARTH + 550_000.0, 0.0, 0.0])
    el = elevation_angle(sat, ground)
    np.testing.assert_allclose(np.rad2deg(el), 90.0, atol=1e-10)


def test_tangent_geometry_has_zero_elevation() -> None:
    # Place satellite so its position vector is perpendicular to the ground
    # vector at the max central angle: cos(lambda_max_at_0deg) = R/(R+h).
    h = 550_000.0
    lam = np.arccos(R_EARTH / (R_EARTH + h))
    ground = np.array([R_EARTH, 0.0, 0.0])
    sat = (R_EARTH + h) * np.array([np.cos(lam), np.sin(lam), 0.0])
    el = elevation_angle(sat, ground)
    np.testing.assert_allclose(np.rad2deg(el), 0.0, atol=1e-8)


def test_max_central_angle_matches_elevation_definition() -> None:
    # Place a satellite at exactly lambda_max for a given elevation mask and
    # check the resulting elevation equals the mask.
    h = 550_000.0
    for eps_deg in [0.0, 5.0, 10.0, 25.0]:
        eps = np.deg2rad(eps_deg)
        lam = max_central_angle(h, eps)
        ground = np.array([R_EARTH, 0.0, 0.0])
        sat = (R_EARTH + h) * np.array([np.cos(lam), np.sin(lam), 0.0])
        el = elevation_angle(sat, ground)
        np.testing.assert_allclose(np.rad2deg(el), eps_deg, atol=1e-6)


def test_dot_product_mask_equivalent_to_elevation_mask() -> None:
    rng = np.random.default_rng(42)
    h = 550_000.0
    eps = np.deg2rad(10.0)
    lam = max_central_angle(h, eps)
    cos_lam = np.cos(lam)

    # Random ground points and satellite positions.
    n = 2000
    lat = rng.uniform(-np.pi / 2.0, np.pi / 2.0, size=n)
    lon = rng.uniform(-np.pi, np.pi, size=n)
    ground = R_EARTH * np.stack([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ], axis=-1)
    sat_lat = rng.uniform(-np.pi / 2.0, np.pi / 2.0, size=n)
    sat_lon = rng.uniform(-np.pi, np.pi, size=n)
    sat = (R_EARTH + h) * np.stack([
        np.cos(sat_lat) * np.cos(sat_lon),
        np.cos(sat_lat) * np.sin(sat_lon),
        np.sin(sat_lat),
    ], axis=-1)

    el = elevation_angle(sat, ground)
    mask_el = el >= eps

    # Equivalent dot-product test using unit vectors.
    g_hat = ground / np.linalg.norm(ground, axis=-1, keepdims=True)
    s_hat = sat / np.linalg.norm(sat, axis=-1, keepdims=True)
    cos_angle = np.sum(g_hat * s_hat, axis=-1)
    mask_dot = cos_angle >= cos_lam

    assert np.array_equal(mask_el, mask_dot)


def test_footprint_radius_matches_arclength() -> None:
    h = 550_000.0
    eps = np.deg2rad(10.0)
    lam = max_central_angle(h, eps)
    expected_km = R_EARTH * lam / 1_000.0
    np.testing.assert_allclose(satellite_footprint_radius_km(h, eps),
                                expected_km, rtol=1e-15)
