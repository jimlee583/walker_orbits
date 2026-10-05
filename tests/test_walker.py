"""Walker Delta generator tests."""

from __future__ import annotations

import numpy as np
import pytest

from walkercov.config import WalkerConfig
from walkercov.orbit.walker import satellite_table, valid_f_values, walker_delta


def test_walker_24_6_1_counts_and_spacing() -> None:
    cfg = WalkerConfig(total_sats=24, num_planes=6, phasing_f=1,
                        altitude_km=550.0, inclination_deg=53.0)
    elems = walker_delta(cfg)

    # 24 satellites, 6 planes, 4 per plane.
    assert elems.num_sats == 24
    assert len(np.unique(elems.plane_id)) == 6
    _, counts = np.unique(elems.plane_id, return_counts=True)
    assert (counts == 4).all()

    # RAAN step = 60 deg.
    raans_deg = np.rad2deg(elems.raan_rad[::4])
    diffs = np.diff(raans_deg)
    np.testing.assert_allclose(diffs, 60.0, atol=1e-9)

    # In-plane spacing = 90 deg.
    first_plane = np.rad2deg(elems.arg_of_latitude0_rad()[elems.plane_id == 0])
    first_plane_sorted = np.sort(first_plane)
    np.testing.assert_allclose(np.diff(first_plane_sorted), 90.0, atol=1e-9)

    # Walker phase increment between adjacent planes = F * 360 / T = 15 deg.
    u_slot0 = np.rad2deg(elems.arg_of_latitude0_rad()[elems.slot_id == 0])
    plane_of_slot0 = elems.plane_id[elems.slot_id == 0]
    order = np.argsort(plane_of_slot0)
    u_ordered = u_slot0[order]
    diffs = np.mod(np.diff(u_ordered), 360.0)
    np.testing.assert_allclose(diffs, 15.0, atol=1e-9)


def test_walker_validation_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        WalkerConfig(total_sats=25, num_planes=6, phasing_f=1,
                      altitude_km=550.0, inclination_deg=53.0)
    with pytest.raises(ValueError):
        WalkerConfig(total_sats=24, num_planes=6, phasing_f=6,
                      altitude_km=550.0, inclination_deg=53.0)
    with pytest.raises(ValueError):
        WalkerConfig(total_sats=24, num_planes=6, phasing_f=-1,
                      altitude_km=550.0, inclination_deg=53.0)
    with pytest.raises(ValueError):
        WalkerConfig(total_sats=24, num_planes=6, phasing_f=0,
                      altitude_km=0.0, inclination_deg=53.0)


def test_valid_f_values() -> None:
    assert valid_f_values(6) == [0, 1, 2, 3, 4, 5]
    assert valid_f_values(1) == [0]
    with pytest.raises(ValueError):
        valid_f_values(0)


def test_walker_table_structure() -> None:
    cfg = WalkerConfig(total_sats=24, num_planes=6, phasing_f=1,
                        altitude_km=550.0, inclination_deg=53.0)
    df = satellite_table(cfg, walker_delta(cfg))
    assert list(df.columns) == [
        "sat_id", "plane", "slot", "raan_deg", "arg_latitude0_deg",
        "altitude_km", "inclination_deg", "period_min",
    ]
    assert df.shape[0] == 24
    assert df["sat_id"].is_unique


def test_walker_f_and_f_plus_p_are_equivalent_up_to_relabelling() -> None:
    # F and F+P produce the same geometry; the UI simply restricts F to [0, P-1].
    cfg = WalkerConfig(total_sats=24, num_planes=6, phasing_f=1,
                        altitude_km=550.0, inclination_deg=53.0)
    e = walker_delta(cfg)
    u1 = np.mod(np.rad2deg(e.arg_of_latitude0_rad()), 360.0)
    # Manually emulate "F=1+6=7" by applying extra 360 deg per plane and
    # wrapping; it should land on the same per-plane satellite set.
    u7 = np.mod(u1 + e.plane_id * 6 * 360.0 / 24.0, 360.0)
    for p in range(6):
        s1 = np.sort(u1[e.plane_id == p])
        s7 = np.sort(u7[e.plane_id == p])
        np.testing.assert_allclose(s1, s7, atol=1e-9)
