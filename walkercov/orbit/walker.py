"""Walker Delta constellation generator.

Walker Delta ``T/P/F`` convention used in this package
------------------------------------------------------

For a constellation of ``T`` satellites in ``P`` evenly spaced circular orbital
planes, with ``S = T / P`` satellites per plane:

* Plane index :math:`p \\in \\{0, 1, \\dots, P-1\\}`.
* Slot index  :math:`s \\in \\{0, 1, \\dots, S-1\\}`.
* RAAN of plane :math:`p`:

    .. math::  \\Omega_p = p \\cdot \\frac{360^\\circ}{P}

* Argument of latitude of satellite :math:`(p, s)` at :math:`t = 0`:

    .. math::  u_{p,s}(0) = s \\cdot \\frac{360^\\circ}{S}
                         + p \\cdot F \\cdot \\frac{360^\\circ}{T}
                         + u_0

  where :math:`F \\in \\{0, 1, \\dots, P-1\\}` is the Walker phasing parameter
  and :math:`u_0` is an optional overall offset (``arg_of_latitude0_deg``).

This matches the convention in Walker's original 1984 paper and in Vallado
(and matches STK's "Walker Delta" seed definition).  ``F`` and ``F + P`` give
identical constellations, so the UI restricts ``F`` to ``[0, P-1]``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import WalkerConfig
from ..constants import R_EARTH
from .elements import OrbitalElements


def walker_delta(config: WalkerConfig) -> OrbitalElements:
    """Build the :class:`OrbitalElements` for a Walker Delta constellation.

    The satellites are returned in plane-major, slot-minor order so that the
    first ``S`` satellites belong to plane 0, the next ``S`` to plane 1, and so
    on.
    """

    T = config.total_sats
    P = config.num_planes
    S = config.sats_per_plane
    F = config.phasing_f

    # Plane index p and in-plane slot index s for each satellite.
    plane_id = np.repeat(np.arange(P, dtype=np.int32), S)
    slot_id = np.tile(np.arange(S, dtype=np.int32), P)

    # Degrees -> radians at the boundary.
    raan_deg = plane_id * (360.0 / P)
    # Walker phasing: each successive plane's slot 0 is advanced by
    # F * 360 / T degrees in argument of latitude relative to the previous
    # plane's slot 0.
    u0_deg = (
        slot_id * (360.0 / S)
        + plane_id * (F * 360.0 / T)
        + config.arg_of_latitude0_deg
    )
    u0_deg = np.mod(u0_deg, 360.0)

    raan_rad = np.deg2rad(raan_deg)
    u0_rad = np.deg2rad(u0_deg)

    inc_rad = np.full(T, np.deg2rad(config.inclination_deg), dtype=np.float64)
    semi_major_axis_m = np.full(T, R_EARTH + config.altitude_km * 1_000.0,
                                dtype=np.float64)
    ecc = np.zeros(T, dtype=np.float64)

    # Circular orbit: fold the entire initial phase into mean anomaly, leave
    # argument of perigee zero.  u0 = omega + M0, so u(t) = u0 + n*t.
    arg_perigee_rad = np.zeros(T, dtype=np.float64)
    mean_anomaly0_rad = u0_rad.copy()

    return OrbitalElements(
        semi_major_axis_m=semi_major_axis_m,
        eccentricity=ecc,
        inclination_rad=inc_rad,
        raan_rad=raan_rad,
        arg_perigee_rad=arg_perigee_rad,
        mean_anomaly0_rad=mean_anomaly0_rad,
        plane_id=plane_id,
        slot_id=slot_id,
        shell_id=np.zeros(T, dtype=np.int32),
    )


def satellite_table(config: WalkerConfig, elements: OrbitalElements) -> pd.DataFrame:
    """Return a human-readable table of satellite initial conditions."""

    T = elements.num_sats
    sat_ids = [f"W{config.total_sats}/{config.num_planes}/{config.phasing_f}-"
               f"P{p:02d}S{s:02d}" for p, s in zip(elements.plane_id,
                                                   elements.slot_id)]
    df = pd.DataFrame({
        "sat_id": sat_ids,
        "plane": elements.plane_id.astype(int),
        "slot": elements.slot_id.astype(int),
        "raan_deg": np.rad2deg(elements.raan_rad),
        "arg_latitude0_deg": np.rad2deg(elements.arg_of_latitude0_rad()),
        "altitude_km": (elements.semi_major_axis_m - R_EARTH) / 1_000.0,
        "inclination_deg": np.rad2deg(elements.inclination_rad),
        "period_min": elements.period_s() / 60.0,
    })
    df.index = pd.RangeIndex(T, name="sat_index")
    return df


def valid_f_values(num_planes: int) -> list[int]:
    """List of valid Walker phasing values F for the given number of planes."""

    if num_planes <= 0:
        raise ValueError("num_planes must be positive")
    return list(range(num_planes))
