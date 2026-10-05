"""Container for a set of satellite orbital elements.

Six classical Keplerian elements are stored (a, e, i, RAAN, argument of
perigee, mean anomaly at epoch) along with bookkeeping (``plane_id``,
``slot_id``, ``shell_id``).  Even though Version 1 only propagates circular
orbits, keeping the full element set here means a J2 secular propagator or a
multi-shell Walker Star generator can be dropped in with no refactor.

All angles are stored in **radians**.  All distances in **metres**.  All
arrays are 1-D with length equal to the number of satellites and are organised
in a consistent satellite order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..constants import MU_EARTH


@dataclass(frozen=True)
class OrbitalElements:
    """Vectorized orbital element block for a constellation.

    Attributes
    ----------
    semi_major_axis_m : (N,) float array
    eccentricity : (N,) float array
    inclination_rad : (N,) float array
    raan_rad : (N,) float array
    arg_perigee_rad : (N,) float array
    mean_anomaly0_rad : (N,) float array
        Mean anomaly at ``t = 0``.  For circular orbits the "argument of
        latitude" at epoch is ``arg_perigee_rad + mean_anomaly0_rad`` and that
        is what the Walker generator populates.
    plane_id : (N,) int array
        Zero-based plane index.
    slot_id : (N,) int array
        Zero-based satellite index within the plane.
    shell_id : (N,) int array
        Zero-based shell index.  Always zero for a single-shell Walker.
    """

    semi_major_axis_m: np.ndarray
    eccentricity: np.ndarray
    inclination_rad: np.ndarray
    raan_rad: np.ndarray
    arg_perigee_rad: np.ndarray
    mean_anomaly0_rad: np.ndarray
    plane_id: np.ndarray
    slot_id: np.ndarray
    shell_id: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        n = self.semi_major_axis_m.shape[0]
        for name, arr in (
            ("eccentricity", self.eccentricity),
            ("inclination_rad", self.inclination_rad),
            ("raan_rad", self.raan_rad),
            ("arg_perigee_rad", self.arg_perigee_rad),
            ("mean_anomaly0_rad", self.mean_anomaly0_rad),
            ("plane_id", self.plane_id),
            ("slot_id", self.slot_id),
        ):
            if arr.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
        if self.shell_id is not None and self.shell_id.shape != (n,):
            raise ValueError("shell_id must have shape (N,)")

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def num_sats(self) -> int:
        return int(self.semi_major_axis_m.shape[0])

    def mean_motion_rad_s(self) -> np.ndarray:
        """Mean motion n = sqrt(mu / a^3) [rad/s] for each satellite."""

        a = self.semi_major_axis_m
        return np.sqrt(MU_EARTH / (a * a * a))

    def period_s(self) -> np.ndarray:
        """Orbital period T = 2*pi / n [s] for each satellite."""

        return 2.0 * np.pi / self.mean_motion_rad_s()

    def arg_of_latitude0_rad(self) -> np.ndarray:
        """Argument of latitude at epoch, u0 = omega + M0 (valid for e=0)."""

        return self.arg_perigee_rad + self.mean_anomaly0_rad
