"""Coordinate frames and geodesy helpers.

The package uses two frames:

* **ECI** (Earth-centred inertial): the frame the propagator works in.
* **ECEF** (Earth-centred, Earth-fixed): rotated from ECI about +Z by the
  Greenwich sidereal angle :math:`\\theta(t)`.

Since no absolute epoch is assumed, :math:`\\theta(t) = \\theta_0 + \\omega_E t`
with :math:`\\theta_0` taken from the config.  All angles are radians and all
distances metres.

Geodesy uses a spherical Earth of radius ``R_EARTH``.  Latitude/longitude pairs
are therefore geocentric.  The approximation is appropriate for coverage
analysis using an elevation-angle mask; adding an ellipsoidal model later only
requires changing :func:`latlon_to_ecef`.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..constants import OMEGA_EARTH, R_EARTH


# ---------------------------------------------------------------------------
# Earth rotation
# ---------------------------------------------------------------------------


def greenwich_angle_rad(t_s: np.ndarray, theta0_rad: float = 0.0) -> np.ndarray:
    """Return the Greenwich sidereal angle at each time sample [rad]."""

    return theta0_rad + OMEGA_EARTH * np.asarray(t_s, dtype=np.float64)


def eci_to_ecef(r_eci: np.ndarray, theta_rad: np.ndarray) -> np.ndarray:
    """Rotate ``r_eci`` into ECEF using a scalar/vector Greenwich angle.

    Parameters
    ----------
    r_eci : (N_t, N_sat, 3) or (N_t, 3) array
        Inertial positions in metres.
    theta_rad : (N_t,) array
        Greenwich sidereal angle at each epoch.
    """

    r = np.asarray(r_eci, dtype=np.float64)
    theta = np.asarray(theta_rad, dtype=np.float64)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    if r.ndim == 3:
        # Broadcast (N_t,) against (N_t, N_sat).
        cos_t = cos_t[:, None]
        sin_t = sin_t[:, None]

    x_i = r[..., 0]
    y_i = r[..., 1]
    z_i = r[..., 2]
    # Rotation about +Z by -theta (ECEF frame leads ECI by theta).
    x_e = cos_t * x_i + sin_t * y_i
    y_e = -sin_t * x_i + cos_t * y_i
    z_e = z_i
    return np.stack((x_e, y_e, z_e), axis=-1)


# ---------------------------------------------------------------------------
# Latitude / longitude <-> ECEF
# ---------------------------------------------------------------------------


def latlon_to_unit_vector(lat_rad: np.ndarray,
                           lon_rad: np.ndarray) -> np.ndarray:
    """Return the ECEF unit vector of a geocentric lat/lon [radians] pair.

    Output shape matches the broadcast of the inputs with a trailing length-3
    axis appended.
    """

    lat = np.asarray(lat_rad, dtype=np.float64)
    lon = np.asarray(lon_rad, dtype=np.float64)
    cos_lat = np.cos(lat)
    x = cos_lat * np.cos(lon)
    y = cos_lat * np.sin(lon)
    z = np.sin(lat)
    return np.stack((x, y, z), axis=-1)


def latlon_to_ecef(lat_rad: np.ndarray,
                    lon_rad: np.ndarray,
                    altitude_m: float = 0.0) -> np.ndarray:
    """Convert spherical lat/lon to an ECEF position vector [m]."""

    r = R_EARTH + altitude_m
    return r * latlon_to_unit_vector(lat_rad, lon_rad)


def ecef_to_latlon(r_ecef: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert ECEF positions to geocentric latitude and longitude [radians]."""

    r = np.asarray(r_ecef, dtype=np.float64)
    radius = np.linalg.norm(r, axis=-1)
    lat = np.arcsin(np.clip(r[..., 2] / radius, -1.0, 1.0))
    lon = np.arctan2(r[..., 1], r[..., 0])
    return lat, lon


# ---------------------------------------------------------------------------
# Sub-satellite point utility (handy for ground tracks)
# ---------------------------------------------------------------------------


def sub_satellite_latlon_deg(r_ecef: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return sub-satellite (lat, lon) in degrees from ECEF positions."""

    lat, lon = ecef_to_latlon(r_ecef)
    return np.rad2deg(lat), np.rad2deg(lon)
