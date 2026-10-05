"""Physical and Earth constants used by the propagator and coverage math.

All quantities are expressed in SI units (metres, seconds, radians) unless an
API explicitly says otherwise.  Keeping units consistent inside the package
avoids accidental scale errors in the vectorized math; unit conversions are
performed only at the UI boundary.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Earth gravity / shape
# ---------------------------------------------------------------------------

#: Earth gravitational parameter mu = G * M_earth [m^3 / s^2].  WGS-84 value.
MU_EARTH: float = 3.986004418e14

#: Mean equatorial Earth radius [m].  WGS-84 value.  The coverage model uses a
#: spherical Earth of this radius; altitudes are referenced to it.
R_EARTH: float = 6378137.0

#: Second zonal harmonic coefficient.  Not used in the Version 1 propagator,
#: but defined here so a future J2 secular propagator has a single source of
#: truth.
J2_EARTH: float = 1.08262668e-3

# ---------------------------------------------------------------------------
# Earth rotation
# ---------------------------------------------------------------------------

#: Earth sidereal rotation rate [rad / s].  This is the inertial rotation rate
#: and is slightly larger than 2*pi / 86400.  Using the sidereal rate matters
#: whenever a satellite's ECEF ground track is needed over hours or longer.
OMEGA_EARTH: float = 7.2921150e-5

#: Length of a mean sidereal day [s], derived from OMEGA_EARTH.
SIDEREAL_DAY_S: float = 2.0 * math.pi / OMEGA_EARTH

# ---------------------------------------------------------------------------
# Unit helpers (kept as plain functions so they inline nicely in NumPy code).
# ---------------------------------------------------------------------------


def km_to_m(value_km: float) -> float:
    """Convert kilometres to metres."""

    return value_km * 1_000.0


def m_to_km(value_m: float) -> float:
    """Convert metres to kilometres."""

    return value_m / 1_000.0
