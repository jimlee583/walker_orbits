"""Orbit propagation.

Only a two-body circular propagator is implemented in Version 1.  The
:class:`Propagator` protocol defines the shape of a future J2 secular or
higher-fidelity propagator so that nothing above this layer needs to change
when one is added.

Vectorized geometry (circular case)
-----------------------------------

For a circular orbit, the inertial position of a satellite with semi-major
axis ``a``, inclination ``i``, RAAN ``Omega`` and argument of latitude
``u(t) = u0 + n * t`` is

.. math::

    \\vec r(t) = a \\begin{bmatrix}
        \\cos\\Omega \\cos u - \\sin\\Omega \\cos i \\sin u \\\\
        \\sin\\Omega \\cos u + \\cos\\Omega \\cos i \\sin u \\\\
        \\sin i \\sin u
    \\end{bmatrix}.

That expression is applied here as pure NumPy broadcasting, in a single
``(N_t, N_sat, 3)`` output array.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .elements import OrbitalElements


class Propagator(Protocol):
    """Minimal propagator interface.

    Implementations propagate an :class:`OrbitalElements` block to a set of
    time samples and return inertial (ECI) positions in metres.
    """

    def propagate(self, elements: OrbitalElements,
                  t_s: np.ndarray) -> np.ndarray:  # pragma: no cover - protocol
        """Return a ``(len(t_s), elements.num_sats, 3)`` ECI position array."""


class CircularTwoBodyPropagator:
    """Analytic two-body propagator for circular orbits.

    This propagator assumes ``e == 0``.  It evaluates all satellites at all
    times with a single vectorized NumPy expression; the cost is
    ``O(N_t * N_sat)`` floating-point operations and one allocation of the
    output array.
    """

    def propagate(self, elements: OrbitalElements, t_s: np.ndarray) -> np.ndarray:
        if np.any(elements.eccentricity > 1e-6):
            raise ValueError(
                "CircularTwoBodyPropagator only supports circular orbits "
                "(eccentricity == 0).  Use a Keplerian propagator for e > 0."
            )

        t = np.asarray(t_s, dtype=np.float64).reshape(-1, 1)       # (N_t, 1)
        a = elements.semi_major_axis_m.reshape(1, -1)              # (1, N_sat)
        inc = elements.inclination_rad.reshape(1, -1)
        raan = elements.raan_rad.reshape(1, -1)
        u0 = elements.arg_of_latitude0_rad().reshape(1, -1)
        n = elements.mean_motion_rad_s().reshape(1, -1)

        u = u0 + n * t                                             # (N_t, N_sat)

        cos_u = np.cos(u)
        sin_u = np.sin(u)
        cos_O = np.cos(raan)
        sin_O = np.sin(raan)
        cos_i = np.cos(inc)
        sin_i = np.sin(inc)

        x = a * (cos_O * cos_u - sin_O * cos_i * sin_u)
        y = a * (sin_O * cos_u + cos_O * cos_i * sin_u)
        z = a * (sin_i * sin_u)

        # Stack along a new last axis -> (N_t, N_sat, 3).  np.stack creates a
        # contiguous array which is optimal for the subsequent matmul.
        return np.stack((x, y, z), axis=-1)
