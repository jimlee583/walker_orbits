"""Satellite-to-ground visibility geometry.

For a spherical Earth of radius :math:`R_E` and a satellite at altitude
:math:`h`, a ground point sees the satellite above elevation :math:`\\varepsilon`
if and only if the Earth-centre angle between the two unit vectors satisfies

.. math::

    \\lambda \\le \\lambda_{max}(h, \\varepsilon)
          = \\arccos\\!\\left( \\frac{R_E}{R_E + h} \\cos\\varepsilon \\right)
            - \\varepsilon .

That identity is a direct consequence of the planar triangle
(ground point, Earth centre, satellite) and the law of sines; it lets us
replace the per-point elevation calculation with a single dot product per
(satellite, ground point) pair.  :func:`elevation_angle` is still provided for
verification and for cases where the explicit angle is useful (for example,
rendering).

The counts kernel :func:`visible_counts_chunked` is the hot path.  For each
epoch chunk it:

1. computes satellite ECEF positions,
2. reduces them to unit vectors,
3. multiplies by the (G, 3) ground unit matrix,
4. compares against ``cos(lambda_max)``,
5. sums along the satellite axis to produce ``(T_chunk, G)`` visibility counts
   stored as ``uint16``.

This moves the inner loop into BLAS and keeps peak memory to one chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import numpy as np

from ..config import RunConfig
from ..constants import R_EARTH
from ..orbit.coordinates import eci_to_ecef, greenwich_angle_rad
from ..orbit.elements import OrbitalElements
from ..orbit.propagator import Propagator
from .grid import GroundGrid


# ---------------------------------------------------------------------------
# Analytic geometry
# ---------------------------------------------------------------------------


def max_central_angle(altitude_m: float | np.ndarray,
                       min_elevation_rad: float) -> float | np.ndarray:
    """Return :math:`\\lambda_{max}` [rad] for a given altitude and elevation mask."""

    r = R_EARTH + np.asarray(altitude_m, dtype=np.float64)
    cos_arg = (R_EARTH / r) * np.cos(min_elevation_rad)
    return np.arccos(np.clip(cos_arg, -1.0, 1.0)) - min_elevation_rad


def elevation_angle(sat_ecef: np.ndarray,
                     ground_ecef: np.ndarray) -> np.ndarray:
    """Return the elevation angle [rad] of a satellite at a ground point.

    Shapes broadcast as usual.  The explicit formula is
    :math:`\\sin\\varepsilon = \\hat u \\cdot \\vec\\rho / |\\vec\\rho|`
    where :math:`\\hat u` is the local vertical and :math:`\\vec\\rho` is the
    satellite line-of-sight vector.
    """

    rho = sat_ecef - ground_ecef
    rho_norm = np.linalg.norm(rho, axis=-1)
    u_hat = ground_ecef / np.linalg.norm(ground_ecef, axis=-1, keepdims=True)
    sin_el = np.sum(u_hat * rho, axis=-1) / rho_norm
    return np.arcsin(np.clip(sin_el, -1.0, 1.0))


def satellite_footprint_radius_km(altitude_m: float,
                                   min_elevation_rad: float) -> float:
    """Great-circle footprint radius on the ground [km]."""

    lam = max_central_angle(altitude_m, min_elevation_rad)
    return float(R_EARTH * lam / 1_000.0)


# ---------------------------------------------------------------------------
# Chunked counts kernel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VisibilityChunk:
    """One chunk of per-epoch satellite visibility counts."""

    t_start_index: int
    t_s: np.ndarray                  # (T_chunk,) float64 seconds
    counts: np.ndarray               # (T_chunk, G) uint16 - satellite counts
    instantaneous_covered: np.ndarray  # (T_chunk,) float64 area-weighted fraction


def _chunk_sizes(num_steps: int, num_sats: int, num_points: int,
                 max_floats: int) -> list[tuple[int, int]]:
    """Choose epoch chunk boundaries so each chunk fits in ``max_floats``."""

    # Peak float count per chunk ~ T_chunk * (num_sats * 3  + num_points).
    per_step = num_sats * 3 + num_points
    chunk = max(1, int(max_floats // max(1, per_step)))
    chunk = min(chunk, num_steps)

    boundaries: list[tuple[int, int]] = []
    i = 0
    while i < num_steps:
        j = min(i + chunk, num_steps)
        boundaries.append((i, j))
        i = j
    return boundaries


def iter_visibility_chunks(
    config: RunConfig,
    elements: OrbitalElements,
    grid: GroundGrid,
    propagator: Propagator,
    *,
    progress: Optional[Callable[[float], None]] = None,
) -> Iterator[VisibilityChunk]:
    """Yield :class:`VisibilityChunk` objects covering the full simulation.

    This is the single hot loop in the pipeline.  Each chunk:

    * propagates the constellation,
    * rotates ECI -> ECEF,
    * normalises satellite positions to unit vectors,
    * computes ``counts[t, g] = sum_sat (sat_unit . ground_unit >= cos_lambda)``.
    """

    sim = config.simulation
    cov = config.coverage

    num_steps = sim.num_steps
    num_sats = elements.num_sats
    num_points = grid.num_points

    # Per-satellite lambda_max (one value today; vector keeps the door open
    # for multi-shell or sensor-cone future work).
    altitudes = elements.semi_major_axis_m - R_EARTH
    lam_max = max_central_angle(altitudes, np.deg2rad(cov.min_elevation_deg))
    cos_lam_max = np.cos(lam_max).astype(np.float64)   # (N_sat,)

    required = int(cov.required_sats)
    theta0 = np.deg2rad(sim.greenwich_angle0_deg)

    boundaries = _chunk_sizes(num_steps, num_sats, num_points,
                              config.chunk_max_floats)

    ground_unit = grid.unit_vectors                     # (G, 3) float64
    ground_unit_T = np.ascontiguousarray(ground_unit.T)  # (3, G) for matmul

    done_steps = 0
    for (i0, i1) in boundaries:
        t_chunk = (np.arange(i0, i1, dtype=np.float64) * sim.time_step_s)
        r_eci = propagator.propagate(elements, t_chunk)             # (T, N, 3)
        theta = greenwich_angle_rad(t_chunk, theta0)                # (T,)
        r_ecef = eci_to_ecef(r_eci, theta)                           # (T, N, 3)

        # Unit vectors (preserves per-satellite altitude on the dot product).
        r_norm = np.linalg.norm(r_ecef, axis=-1, keepdims=True)
        sat_unit = r_ecef / r_norm                                   # (T, N, 3)

        # cos(central angle) = sat_unit . ground_unit, shape (T, N, G).
        # Reshape to (T*N, 3) @ (3, G) for a single BLAS call.
        T_chunk = i1 - i0
        cos_angle = sat_unit.reshape(T_chunk * num_sats, 3) @ ground_unit_T
        cos_angle = cos_angle.reshape(T_chunk, num_sats, num_points)

        visible = cos_angle >= cos_lam_max[None, :, None]            # bool
        counts = visible.sum(axis=1, dtype=np.uint16)                # (T, G)

        covered = counts >= required                                  # bool
        inst_fraction = covered.astype(np.float64) @ grid.area_weights

        yield VisibilityChunk(
            t_start_index=i0,
            t_s=t_chunk,
            counts=counts,
            instantaneous_covered=inst_fraction,
        )

        done_steps += T_chunk
        if progress is not None:
            progress(done_steps / num_steps)
