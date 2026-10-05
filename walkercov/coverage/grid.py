"""Latitude/longitude ground grid with proper equal-area weighting.

A naive lat/lon grid treats every (lat, lon) cell as if it had the same area,
which heavily over-weights the poles.  The weight of a cell spanning
latitudes :math:`\\phi_s..\\phi_n` and longitudes :math:`\\lambda_w..\\lambda_e`
on a unit sphere is

.. math::  w = (\\sin\\phi_n - \\sin\\phi_s)(\\lambda_e - \\lambda_w),

summing to :math:`4\\pi` over the whole sphere.  We store the weight as a
fraction of the (possibly regional) area so that any weighted mean yields a
percentage in :math:`[0, 1]`.

Grid points are placed at the **centre** of each cell.  This avoids the
degenerate behaviour at the poles that comes from grids that include
:math:`\\phi = \\pm 90^\\circ` directly, and matches the standard way
area-weighted global averages are computed in meteorology and remote sensing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import CoverageConfig
from .. import orbit  # noqa: F401 - makes it clear which package the sibling is
from ..orbit.coordinates import latlon_to_unit_vector


@dataclass(frozen=True)
class GroundGrid:
    """A latitude/longitude grid with precomputed ECEF unit vectors.

    Attributes
    ----------
    lat_centres_deg, lon_centres_deg : 1-D float arrays
        Cell-centre coordinates in degrees.
    lat_rad, lon_rad : 1-D float arrays flattened over the grid [N_lat*N_lon]
        Cell-centre coordinates in radians, in row-major order
        (lat is the slower-varying axis).
    unit_vectors : (N_points, 3) float64 array
        ECEF unit vectors of each grid point (``r`` = 1).  Multiplying by a
        satellite ECEF unit vector gives ``cos(central angle)``.
    area_weights : (N_points,) float64 array
        Normalised cell areas.  They sum to 1 over the covered region, so
        ``(flag * area_weights).sum()`` directly gives the covered area
        fraction.
    cos_lat : (N_points,) float64 array
        Cosine of latitude at each point.  Convenient for latitude banding.
    shape : (N_lat, N_lon) tuple
        Shape of the 2-D grid for reshaping metric arrays before plotting.
    """

    lat_centres_deg: np.ndarray
    lon_centres_deg: np.ndarray
    lat_rad: np.ndarray
    lon_rad: np.ndarray
    unit_vectors: np.ndarray
    area_weights: np.ndarray
    cos_lat: np.ndarray
    shape: tuple[int, int]

    @property
    def num_points(self) -> int:
        return int(self.unit_vectors.shape[0])


def build_grid(config: CoverageConfig) -> GroundGrid:
    """Construct a :class:`GroundGrid` from a :class:`CoverageConfig`."""

    lat_lo, lat_hi = config.lat_bounds
    lon_lo, lon_hi = config.lon_bounds

    # Choose cell counts that evenly tile the requested span.
    n_lat = max(1, int(round((lat_hi - lat_lo) / config.lat_step_deg)))
    n_lon = max(1, int(round((lon_hi - lon_lo) / config.lon_step_deg)))
    dlat = (lat_hi - lat_lo) / n_lat
    dlon = (lon_hi - lon_lo) / n_lon

    lat_edges = lat_lo + np.arange(n_lat + 1) * dlat
    lon_edges = lon_lo + np.arange(n_lon + 1) * dlon
    lat_centres = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_centres = 0.5 * (lon_edges[:-1] + lon_edges[1:])

    # Band areas in units of steradian per 2*pi of longitude: sin(north) - sin(south).
    sin_edges = np.sin(np.deg2rad(lat_edges))
    band_sin = sin_edges[1:] - sin_edges[:-1]            # (n_lat,)
    lon_frac = np.full(n_lon, dlon / (lon_hi - lon_lo))  # (n_lon,)

    # 2-D grid in row-major (lat, lon) order.
    lat_grid_deg, lon_grid_deg = np.meshgrid(lat_centres, lon_centres,
                                             indexing="ij")
    lat_rad = np.deg2rad(lat_grid_deg).ravel()
    lon_rad = np.deg2rad(lon_grid_deg).ravel()
    unit_vectors = latlon_to_unit_vector(lat_rad, lon_rad).astype(np.float64)
    unit_vectors = np.ascontiguousarray(unit_vectors)

    # Area weights normalised so that they sum to 1 over the region.
    weight_2d = np.outer(band_sin, lon_frac)
    weight_2d /= weight_2d.sum()
    area_weights = weight_2d.ravel()

    cos_lat = np.cos(lat_rad)

    return GroundGrid(
        lat_centres_deg=lat_centres,
        lon_centres_deg=lon_centres,
        lat_rad=lat_rad,
        lon_rad=lon_rad,
        unit_vectors=unit_vectors,
        area_weights=area_weights,
        cos_lat=cos_lat,
        shape=(n_lat, n_lon),
    )


def latitude_band_weights(grid: GroundGrid) -> np.ndarray:
    """Return the area-weight sum per latitude band, shape ``(n_lat,)``."""

    weights_2d = grid.area_weights.reshape(grid.shape)
    return weights_2d.sum(axis=1)
