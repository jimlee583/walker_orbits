"""Immutable, hashable configuration objects used throughout the pipeline.

These dataclasses carry the user's choices (constellation, simulation window,
coverage definition) from the Streamlit UI into the analysis layer.  They are
``frozen`` and ``slots``-friendly so that ``st.cache_data`` can key its memoised
results on the complete input state.

All angles are stored in degrees in the config layer because that is what the
UI exposes; the analysis layer converts to radians at the boundary.  All
distances are stored in metres and all times in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


# ---------------------------------------------------------------------------
# Walker constellation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkerConfig:
    """Walker Delta T/P/F definition plus the common circular-orbit parameters.

    Parameters
    ----------
    total_sats : int
        Total number of satellites ``T``.
    num_planes : int
        Number of orbital planes ``P``.  Must divide ``T`` evenly.
    phasing_f : int
        Walker phasing parameter ``F``, valid in ``0..P-1``.
    altitude_km : float
        Circular orbit altitude above the spherical Earth, in kilometres.
    inclination_deg : float
        Common orbital inclination of every plane, in degrees (``0 <= i <= 180``).
    arg_of_latitude0_deg : float, optional
        Argument of latitude of the first satellite of the first plane at
        ``t = 0``.  Defaults to zero.  Shifting this value rotates the whole
        constellation; it does not affect the coverage statistics.
    """

    total_sats: int
    num_planes: int
    phasing_f: int
    altitude_km: float
    inclination_deg: float
    arg_of_latitude0_deg: float = 0.0

    def __post_init__(self) -> None:  # noqa: D401 - validation
        if self.total_sats <= 0:
            raise ValueError("total_sats must be positive")
        if self.num_planes <= 0:
            raise ValueError("num_planes must be positive")
        if self.total_sats % self.num_planes != 0:
            raise ValueError(
                f"total_sats ({self.total_sats}) must be divisible by "
                f"num_planes ({self.num_planes})"
            )
        if not (0 <= self.phasing_f < self.num_planes):
            raise ValueError(
                f"phasing_f ({self.phasing_f}) must satisfy "
                f"0 <= F < P ({self.num_planes})"
            )
        if self.altitude_km <= 0.0:
            raise ValueError("altitude_km must be positive")
        if not (0.0 <= self.inclination_deg <= 180.0):
            raise ValueError("inclination_deg must be in [0, 180]")

    @property
    def sats_per_plane(self) -> int:
        return self.total_sats // self.num_planes

    @property
    def raan_step_deg(self) -> float:
        """RAAN increment between adjacent planes, in degrees."""

        return 360.0 / self.num_planes

    @property
    def in_plane_step_deg(self) -> float:
        """Argument-of-latitude spacing of satellites within one plane."""

        return 360.0 / self.sats_per_plane

    @property
    def phase_increment_deg(self) -> float:
        """Walker phase increment between adjacent planes = F * 360 / T."""

        return self.phasing_f * 360.0 / self.total_sats


# ---------------------------------------------------------------------------
# Simulation window
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulationConfig:
    """Time window over which the constellation is propagated.

    Parameters
    ----------
    duration_s : float
        Total simulated duration in seconds.
    time_step_s : float
        Propagation / sampling step in seconds.  Coverage events are only
        resolved to within one time step.
    greenwich_angle0_deg : float
        Greenwich sidereal angle at ``t = 0`` in degrees.  Since no absolute
        epoch is specified, this just defines where the sub-satellite longitude
        pattern begins on the map at ``t = 0``.  Changing it shifts the maps in
        longitude but does not change the statistics.
    """

    duration_s: float
    time_step_s: float
    greenwich_angle0_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive")
        if self.time_step_s <= 0.0:
            raise ValueError("time_step_s must be positive")
        if self.time_step_s > self.duration_s:
            raise ValueError("time_step_s must be <= duration_s")

    @property
    def num_steps(self) -> int:
        """Number of time samples including both endpoints."""

        return int(round(self.duration_s / self.time_step_s)) + 1


# ---------------------------------------------------------------------------
# Coverage geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoverageConfig:
    """Ground grid and visibility definition.

    Parameters
    ----------
    lat_step_deg, lon_step_deg : float
        Cell size of the latitude/longitude grid.  Cell centres are used, so a
        2 deg grid has 90 x 180 = 16200 points.
    min_elevation_deg : float
        Minimum elevation angle for a ground point to be considered covered by
        a satellite.
    required_sats : int
        Number of simultaneously visible satellites required to count as
        "covered" (k-fold coverage).  Defaults to 1.
    lat_bounds, lon_bounds : (float, float)
        Latitude and longitude bounds in degrees.  Defaults to the whole
        sphere.  Narrower bounds let the engine do a cheaper regional analysis
        later.
    """

    lat_step_deg: float = 2.0
    lon_step_deg: float = 2.0
    min_elevation_deg: float = 10.0
    required_sats: int = 1
    lat_bounds: Tuple[float, float] = (-90.0, 90.0)
    lon_bounds: Tuple[float, float] = (-180.0, 180.0)

    def __post_init__(self) -> None:
        if self.lat_step_deg <= 0.0 or self.lon_step_deg <= 0.0:
            raise ValueError("grid step must be positive")
        if not (-90.0 <= self.lat_bounds[0] < self.lat_bounds[1] <= 90.0):
            raise ValueError("invalid lat_bounds")
        if not (-180.0 <= self.lon_bounds[0] < self.lon_bounds[1] <= 180.0):
            raise ValueError("invalid lon_bounds")
        if not (-90.0 <= self.min_elevation_deg < 90.0):
            raise ValueError("min_elevation_deg must be in [-90, 90)")
        if self.required_sats < 1:
            raise ValueError("required_sats must be >= 1")


# ---------------------------------------------------------------------------
# Convenience bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    """Full set of inputs needed to run one coverage analysis."""

    walker: WalkerConfig
    simulation: SimulationConfig
    coverage: CoverageConfig
    chunk_max_floats: int = 20_000_000  # ~160 MB of float64 per chunk
    label: str = field(default="run")
