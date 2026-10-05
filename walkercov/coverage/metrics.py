"""Streaming coverage metric accumulation.

Rather than keeping the full ``(num_steps, num_points)`` boolean cube in
memory, this module walks through the simulation in chunks and maintains a
small per-point running state.  All updates are fully vectorized across the
grid.

Metrics tracked per ground point
--------------------------------

The engine makes a strict distinction between three closely related, often
conflated quantities:

* **Instantaneous coverage** - whether at least ``k`` satellites are visible
  right now.  Captured by ``covered_steps`` / total steps.
* **Coverage gap** - a maximal consecutive run of epochs during which the
  point is *not* covered.  Gaps bounded by the simulation window on either
  end are **censored**; we record their length and count separately.
* **Revisit interval** - the time between the starts of two successive
  access intervals at a point.  Only intervals bounded on both sides by an
  observed access start are counted.  Equivalently, it is "gap + previous
  access duration" or "access-start[k+1] - access-start[k]".

A fourth, useful summary statistic is the **mean response time**: the
time-averaged wait a user at the ground point would experience before the
next access.  For a stationary pattern of gaps :math:`g_i`,
:math:`\\overline{R} = \\frac{\\sum g_i^2}{2 T}`.  Over a finite simulation
we treat censored gaps (which may actually extend further) as if the full
observed length were a gap; that is the standard STK formulation and is
conservative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np

from ..config import RunConfig
from .grid import GroundGrid, latitude_band_weights
from .visibility import VisibilityChunk


SENTINEL = -1  # "no observed access/gap start yet"


# ---------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------


@dataclass
class CoverageAccumulator:
    """Streaming accumulator for coverage metrics across a grid."""

    grid: GroundGrid
    required_sats: int
    time_step_s: float
    num_steps: int

    # Running state, allocated in ``__post_init__``.
    _covered_steps: np.ndarray = field(init=False, repr=False)
    _count_sum: np.ndarray = field(init=False, repr=False)
    _count_min: np.ndarray = field(init=False, repr=False)
    _count_max: np.ndarray = field(init=False, repr=False)

    # Gap bookkeeping.
    _in_gap: np.ndarray = field(init=False, repr=False)              # bool
    _gap_start_idx: np.ndarray = field(init=False, repr=False)        # int
    _gap_sum_steps: np.ndarray = field(init=False, repr=False)        # int64 (interior only)
    _gap_sq_sum_steps2: np.ndarray = field(init=False, repr=False)    # int64, includes censored
    _gap_count: np.ndarray = field(init=False, repr=False)            # int (interior only)
    _max_gap_steps: np.ndarray = field(init=False, repr=False)        # int
    _leading_gap_steps: np.ndarray = field(init=False, repr=False)    # int, censored at t=0
    _trailing_gap_steps: np.ndarray = field(init=False, repr=False)   # int, censored at t_end
    _any_access: np.ndarray = field(init=False, repr=False)           # bool

    # Revisit bookkeeping: time between successive access-starts.
    _first_access_idx: np.ndarray = field(init=False, repr=False)
    _last_access_start_idx: np.ndarray = field(init=False, repr=False)
    _revisit_sum_steps: np.ndarray = field(init=False, repr=False)
    _revisit_max_steps: np.ndarray = field(init=False, repr=False)
    _revisit_count: np.ndarray = field(init=False, repr=False)
    _in_access_prev: np.ndarray = field(init=False, repr=False)

    # Global time series.
    _global_coverage: np.ndarray = field(init=False, repr=False)
    _steps_seen: int = field(init=False, repr=False, default=0)

    def __post_init__(self) -> None:
        G = self.grid.num_points
        self._covered_steps = np.zeros(G, dtype=np.int64)
        self._count_sum = np.zeros(G, dtype=np.int64)
        self._count_min = np.full(G, np.iinfo(np.uint16).max, dtype=np.int32)
        self._count_max = np.zeros(G, dtype=np.int32)

        self._in_gap = np.ones(G, dtype=bool)  # we start assumed-uncovered until proven otherwise
        self._gap_start_idx = np.zeros(G, dtype=np.int64)
        self._gap_sum_steps = np.zeros(G, dtype=np.int64)
        self._gap_sq_sum_steps2 = np.zeros(G, dtype=np.int64)
        self._gap_count = np.zeros(G, dtype=np.int64)
        self._max_gap_steps = np.zeros(G, dtype=np.int64)
        self._leading_gap_steps = np.full(G, SENTINEL, dtype=np.int64)
        self._trailing_gap_steps = np.full(G, SENTINEL, dtype=np.int64)
        self._any_access = np.zeros(G, dtype=bool)

        self._first_access_idx = np.full(G, SENTINEL, dtype=np.int64)
        self._last_access_start_idx = np.full(G, SENTINEL, dtype=np.int64)
        self._revisit_sum_steps = np.zeros(G, dtype=np.int64)
        self._revisit_max_steps = np.zeros(G, dtype=np.int64)
        self._revisit_count = np.zeros(G, dtype=np.int64)
        self._in_access_prev = np.zeros(G, dtype=bool)

        self._global_coverage = np.empty(self.num_steps, dtype=np.float64)

    # ------------------------------------------------------------------
    # Chunk update
    # ------------------------------------------------------------------

    def update(self, chunk: VisibilityChunk) -> None:
        counts = chunk.counts.astype(np.int32, copy=False)        # (T, G)
        covered = counts >= self.required_sats                     # (T, G) bool

        # Record the (T,) area-weighted global time series straight from the
        # chunk, which already computed it.
        T_chunk = counts.shape[0]
        self._global_coverage[chunk.t_start_index:
                              chunk.t_start_index + T_chunk] = \
            chunk.instantaneous_covered

        # Simple reductions (vectorised, no Python per-point loop).
        self._covered_steps += covered.sum(axis=0, dtype=np.int64)
        self._count_sum += counts.sum(axis=0, dtype=np.int64)
        chunk_min = counts.min(axis=0).astype(np.int32)
        chunk_max = counts.max(axis=0).astype(np.int32)
        np.minimum(self._count_min, chunk_min, out=self._count_min)
        np.maximum(self._count_max, chunk_max, out=self._count_max)

        # ------------------------------------------------------------------
        # Event-based gap and revisit statistics.  We detect edges with a
        # single diff per chunk and loop over the (small) list of transitions.
        # ------------------------------------------------------------------
        # Prepend previous covered state (= not in_gap) to detect edges across
        # chunk boundaries.
        prev_covered = (~self._in_gap)[None, :]                   # (1, G)
        extended = np.vstack((prev_covered, covered))             # (T+1, G)
        diff = extended[1:].astype(np.int8) - extended[:-1].astype(np.int8)
        # +1: start of access, -1: end of access.
        start_t, start_g = np.where(diff == 1)
        end_t, end_g = np.where(diff == -1)
        start_abs = start_t + chunk.t_start_index
        end_abs = end_t + chunk.t_start_index

        # Merge start and end events for each point in strict temporal order.
        # Events at a single point must alternate start -> end -> start -> ...,
        # so sorting by (point, time) and dispatching one by one keeps the per-
        # point state machine correct.  The edge list is tiny compared to
        # G*T_chunk, so a Python loop over it is cheap.
        all_t = np.concatenate((start_abs, end_abs))
        all_g = np.concatenate((start_g, end_g))
        all_type = np.concatenate((
            np.ones(start_abs.shape[0], dtype=np.int8),
            -np.ones(end_abs.shape[0], dtype=np.int8),
        ))
        order = np.lexsort((all_t, all_g))
        for i in order.tolist():
            g = int(all_g[i])
            t_abs = int(all_t[i])
            if all_type[i] == 1:
                self._on_access_start(g, t_abs)
            else:
                self._on_access_end(g, t_abs)

        # After processing all edges in this chunk, update the running
        # "currently covered?" state from the last row of ``covered``.
        self._in_gap = ~covered[-1]
        self._in_access_prev = covered[-1]
        self._steps_seen = chunk.t_start_index + T_chunk

    # ------------------------------------------------------------------
    # Edge handlers
    # ------------------------------------------------------------------

    def _on_access_start(self, g: int, t_idx: int) -> None:
        """Record that point ``g`` just transitioned from gap -> access at ``t_idx``."""

        # Close out the gap that ended at t_idx - 1.
        gap_start = self._gap_start_idx[g]
        gap_len = t_idx - gap_start
        if not self._any_access[g]:
            # This is the first-ever access -> the preceding gap is censored
            # by the start of the simulation.
            self._leading_gap_steps[g] = gap_len
            self._any_access[g] = True
            self._first_access_idx[g] = t_idx
        else:
            # Fully observed interior gap.
            self._gap_sum_steps[g] += gap_len
            self._gap_count[g] += 1
            if gap_len > self._max_gap_steps[g]:
                self._max_gap_steps[g] = gap_len
            # Revisit interval = this access start - last access start.
            prev_start = self._last_access_start_idx[g]
            if prev_start != SENTINEL:
                rev = t_idx - prev_start
                self._revisit_sum_steps[g] += rev
                self._revisit_count[g] += 1
                if rev > self._revisit_max_steps[g]:
                    self._revisit_max_steps[g] = rev

        self._gap_sq_sum_steps2[g] += gap_len * gap_len
        self._last_access_start_idx[g] = t_idx
        self._gap_start_idx[g] = -1  # mark "not in gap"

    def _on_access_end(self, g: int, t_idx: int) -> None:
        """Access ended between index ``t_idx - 1`` (covered) and ``t_idx`` (not)."""

        # A new gap starts at t_idx.
        self._gap_start_idx[g] = t_idx

    # ------------------------------------------------------------------
    # Finalise
    # ------------------------------------------------------------------

    def finalize(self) -> "CoverageResult":
        """Flush any trailing gap and return a :class:`CoverageResult`."""

        G = self.grid.num_points
        total_steps = self._steps_seen

        # Any point still "in gap" at simulation end has a trailing censored
        # gap from gap_start_idx .. total_steps - 1.
        trailing_mask = self._in_gap
        trailing_start = self._gap_start_idx.copy()
        trailing_len = np.where(trailing_mask,
                                total_steps - trailing_start,
                                SENTINEL)
        # If a point never had an access, we treat the entire window as a
        # single censored gap of length ``total_steps``.  The leading gap is
        # recorded that way and no trailing gap is reported.
        never_access = ~self._any_access
        if np.any(never_access):
            self._leading_gap_steps[never_access] = total_steps
            trailing_len[never_access] = SENTINEL
        self._trailing_gap_steps = trailing_len

        # Response-time formula: sum(g^2) / (2 * T).  Include censored gaps
        # (both leading and trailing) in the sum so the fraction equals a
        # time-averaged wait, which is the conventional STK meaning.
        resp_sq_sum = self._gap_sq_sum_steps2.astype(np.float64)
        # Trailing (not yet counted) censored gap:
        valid = self._trailing_gap_steps > 0
        resp_sq_sum[valid] += (self._trailing_gap_steps[valid].astype(np.float64)
                               ** 2)

        dt = self.time_step_s
        duration_s = total_steps * dt
        # Convert everything to seconds.
        percent_time_covered = self._covered_steps / max(1, total_steps)
        mean_sats = self._count_sum / max(1, total_steps)
        min_sats = self._count_min.astype(np.int32)
        # If the point never saw a satellite the running min stayed at the
        # sentinel ``uint16 max``; clamp to zero in that case so plots are
        # not dominated by garbage values.
        min_sats = np.where(min_sats > self._count_max, 0, min_sats)
        max_sats = self._count_max.astype(np.int32)

        max_gap_s = self._max_gap_steps.astype(np.float64) * dt
        # Also consider censored gaps in max gap (they are lower bounds but
        # usually dominate for sparse constellations).
        leading_valid = self._leading_gap_steps >= 0
        max_gap_s = np.where(leading_valid,
                             np.maximum(max_gap_s,
                                        self._leading_gap_steps.astype(np.float64) * dt),
                             max_gap_s)
        trailing_valid = self._trailing_gap_steps >= 0
        max_gap_s = np.where(trailing_valid,
                             np.maximum(max_gap_s,
                                        self._trailing_gap_steps.astype(np.float64) * dt),
                             max_gap_s)

        # Mean revisit from the fully observed interior revisit intervals.
        has_revisit = self._revisit_count > 0
        mean_revisit_s = np.full(G, np.nan, dtype=np.float64)
        mean_revisit_s[has_revisit] = (
            self._revisit_sum_steps[has_revisit].astype(np.float64)
            / self._revisit_count[has_revisit]
        ) * dt
        max_revisit_s = np.where(has_revisit,
                                 self._revisit_max_steps.astype(np.float64) * dt,
                                 np.nan)

        mean_response_s = (resp_sq_sum / (2.0 * max(1, total_steps))) * dt
        # Points with continuous coverage have no response time and we leave
        # the value at zero (which is already correct).

        return CoverageResult(
            grid=self.grid,
            time_step_s=dt,
            duration_s=duration_s,
            num_steps=total_steps,
            required_sats=self.required_sats,
            percent_time_covered=percent_time_covered,
            mean_satellites_visible=mean_sats,
            min_satellites_visible=min_sats,
            max_satellites_visible=max_sats,
            max_gap_s=max_gap_s,
            mean_revisit_s=mean_revisit_s,
            max_revisit_s=max_revisit_s,
            mean_response_time_s=mean_response_s,
            any_access=self._any_access.copy(),
            leading_gap_s=np.where(leading_valid,
                                   self._leading_gap_steps.astype(np.float64) * dt,
                                   np.nan),
            trailing_gap_s=np.where(trailing_valid,
                                    self._trailing_gap_steps.astype(np.float64) * dt,
                                    np.nan),
            global_coverage_fraction=self._global_coverage[:total_steps].copy(),
        )


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoverageResult:
    """Final per-point and global coverage metrics for a single run."""

    grid: GroundGrid
    time_step_s: float
    duration_s: float
    num_steps: int
    required_sats: int

    percent_time_covered: np.ndarray        # [0, 1]
    mean_satellites_visible: np.ndarray
    min_satellites_visible: np.ndarray
    max_satellites_visible: np.ndarray
    max_gap_s: np.ndarray                   # includes censored contributions
    mean_revisit_s: np.ndarray              # NaN if < 2 observed access starts
    max_revisit_s: np.ndarray               # NaN if < 2 observed access starts
    mean_response_time_s: np.ndarray
    any_access: np.ndarray                  # bool: True if point ever saw a sat
    leading_gap_s: np.ndarray               # NaN if first-ever epoch is covered
    trailing_gap_s: np.ndarray              # NaN if last epoch is covered
    global_coverage_fraction: np.ndarray    # (num_steps,) area-weighted fraction

    # ------------------------------------------------------------------
    # Reshaping helpers
    # ------------------------------------------------------------------

    def reshape_grid(self, flat: np.ndarray) -> np.ndarray:
        """Reshape a flat (G,) array into the (n_lat, n_lon) grid order."""

        return flat.reshape(self.grid.shape)

    # ------------------------------------------------------------------
    # Scalar summaries (area-weighted)
    # ------------------------------------------------------------------

    def area_mean(self, metric: np.ndarray) -> float:
        """Area-weighted mean of a per-point metric, ignoring NaNs."""

        w = self.grid.area_weights
        mask = np.isfinite(metric)
        if not np.any(mask):
            return float("nan")
        wm = w[mask]
        wm = wm / wm.sum()
        return float(np.sum(wm * metric[mask]))

    def area_max(self, metric: np.ndarray) -> float:
        """Max of a per-point metric over the covered region (NaN-safe)."""

        if np.all(np.isnan(metric)):
            return float("nan")
        return float(np.nanmax(metric))

    def summary(self) -> dict[str, float]:
        """Return the headline scalar metrics."""

        gc = self.global_coverage_fraction
        return {
            "mean_global_coverage": float(gc.mean()),
            "min_global_coverage": float(gc.min()),
            "max_global_coverage": float(gc.max()),
            "mean_percent_time_covered": self.area_mean(self.percent_time_covered),
            "mean_mean_sats_visible": self.area_mean(self.mean_satellites_visible),
            "area_fraction_ever_covered": float(
                np.sum(self.grid.area_weights * self.any_access.astype(np.float64))
            ),
            "max_gap_s_global": self.area_max(self.max_gap_s),
            "mean_response_time_s_global": self.area_mean(self.mean_response_time_s),
            "mean_revisit_s_global": self.area_mean(self.mean_revisit_s),
            "max_revisit_s_global": self.area_max(self.max_revisit_s),
        }

    # ------------------------------------------------------------------
    # Latitude banding
    # ------------------------------------------------------------------

    def by_latitude(self, metric: np.ndarray) -> np.ndarray:
        """Area-weighted mean of a per-point metric per latitude band."""

        n_lat, n_lon = self.grid.shape
        m = metric.reshape(n_lat, n_lon)
        w = self.grid.area_weights.reshape(n_lat, n_lon)
        valid = np.isfinite(m)
        num = np.sum(np.where(valid, w * m, 0.0), axis=1)
        den = np.sum(np.where(valid, w, 0.0), axis=1)
        out = np.full(n_lat, np.nan)
        nz = den > 0
        out[nz] = num[nz] / den[nz]
        return out
