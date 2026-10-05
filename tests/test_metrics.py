"""Unit tests for the streaming CoverageAccumulator on synthetic series."""

from __future__ import annotations

import numpy as np

from walkercov.config import CoverageConfig
from walkercov.coverage.grid import build_grid
from walkercov.coverage.metrics import CoverageAccumulator
from walkercov.coverage.visibility import VisibilityChunk


def _run_series(series: np.ndarray, dt: float = 1.0):
    """Feed a (T, 1) boolean coverage series into the accumulator."""

    cov = CoverageConfig(lat_step_deg=90.0, lon_step_deg=180.0,
                         min_elevation_deg=0.0, required_sats=1)
    grid = build_grid(cov)
    # Reduce to a single-point grid for the test.
    class _Fake:
        area_weights = np.array([1.0])
        unit_vectors = np.array([[1.0, 0.0, 0.0]])
        shape = (1, 1)
        num_points = 1
    grid = _Fake()  # type: ignore[assignment]

    acc = CoverageAccumulator(grid=grid, required_sats=1, time_step_s=dt,
                                num_steps=series.shape[0])
    counts = series.astype(np.uint16).reshape(-1, 1)
    inst = series.astype(np.float64)  # single point: fraction == value
    chunk = VisibilityChunk(t_start_index=0, t_s=np.arange(series.shape[0]) * dt,
                             counts=counts, instantaneous_covered=inst)
    acc.update(chunk)
    return acc.finalize()


def test_simple_access_pattern_gaps_and_revisits() -> None:
    # Pattern index:  0 1 2 3 4 5 6 7 8 9
    #                 0 0 1 1 0 0 1 0 0 1
    series = np.array([0, 0, 1, 1, 0, 0, 1, 0, 0, 1])
    res = _run_series(series, dt=1.0)

    # Access starts at t=2, 6, 9.  Revisit intervals: (6-2)=4, (9-6)=3.
    assert res.mean_revisit_s[0] == 3.5
    assert res.max_revisit_s[0] == 4.0
    # Interior gaps (fully observed): the one between the first and second
    # access (indices 4..5, length 2) and between the second and third (index
    # 7..8, length 2).  Leading gap is 2 (censored by t=0); trailing is 0.
    assert res.max_gap_s[0] == 2.0  # interior and leading both equal 2
    assert np.isclose(res.leading_gap_s[0], 2.0)
    assert np.isnan(res.trailing_gap_s[0])
    # Percent time covered: 4 out of 10 epochs.
    assert np.isclose(res.percent_time_covered[0], 0.4)


def test_never_covered_point() -> None:
    series = np.zeros(10, dtype=np.int32)
    res = _run_series(series)
    assert not res.any_access[0]
    assert np.isnan(res.mean_revisit_s[0])
    # Leading gap equals full window; trailing is NaN (we recorded only the
    # single censored gap under leading_gap_s).
    assert res.leading_gap_s[0] == 10.0
    assert np.isnan(res.trailing_gap_s[0])
    assert res.max_gap_s[0] == 10.0
    assert res.percent_time_covered[0] == 0.0


def test_always_covered_point() -> None:
    series = np.ones(10, dtype=np.int32)
    res = _run_series(series)
    assert res.any_access[0]
    # No gaps and no interior revisits.
    assert np.isnan(res.mean_revisit_s[0])
    assert res.max_gap_s[0] == 0.0
    assert res.percent_time_covered[0] == 1.0
    assert np.isclose(res.mean_response_time_s[0], 0.0)


def test_response_time_matches_hand_calculation() -> None:
    # Series: 1 0 0 1 0 1  -> gaps of length 2 (interior) and 1 (interior).
    # Trailing: pattern ends with 1 so no trailing censoring.
    # Leading: starts with 1 so leading gap is 0 (not censored).
    series = np.array([1, 0, 0, 1, 0, 1])
    res = _run_series(series, dt=1.0)
    # sum(g^2) = 4 + 1 = 5; T = 6; response time = 5 / (2 * 6) = 0.41666...
    np.testing.assert_allclose(res.mean_response_time_s[0], 5.0 / 12.0)


def test_chunk_boundary_does_not_change_result() -> None:
    series = np.array([0, 0, 1, 1, 0, 0, 1, 0, 0, 1])

    cov = CoverageConfig(lat_step_deg=90.0, lon_step_deg=180.0,
                         min_elevation_deg=0.0, required_sats=1)
    class _Fake:
        area_weights = np.array([1.0])
        unit_vectors = np.array([[1.0, 0.0, 0.0]])
        shape = (1, 1)
        num_points = 1
    grid = _Fake()

    acc = CoverageAccumulator(grid=grid, required_sats=1, time_step_s=1.0,
                                num_steps=series.shape[0])
    for i0, i1 in [(0, 3), (3, 7), (7, 10)]:
        counts = series[i0:i1].astype(np.uint16).reshape(-1, 1)
        inst = series[i0:i1].astype(np.float64)
        acc.update(VisibilityChunk(t_start_index=i0,
                                     t_s=np.arange(i0, i1).astype(np.float64),
                                     counts=counts,
                                     instantaneous_covered=inst))
    chunked = acc.finalize()

    monolithic = _run_series(series, dt=1.0)
    np.testing.assert_array_equal(chunked.max_gap_s, monolithic.max_gap_s)
    np.testing.assert_array_equal(chunked.mean_revisit_s,
                                  monolithic.mean_revisit_s)
    np.testing.assert_array_equal(chunked.percent_time_covered,
                                  monolithic.percent_time_covered)
