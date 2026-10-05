# Walker constellation coverage

Interactive Streamlit tool for Walker Delta constellation design and global
Earth coverage analysis.  The orbital mechanics and coverage geometry are
written out explicitly in NumPy, not hidden behind an astrodynamics
library, so the tool doubles as a reference implementation.

Default case (one click, a few seconds on a modern laptop):

* Walker Delta **24/6/1**
* 550 km altitude, 53 deg inclination
* 10 deg minimum elevation, k = 1
* 24 hour simulation, 60 s step, 2 deg x 2 deg equal-area ground grid

## Features

* Walker Delta T/P/F generator with the standard STK / Vallado convention,
  explicitly documented in `walkercov/orbit/walker.py` and `walkercov/ui/methods.md`.
* Vectorized two-body circular propagator (`walkercov/orbit/propagator.py`),
  with a `Propagator` protocol so J2 secular or higher-fidelity propagators
  can be dropped in later without touching the UI or coverage engine.
* ECI -> ECEF rotation using the sidereal rotation rate.
* **Equal-area** latitude/longitude ground grid.  All percentages and means
  are area weighted so the poles are not over-counted.
* **Streaming** coverage engine that keeps O(G) state, where G is the number
  of grid points.  Memory does not grow with the number of time steps, so
  high-resolution runs stay feasible.
* Rigorous separation of instantaneous coverage, coverage gap, revisit
  interval and mean response time.  Censored gaps at the simulation boundaries
  are handled explicitly.
* Plotly 3D view, coverage heatmaps for several metrics, ground tracks,
  coverage-vs-time, latitude-banded statistics, and area-weighted histograms.
* Built-in "Compare F values" mode: automatically runs every valid F for the
  current T and P so you can see how Walker phasing changes global coverage,
  mean response time and max gap.
* 28 pytest tests, including analytic coverage cases (spherical cap fraction,
  polar revisit = orbital period, GEO sub-sat always covered, latitude band
  outside i + lambda_max never covered).

## Install and run

The project is a standard [uv](https://docs.astral.sh/uv/) project
(`pyproject.toml` + `uv.lock` + `.python-version`), so the shortest path is:

```bash
# One-liner.  uv creates the venv, installs locked deps, launches the app.
uv run streamlit run app.py

# Run the tests the same way:
uv run pytest
```

The first `uv run` takes a few seconds to install the locked dependencies into
`.venv/`; subsequent runs start instantly.

If you prefer to manage the venv yourself:

```bash
# with uv:
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
streamlit run app.py

# or with plain pip:
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
streamlit run app.py
```

## Project structure

```
app.py                       Streamlit entry point (thin)
pyproject.toml
walkercov/
  constants.py               SI constants: mu, R_Earth, omega_Earth, J2
  config.py                  WalkerConfig, SimulationConfig, CoverageConfig, RunConfig
  orbit/
    elements.py              OrbitalElements block (plane, slot, shell ids)
    walker.py                Walker Delta generator + satellite table
    propagator.py            Propagator protocol + CircularTwoBodyPropagator
    coordinates.py           Earth rotation, lat/lon <-> ECEF
  coverage/
    grid.py                  Cell-centred equal-area lat/lon grid
    visibility.py            lambda_max, elevation, chunked visibility counts
    metrics.py               Streaming CoverageAccumulator + CoverageResult
    analysis.py              run_coverage, compare_f_values, comparison table
  visualization/
    globe3d.py               3D constellation view (Earth, orbits, satellites)
    maps.py                  2-D heatmaps, instantaneous snapshot, ground tracks
    charts.py                Coverage vs time, vs latitude, histograms, F-compare
  ui/
    sidebar.py               Streamlit form -> RunConfig
    tabs.py                  Constellation, Coverage, Statistics, Compare F, Methods
    methods.md               Equations and conventions
tests/
  test_walker.py             Walker 24/6/1 counts, spacings, validation
  test_propagator.py         Period at 550 km, GEO sidereal day, orthogonality
  test_coordinates.py        ECI->ECEF rotation, lat/lon roundtrip
  test_visibility.py         Zenith, tangent, lambda_max, dot-product equivalence
  test_metrics.py            Gaps, revisit, censoring, response time, chunk-invariance
  test_coverage_cases.py     Analytic end-to-end: cap fraction, polar revisit, GEO
```

## Walker Delta 24/6/1 (default)

| Quantity                           | Value               |
|------------------------------------|---------------------|
| Total satellites T                 | 24                  |
| Orbital planes P                   | 6                   |
| Satellites per plane S = T/P       | 4                   |
| Walker phasing F                   | 1 (valid: 0..5)     |
| RAAN step 360 / P                  | 60 deg              |
| In-plane step 360 / S              | 90 deg              |
| Walker phase step F * 360 / T      | 15 deg              |
| Period (550 km)                    | ~95.65 min          |
| Footprint half-angle (10 deg mask) | ~14.96 deg          |
| Footprint radius                   | ~1665 km            |
| Max pass duration                  | ~7.95 min           |

## What the engine does (summary)

Visibility for a spherical Earth of radius R and a satellite at altitude h
with elevation mask eps reduces to a central-angle test:

    lambda_max = acos( (R / (R + h)) * cos(eps) ) - eps
    visible if  r_hat_sat . r_hat_ground  >=  cos(lambda_max)

This lets the hot path collapse to one matrix multiply per epoch chunk.  The
explicit elevation formula is also implemented and the two give identical
results in the unit tests.

The streaming metrics engine keeps per-point running state (counts,
accumulated covered time, current gap start, last access start,
sum/squared-sum of gap lengths, revisit list, min/max satellites visible) and
updates these across chunks with vectorized NumPy operations plus a tiny
Python loop over the sparse list of (access-start, access-end) edge events.

## Future work

The architecture leaves room for, in increasing order of effort: J2 secular
propagation, sun-synchronous design helpers, Walker Star constellations,
multiple shells, ground stations, regional areas of interest, sensor
field-of-view constraints, inter-satellite link visibility, link-budget
constraints, satellite failures, constellation optimisation, orbital-state
export, and comparison against STK or an independent reference.  None of
these should require touching the top-level analysis layer.

## Licence

MIT.
