"""High-level entry points: ``run_coverage`` and ``compare_f_values``.

These functions glue together the Walker generator, the propagator, the
visibility kernel and the metrics accumulator.  They are the only functions
the Streamlit UI needs to call for coverage analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..config import RunConfig, WalkerConfig
from ..orbit.elements import OrbitalElements
from ..orbit.propagator import CircularTwoBodyPropagator, Propagator
from ..orbit.walker import walker_delta, valid_f_values
from .grid import build_grid
from .metrics import CoverageAccumulator, CoverageResult
from .visibility import iter_visibility_chunks


ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True)
class RunArtifacts:
    """Everything a single coverage run produces."""

    config: RunConfig
    elements: OrbitalElements
    result: CoverageResult


def run_coverage(config: RunConfig,
                 *,
                 propagator: Optional[Propagator] = None,
                 progress: Optional[ProgressCallback] = None) -> RunArtifacts:
    """Propagate a constellation and compute full coverage metrics."""

    prop = propagator or CircularTwoBodyPropagator()
    elements = walker_delta(config.walker)
    grid = build_grid(config.coverage)

    accumulator = CoverageAccumulator(
        grid=grid,
        required_sats=config.coverage.required_sats,
        time_step_s=config.simulation.time_step_s,
        num_steps=config.simulation.num_steps,
    )

    def chunk_progress(frac: float) -> None:
        if progress is not None:
            progress(frac, f"Propagating + scoring coverage ({frac*100:4.1f}%)")

    for chunk in iter_visibility_chunks(config, elements, grid, prop,
                                        progress=chunk_progress):
        accumulator.update(chunk)

    if progress is not None:
        progress(1.0, "Finalising metrics")

    result = accumulator.finalize()
    return RunArtifacts(config=config, elements=elements, result=result)


# ---------------------------------------------------------------------------
# Compare F values
# ---------------------------------------------------------------------------


def compare_f_values(base: RunConfig,
                     *,
                     propagator: Optional[Propagator] = None,
                     progress: Optional[ProgressCallback] = None,
                     ) -> dict[int, RunArtifacts]:
    """Run coverage for every valid F in [0, P-1] with everything else fixed."""

    out: dict[int, RunArtifacts] = {}
    f_values = valid_f_values(base.walker.num_planes)
    for i, f in enumerate(f_values):
        walker = WalkerConfig(
            total_sats=base.walker.total_sats,
            num_planes=base.walker.num_planes,
            phasing_f=f,
            altitude_km=base.walker.altitude_km,
            inclination_deg=base.walker.inclination_deg,
            arg_of_latitude0_deg=base.walker.arg_of_latitude0_deg,
        )
        cfg = replace(base, walker=walker, label=f"F={f}")

        def sub_progress(frac: float, msg: str, i=i, n=len(f_values), f=f) -> None:
            if progress is not None:
                progress((i + frac) / n, f"F={f}: {msg}")

        out[f] = run_coverage(cfg, propagator=propagator, progress=sub_progress)

    return out


def comparison_dataframe(results: dict[int, RunArtifacts]) -> pd.DataFrame:
    """Compact per-F summary table built from a :func:`compare_f_values` result."""

    rows = []
    for f, art in results.items():
        s = art.result.summary()
        rows.append({
            "F": f,
            "mean_global_coverage_%": 100.0 * s["mean_global_coverage"],
            "min_global_coverage_%": 100.0 * s["min_global_coverage"],
            "mean_percent_time_covered_%": 100.0 * s["mean_percent_time_covered"],
            "mean_sats_visible": s["mean_mean_sats_visible"],
            "max_gap_min": s["max_gap_s_global"] / 60.0,
            "mean_response_min": s["mean_response_time_s_global"] / 60.0,
            "mean_revisit_min": s["mean_revisit_s_global"] / 60.0 if np.isfinite(s["mean_revisit_s_global"]) else np.nan,
            "max_revisit_min": s["max_revisit_s_global"] / 60.0 if np.isfinite(s["max_revisit_s_global"]) else np.nan,
            "area_fraction_ever_covered_%": 100.0 * s["area_fraction_ever_covered"],
        })
    return pd.DataFrame(rows).set_index("F")
