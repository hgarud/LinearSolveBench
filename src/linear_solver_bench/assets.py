"""Locate the benchmark assets in a checkout or an installed wheel."""

from __future__ import annotations

import os
import sysconfig
from pathlib import Path


def asset_root(*required: str) -> Path:
    candidates = []
    if configured := os.environ.get("LINEAR_SOLVER_BENCH_ROOT"):
        candidates.append(Path(configured))
    candidates.extend(
        (
            Path(__file__).resolve().parents[2],
            Path(sysconfig.get_path("data")) / "share" / "linear-solver-bench",
        )
    )
    required = required or ("families", "native", "reference", "examples")
    for candidate in candidates:
        if all((candidate / name).is_dir() for name in required):
            return candidate
    raise RuntimeError("LinearSolverBench assets are missing; reinstall the package")
