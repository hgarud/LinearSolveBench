"""Repository paths with explicit overrides for packaged/operator use."""

from __future__ import annotations

import os
import pathlib


def repository_root() -> pathlib.Path:
    override = os.environ.get("LINEAR_SOLVER_BENCH_ROOT")
    if override:
        root = pathlib.Path(override).expanduser().resolve()
    else:
        root = pathlib.Path(__file__).resolve().parents[2]
    if not (root / "benchmark.toml").is_file():
        raise RuntimeError(
            "cannot locate benchmark.toml; set LINEAR_SOLVER_BENCH_ROOT to the checkout"
        )
    return root


def data_dir() -> pathlib.Path:
    override = os.environ.get("LINEAR_SOLVER_BENCH_DATA")
    return (
        pathlib.Path(override).expanduser().resolve()
        if override
        else repository_root() / "data"
    )


def native_dir() -> pathlib.Path:
    return repository_root() / "native"
