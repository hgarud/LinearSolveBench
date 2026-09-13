"""Public resources and writable caches, in a checkout or installed package."""

from __future__ import annotations

import atexit
import os
import pathlib
from contextlib import ExitStack
from functools import cache
from importlib import resources

_RESOURCE_CONTEXTS = ExitStack()
atexit.register(_RESOURCE_CONTEXTS.close)


@cache
def _installed_root() -> pathlib.Path:
    # Keep an extracted directory alive for the process if imported from a zip.
    resource = resources.files("linear_solver_bench").joinpath("_resources")
    if not resource.is_dir():
        raise RuntimeError("installed package is missing public benchmark resources")
    return _RESOURCE_CONTEXTS.enter_context(resources.as_file(resource))


def repository_root() -> pathlib.Path:
    """Return the root of the public benchmark assets, which may be installed."""
    override = os.environ.get("LINEAR_SOLVER_BENCH_ROOT")
    if override:
        root = pathlib.Path(override).expanduser().resolve()
    else:
        checkout = pathlib.Path(__file__).resolve().parents[2]
        root = (
            checkout if (checkout / "benchmark.toml").is_file() else _installed_root()
        )
    if not (root / "benchmark.toml").is_file():
        raise RuntimeError(
            "cannot locate benchmark.toml; set LINEAR_SOLVER_BENCH_ROOT "
            "to a directory containing public benchmark assets"
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


def cache_dir() -> pathlib.Path:
    """Return writable storage separately from read-only installed resources."""
    override = os.environ.get("LINEAR_SOLVER_BENCH_CACHE")
    if override:
        return pathlib.Path(override).expanduser().resolve()
    base = pathlib.Path(os.environ.get("XDG_CACHE_HOME") or "~/.cache").expanduser()
    return (base / "linear-solver-bench").resolve()
