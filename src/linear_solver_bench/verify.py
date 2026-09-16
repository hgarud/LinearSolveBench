"""Decode native output and independently check a returned solution."""

from __future__ import annotations

import math
import struct
from dataclasses import asdict, dataclass

import numpy as np

from .models import BenchmarkCase, RawExecution

OUTPUT_MAGIC = b"LSBOUT01"
OUTPUT_HEADER = struct.Struct("<8sIiQQQQQB7x")
FLOAT64_EPSILON = float(np.finfo(np.float64).eps)


@dataclass(frozen=True)
class Metrics:
    normwise_backward_error: float
    componentwise_backward_error: float
    relative_residual: float
    relative_l2_forward_error: float | None
    relative_linf_forward_error: float | None


@dataclass(frozen=True)
class RunResult:
    passed: bool
    elapsed_seconds: float | None
    failure: str | None
    process_returncode: int
    process_wall_seconds: float
    status: int | None = None
    create_seconds: float | None = None
    setup_seconds: float | None = None
    solve_seconds: float | None = None
    input_mutated: bool | None = None
    metrics: Metrics | None = None
    diagnostics: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "elapsed_seconds": self.elapsed_seconds,
            "failure": self.failure,
            "process_returncode": self.process_returncode,
            "process_wall_seconds": self.process_wall_seconds,
            "status": self.status,
            "create_seconds": self.create_seconds,
            "setup_seconds": self.setup_seconds,
            "solve_seconds": self.solve_seconds,
            "input_mutated": self.input_mutated,
            "metrics": asdict(self.metrics) if self.metrics else None,
            "diagnostics": self.diagnostics,
        }


def _max_abs(vector: np.ndarray) -> float:
    return float(np.max(np.abs(vector), initial=0.0))


def _ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return math.inf
    if denominator:
        return numerator / denominator
    return 0.0 if numerator == 0.0 else math.inf


def _relative_l2(numerator: np.ndarray, denominator: np.ndarray) -> float:
    """Compute a norm ratio without overflowing the individual norms."""
    numerator_scale = _max_abs(numerator)
    denominator_scale = _max_abs(denominator)
    if not numerator_scale or not denominator_scale:
        return _ratio(numerator_scale, denominator_scale)
    if not math.isfinite(numerator_scale) or not math.isfinite(denominator_scale):
        return math.inf
    scaled = float(np.linalg.norm(numerator / numerator_scale)) / float(
        np.linalg.norm(denominator / denominator_scale)
    )
    numerator_mantissa, numerator_exponent = math.frexp(numerator_scale)
    denominator_mantissa, denominator_exponent = math.frexp(denominator_scale)
    try:
        return math.ldexp(
            scaled * numerator_mantissa / denominator_mantissa,
            numerator_exponent - denominator_exponent,
        )
    except OverflowError:
        return math.inf


def _metrics(case: BenchmarkCase, solution: np.ndarray) -> Metrics:
    value = case.input
    matrix = value.matrix.to_scipy()
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        residual = np.asarray(value.b - matrix @ solution, dtype=np.float64)
        absolute_matrix = abs(matrix)
        matrix_norm = float(
            np.max(np.asarray(absolute_matrix.sum(axis=1)).reshape(-1), initial=0.0)
        )
        normwise = _ratio(
            _max_abs(residual), matrix_norm * _max_abs(solution) + _max_abs(value.b)
        )
        denominator = np.asarray(absolute_matrix @ np.abs(solution), dtype=np.float64)
        denominator += np.abs(value.b)
        ratios = np.zeros_like(residual)
        positive = denominator > 0
        ratios[positive] = np.abs(residual[positive]) / denominator[positive]
        ratios[~positive & (np.abs(residual) > 0)] = math.inf
        componentwise = float(np.max(ratios, initial=0.0))
        relative_residual = _relative_l2(residual, value.b)
        if case.target is None:
            forward_l2 = forward_linf = None
        else:
            error = solution - case.target
            forward_l2 = _relative_l2(error, case.target)
            forward_linf = _ratio(_max_abs(error), _max_abs(case.target))
    return Metrics(
        normwise,
        componentwise,
        relative_residual,
        forward_l2,
        forward_linf,
    )


def _failure(family: str, tolerance: float, metrics: Metrics) -> str | None:
    values = asdict(metrics)
    required: dict[str, float]
    if family == "ns-mesh-pde":
        required = {
            "normwise_backward_error": 1.0e-10,
            "componentwise_backward_error": 1.0e-8,
            "relative_l2_forward_error": 1.0e-5,
            "relative_linf_forward_error": 1.0e-5,
        }
    elif family == "magnetic_diffusion_flash":
        required = {
            "normwise_backward_error": max(tolerance, 20 * FLOAT64_EPSILON),
            "relative_residual": tolerance * (1 + 1.0e-6),
        }
    else:
        raise ValueError(f"unknown benchmark family: {family}")
    for name, limit in required.items():
        value = values[name]
        if value is None or not math.isfinite(value):
            return f"invalid {name}"
        if value > limit:
            return f"{name} exceeds {limit:g}"
    return None


def _decode(payload: bytes, expected_size: int):
    expected_bytes = OUTPUT_HEADER.size + expected_size * 8
    if len(payload) != expected_bytes:
        raise ValueError("driver output has the wrong size")
    magic, version, status, size, elapsed, create, setup, solve, mutated = (
        OUTPUT_HEADER.unpack_from(payload)
    )
    if magic != OUTPUT_MAGIC or version != 1 or size != expected_size:
        raise ValueError("driver output header is invalid")
    if mutated not in (0, 1) or elapsed <= 0 or create + setup + solve != elapsed:
        raise ValueError("driver timing or mutation fields are invalid")
    solution = np.frombuffer(payload, "<f8", offset=OUTPUT_HEADER.size).copy()
    return status, elapsed, create, setup, solve, bool(mutated), solution


def verify(raw: RawExecution, case: BenchmarkCase, family: str) -> RunResult:
    """Return a fail-closed result; candidate failures do not raise."""
    common = {
        "process_returncode": raw.returncode,
        "process_wall_seconds": raw.wall_seconds,
        "diagnostics": raw.diagnostics,
    }
    if raw.returncode != 0:
        return RunResult(False, None, "process failed", **common)
    if raw.output is None:
        return RunResult(False, None, "missing driver output", **common)
    try:
        status, elapsed, create, setup, solve, mutated, solution = _decode(
            raw.output, case.input.matrix.n
        )
    except ValueError as error:
        return RunResult(False, None, str(error), **common)
    seconds = elapsed * 1.0e-9
    details = {
        "elapsed_seconds": seconds,
        "status": status,
        "create_seconds": create * 1.0e-9,
        "setup_seconds": setup * 1.0e-9,
        "solve_seconds": solve * 1.0e-9,
        "input_mutated": mutated,
    }
    if status != 0:
        return RunResult(False, failure="solver returned an error", **common, **details)
    if mutated:
        return RunResult(False, failure="solver mutated its input", **common, **details)
    if not np.all(np.isfinite(solution)):
        return RunResult(False, failure="solution is not finite", **common, **details)
    try:
        metrics = _metrics(case, solution)
        failure = _failure(family, case.input.tolerance, metrics)
    except (FloatingPointError, OverflowError, ValueError):
        return RunResult(False, failure="verification failed", **common, **details)
    return RunResult(
        failure is None, failure=failure, metrics=metrics, **common, **details
    )
