"""Independent, fail-closed terminal verification."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .accuracy import AccuracyThresholds
from .models import EvaluationSystem


@dataclass(frozen=True)
class AccuracyMetrics:
    normwise_backward_error: float
    componentwise_backward_error: float
    relative_l2_forward_error: float
    relative_linf_forward_error: float

    @property
    def forward_error(self) -> float:
        return max(self.relative_l2_forward_error, self.relative_linf_forward_error)


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    failure: str | None
    metrics: AccuracyMetrics | None
    thresholds: AccuracyThresholds

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "failure": self.failure,
            "metrics": asdict(self.metrics) if self.metrics else None,
            "thresholds": asdict(self.thresholds),
        }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator > 0.0:
        return numerator / denominator
    return 0.0 if numerator == 0.0 else math.inf


def accuracy_metrics(system: EvaluationSystem, solution: np.ndarray) -> AccuracyMetrics:
    public = system.public
    matrix = public.matrix.to_scipy()
    residual = np.asarray(public.b - matrix @ solution, dtype=np.float64)
    abs_residual = np.abs(residual)
    matrix_norm = float(
        np.max(np.asarray(abs(matrix).sum(axis=1)).reshape(-1), initial=0.0)
    )
    x_norm = float(np.max(np.abs(solution), initial=0.0))
    b_norm = float(np.max(np.abs(public.b), initial=0.0))
    normwise = _safe_ratio(
        float(np.max(abs_residual, initial=0.0)), matrix_norm * x_norm + b_norm
    )

    denominator = np.asarray(abs(matrix) @ np.abs(solution), dtype=np.float64)
    denominator += np.abs(public.b)
    ratios = np.zeros_like(abs_residual)
    positive = denominator > 0.0
    ratios[positive] = abs_residual[positive] / denominator[positive]
    ratios[~positive & (abs_residual > 0.0)] = math.inf
    componentwise = float(np.max(ratios, initial=0.0))

    difference = solution - system.x_star
    l2 = _safe_ratio(
        float(np.linalg.norm(difference)), float(np.linalg.norm(system.x_star))
    )
    linf = _safe_ratio(
        float(np.max(np.abs(difference), initial=0.0)),
        float(np.max(np.abs(system.x_star), initial=0.0)),
    )
    return AccuracyMetrics(normwise, componentwise, l2, linf)


def verify_solution(
    system: EvaluationSystem,
    solution: object,
    *,
    status: int = 0,
    input_mutated: bool = False,
) -> VerificationResult:
    thresholds = AccuracyThresholds.from_tolerance(system.public.tolerance)
    if status != 0:
        return VerificationResult(False, "nonzero_status", None, thresholds)
    if input_mutated:
        return VerificationResult(False, "input_mutation", None, thresholds)
    raw = np.asarray(solution)
    if raw.shape != (system.public.matrix.n,) or raw.dtype.kind not in "fiu":
        return VerificationResult(False, "invalid_shape_or_dtype", None, thresholds)
    candidate = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(candidate)):
        return VerificationResult(False, "nonfinite_solution", None, thresholds)
    try:
        metrics = accuracy_metrics(system, candidate)
    except (FloatingPointError, OverflowError, ValueError):
        return VerificationResult(False, "verification_error", None, thresholds)
    values = (
        metrics.normwise_backward_error,
        metrics.componentwise_backward_error,
        metrics.relative_l2_forward_error,
        metrics.relative_linf_forward_error,
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        return VerificationResult(False, "nonfinite_metric", metrics, thresholds)
    failures = []
    if metrics.normwise_backward_error > thresholds.normwise_backward_error:
        failures.append("normwise_backward_error")
    if metrics.componentwise_backward_error > thresholds.componentwise_backward_error:
        failures.append("componentwise_backward_error")
    if metrics.forward_error > thresholds.forward_error:
        failures.append("forward_error")
    return VerificationResult(
        not failures,
        ",".join(failures) if failures else None,
        metrics,
        thresholds,
    )
