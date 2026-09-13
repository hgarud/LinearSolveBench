"""Independent terminal metrics and explicit, fail-closed acceptance gates."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .accuracy import ACCURACY_CONTRACT_ID, AccuracyThresholds
from .models import EvaluationSystem


@dataclass(frozen=True)
class AccuracyMetrics:
    normwise_backward_error: float | None
    componentwise_backward_error: float | None
    relative_l2_forward_error: float | None
    relative_linf_forward_error: float | None
    relative_residual: float | None = None
    diagnostic_reasons: dict[str, str] = field(default_factory=dict)

    @property
    def forward_error(self) -> float | None:
        if (
            self.relative_l2_forward_error is None
            or self.relative_linf_forward_error is None
        ):
            return None
        return max(self.relative_l2_forward_error, self.relative_linf_forward_error)


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    failure: str | None
    metrics: AccuracyMetrics | None
    thresholds: AccuracyThresholds
    contract_id: str = ACCURACY_CONTRACT_ID

    def to_dict(self) -> dict[str, object]:
        metrics = asdict(self.metrics) if self.metrics else None
        thresholds = asdict(self.thresholds)
        if self.contract_id == ACCURACY_CONTRACT_ID:
            # Keep the established v1 report shape for existing callers.
            thresholds.pop("relative_residual")
            if metrics is not None:
                metrics.pop("relative_residual")
                metrics.pop("diagnostic_reasons")
        result = {
            "ok": self.ok,
            "failure": self.failure,
            "metrics": metrics,
            "thresholds": thresholds,
        }
        if self.contract_id != ACCURACY_CONTRACT_ID:
            result["accuracy_contract_id"] = self.contract_id
        return result


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return math.inf
    if denominator > 0.0:
        return numerator / denominator
    return 0.0 if numerator == 0.0 else math.inf


def _max_abs(vector: np.ndarray) -> float:
    return float(np.max(np.abs(vector), initial=0.0))


def _relative_l2(numerator: np.ndarray, denominator: np.ndarray) -> float:
    """Evaluate a norm ratio without squaring large values or tiny values.

    Splitting scales into mantissas and exponents also avoids overflowing the
    individual norms when their ratio is representable.
    """
    nscale, dscale = _max_abs(numerator), _max_abs(denominator)
    if not math.isfinite(nscale) or not math.isfinite(dscale):
        return math.inf
    if not nscale or not dscale:
        return _safe_ratio(nscale, dscale)
    nunit = float(np.linalg.norm(numerator / nscale))
    dunit = float(np.linalg.norm(denominator / dscale))
    nmantissa, nexponent = math.frexp(nscale)
    dmantissa, dexponent = math.frexp(dscale)
    try:
        return math.ldexp(
            (nmantissa / dmantissa) * (nunit / dunit), nexponent - dexponent
        )
    except OverflowError:
        return math.inf


def accuracy_metrics(system: EvaluationSystem, solution: np.ndarray) -> AccuracyMetrics:
    public = system.public
    matrix = public.matrix.to_scipy()
    reasons: dict[str, str] = {}

    def record(name: str, value: float, reason: str) -> float | None:
        if not math.isfinite(value) or value < 0.0:
            reasons[name] = reason
            return None
        return value

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        residual = np.asarray(public.b - matrix @ solution, dtype=np.float64)
        abs_residual = np.abs(residual)
        abs_matrix = abs(matrix)
        matrix_norm = float(
            np.max(np.asarray(abs_matrix.sum(axis=1)).reshape(-1), initial=0.0)
        )
        normwise_denominator = matrix_norm * _max_abs(solution) + _max_abs(public.b)
        normwise = record(
            "normwise_backward_error",
            _safe_ratio(_max_abs(residual), normwise_denominator),
            "nonfinite_residual_or_normwise_denominator",
        )
        denominator = np.asarray(abs_matrix @ np.abs(solution), dtype=np.float64)
        denominator += np.abs(public.b)
        if np.all(np.isfinite(denominator)) and np.all(np.isfinite(abs_residual)):
            ratios = np.zeros_like(abs_residual)
            positive = denominator > 0.0
            ratios[positive] = abs_residual[positive] / denominator[positive]
            ratios[~positive & (abs_residual > 0.0)] = math.inf
            componentwise_value = float(np.max(ratios, initial=0.0))
        else:
            componentwise_value = math.inf
        componentwise = record(
            "componentwise_backward_error",
            componentwise_value,
            "nonfinite_residual_or_componentwise_ratio",
        )
        relative_residual = record(
            "relative_residual",
            _relative_l2(residual, public.b),
            "zero_rhs_with_nonzero_residual"
            if not np.any(public.b) and np.any(residual)
            else "nonfinite_residual_ratio",
        )
        if system.x_star is None:
            l2 = linf = None
            reasons["relative_l2_forward_error"] = "reference_unavailable"
            reasons["relative_linf_forward_error"] = "reference_unavailable"
        else:
            difference = solution - system.x_star
            l2 = record(
                "relative_l2_forward_error",
                _relative_l2(difference, system.x_star),
                "nonfinite_forward_error_or_zero_reference",
            )
            linf = record(
                "relative_linf_forward_error",
                _safe_ratio(_max_abs(difference), _max_abs(system.x_star)),
                "nonfinite_forward_error_or_zero_reference",
            )
    return AccuracyMetrics(
        normwise, componentwise, l2, linf, relative_residual, reasons
    )


def verify_solution(
    system: EvaluationSystem,
    solution: object,
    *,
    status: int = 0,
    input_mutated: bool = False,
    contract_id: str = ACCURACY_CONTRACT_ID,
) -> VerificationResult:
    thresholds = AccuracyThresholds.for_contract(contract_id, system.public.tolerance)

    def result(
        failure: str | None, metrics: AccuracyMetrics | None = None
    ) -> VerificationResult:
        return VerificationResult(
            failure is None, failure, metrics, thresholds, contract_id
        )

    if thresholds.forward_error is not None and system.reference_kind != "manufactured":
        return result("missing_manufactured_reference")
    if thresholds.forward_error is None and system.reference_kind == "manufactured":
        return result("invalid_reference_kind")
    if status != 0:
        return result("nonzero_status")
    if input_mutated:
        return result("input_mutation")
    raw = np.asarray(solution)
    if raw.shape != (system.public.matrix.n,) or raw.dtype.kind not in "fiu":
        return result("invalid_shape_or_dtype")
    with np.errstate(over="ignore", invalid="ignore"):
        candidate = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(candidate)):
        return result("nonfinite_solution")
    try:
        metrics = accuracy_metrics(system, candidate)
    except (FloatingPointError, OverflowError, ValueError):
        return result("verification_error")
    gates = {
        "normwise_backward_error": (
            metrics.normwise_backward_error,
            thresholds.normwise_backward_error,
        ),
        "componentwise_backward_error": (
            metrics.componentwise_backward_error,
            thresholds.componentwise_backward_error,
        ),
        "forward_error": (metrics.forward_error, thresholds.forward_error),
        "relative_residual": (metrics.relative_residual, thresholds.relative_residual),
    }
    required = [
        (name, value, limit)
        for name, (value, limit) in gates.items()
        if limit is not None
    ]
    if any(
        value is None or not math.isfinite(value) or value < 0
        for _, value, _ in required
    ):
        return result("nonfinite_metric", metrics)
    failures = [name for name, value, limit in required if value > limit]
    return result(",".join(failures) if failures else None, metrics)
