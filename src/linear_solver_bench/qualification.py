"""Uniform numerical-rank and guarded one-norm condition qualification."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, onenormest

from .dataset import identity_sha256

QUALIFICATION_SCHEMA_VERSION = 2
QUALIFICATION_KIND = "linear-solver-bench-condition-evidence-v2"
CONDITION_METHOD = "spqr-guarded-onenormest-v1"
CONDITION_NORM = "1"
FLOAT64_EPSILON = float(np.finfo(np.float64).eps)
CONDITION_SAFETY_FACTOR = 100.0
FORWARD_ERROR_FACTOR = 5.0
ESTIMATOR_GUARD_FACTOR = 3.0
ESTIMATOR_REPEATS = 3
ONENORMEST_T = 4
ONENORMEST_ITMAX = 10
TOLERANCE_TIERS = (1.0e-4, 1.0e-3, 1.0e-2, 4.0e-2)
SOLVE_RESIDUAL_LIMIT = 1.0e-8


@dataclass(frozen=True)
class InverseNormEvidence:
    estimates: tuple[float, ...]
    seeds: tuple[int, ...]
    maximum_solve_residual: float

    def as_record(self) -> dict[str, object]:
        return {
            "estimates": list(self.estimates),
            "seeds": list(self.seeds),
            "maximum_solve_residual": self.maximum_solve_residual,
        }


def condition_ceiling(tolerance: float) -> float:
    value = float(tolerance)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    return value * FORWARD_ERROR_FACTOR / (
        CONDITION_SAFETY_FACTOR * FLOAT64_EPSILON
    )


def minimum_safe_tolerance(condition_estimate: float) -> float:
    value = float(condition_estimate)
    if not math.isfinite(value) or value < 1.0:
        raise ValueError("condition estimate must be finite and at least one")
    return (
        CONDITION_SAFETY_FACTOR * value * FLOAT64_EPSILON
        / FORWARD_ERROR_FACTOR
    )


def _matrix_one_norm(matrix: sparse.spmatrix) -> float:
    columns = np.asarray(abs(matrix).sum(axis=0)).reshape(-1)
    result = float(np.max(columns, initial=0.0))
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("matrix one-norm must be finite and positive")
    return result


def maximum_column_two_norm(matrix: sparse.spmatrix) -> float:
    """Return the largest column 2-norm without unsafe unscaled squaring."""
    if not sparse.issparse(matrix) or matrix.ndim != 2:
        raise ValueError("column norm requires a sparse matrix")
    csc = sparse.csc_matrix(matrix, dtype=np.float64, copy=False)
    if not np.all(np.isfinite(csc.data)):
        raise ValueError("column norm requires finite matrix values")
    scale = float(np.max(np.abs(csc.data), initial=0.0))
    if scale == 0.0:
        return 0.0
    squared = np.square(csc.data / scale)
    starts = csc.indptr[:-1]
    stops = csc.indptr[1:]
    nonempty = stops > starts
    column_sums = np.zeros(csc.shape[1], dtype=np.float64)
    column_sums[nonempty] = np.add.reduceat(squared, starts[nonempty])
    result = scale * math.sqrt(float(np.max(column_sums, initial=0.0)))
    if not math.isfinite(result):
        raise ValueError("column norm is not finite")
    return result


def spqr_rank_tolerance(matrix: sparse.spmatrix) -> float:
    """Return the explicit SuiteSparseQR default-style numerical-rank cutoff."""
    if not sparse.issparse(matrix) or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("SPQR rank tolerance requires a square sparse matrix")
    n = int(matrix.shape[0])
    tolerance = (
        20.0 * (2 * n) * FLOAT64_EPSILON * maximum_column_two_norm(matrix)
    )
    if not math.isfinite(tolerance):
        raise ValueError("SPQR rank tolerance is not finite")
    return tolerance


def _seed(case_id: str, repeat: int) -> int:
    payload = (
        f"linear-solver-bench-condition-onenormest-v1\0{case_id}\0{repeat}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _checked_solve(
    solve: Callable[[np.ndarray], object], value: np.ndarray
) -> np.ndarray:
    result = np.asarray(solve(value), dtype=np.float64)
    if result.shape != value.shape or not np.all(np.isfinite(result)):
        raise ValueError("factor solve returned an invalid value")
    return result


def inverse_one_norm_evidence(
    matrix: sparse.spmatrix,
    solve: Callable[[np.ndarray], object],
    solve_transpose: Callable[[np.ndarray], object],
    *,
    case_id: str,
    repeats: int = ESTIMATOR_REPEATS,
    t: int = ONENORMEST_T,
    itmax: int = ONENORMEST_ITMAX,
) -> InverseNormEvidence:
    """Estimate ``||A^-1||_1`` reproducibly from ordinary/transpose solves."""
    if not sparse.issparse(matrix) or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("inverse-norm estimation requires a square sparse matrix")
    if not case_id or not case_id.isascii():
        raise ValueError("case_id must be nonempty ASCII")
    if (
        type(repeats) is not int
        or repeats < 1
        or type(t) is not int
        or t < 1
        or type(itmax) is not int
        or itmax < 2
    ):
        raise ValueError("inverse-norm estimator controls are invalid")
    n = int(matrix.shape[0])
    operator = LinearOperator(
        shape=(n, n),
        dtype=np.dtype(np.float64),
        matvec=lambda value: _checked_solve(solve, value),
        rmatvec=lambda value: _checked_solve(solve_transpose, value),
    )
    probe = np.where(np.arange(n) % 2, -1.0, 1.0)
    matrix_one_norm = _matrix_one_norm(matrix)
    transpose_one_norm = _matrix_one_norm(matrix.T)
    residuals = []
    for operator_matrix, solver, operator_norm in (
        (matrix, solve, matrix_one_norm),
        (matrix.T, solve_transpose, transpose_one_norm),
    ):
        solution = _checked_solve(solver, probe)
        residual = np.asarray(operator_matrix @ solution - probe)
        denominator = operator_norm * float(np.linalg.norm(solution, 1)) + n
        residuals.append(float(np.linalg.norm(residual, 1)) / denominator)
    maximum_residual = max(residuals)
    if (
        not math.isfinite(maximum_residual)
        or maximum_residual > SOLVE_RESIDUAL_LIMIT
    ):
        raise ValueError("factor solve residual is too large")

    seeds = tuple(_seed(case_id, repeat) for repeat in range(repeats))
    estimates = []
    random_state = np.random.get_state()
    try:
        for seed in seeds:
            np.random.seed(seed)
            estimate = float(onenormest(operator, t=t, itmax=itmax))
            if not math.isfinite(estimate) or estimate <= 0.0:
                raise ValueError("inverse-norm estimate must be finite and positive")
            estimates.append(estimate)
    finally:
        np.random.set_state(random_state)
    return InverseNormEvidence(
        estimates=tuple(estimates),
        seeds=seeds,
        maximum_solve_residual=maximum_residual,
    )


def _empty_condition(
    *,
    status: str,
    rank_estimate: int | None,
    rank_tolerance: float | None,
    reason: str,
) -> dict[str, object]:
    return {
        "status": status,
        "method": CONDITION_METHOD,
        "rank_estimate": rank_estimate,
        "rank_tolerance": rank_tolerance,
        "condition_norm": None,
        "condition_estimate_raw": None,
        "estimator_guard_factor": None,
        "condition_estimate": None,
        "roundoff_budget": None,
        "minimum_safe_tolerance": None,
        "supported_tolerances": [],
        "tightest_supported_tolerance": None,
        "reason": reason,
    }


def indeterminate_condition(reason: str) -> dict[str, object]:
    if not reason:
        raise ValueError("indeterminate condition evidence requires a reason")
    return _empty_condition(
        status="indeterminate",
        rank_estimate=None,
        rank_tolerance=None,
        reason=reason,
    )


def condition_from_factor(
    matrix: sparse.spmatrix,
    *,
    case_id: str,
    rank_estimate: int,
    rank_tolerance: float,
    solve: Callable[[np.ndarray], object],
    solve_transpose: Callable[[np.ndarray], object],
) -> tuple[dict[str, object], InverseNormEvidence | None]:
    """Build one uniform rank/condition record from a reusable SPQR factor."""
    n = int(matrix.shape[0])
    if type(rank_estimate) is not int or not 0 <= rank_estimate <= n:
        raise ValueError("SPQR rank estimate is invalid")
    if rank_estimate < n:
        return (
            _empty_condition(
                status="rank_deficient",
                rank_estimate=rank_estimate,
                rank_tolerance=rank_tolerance,
                reason="SuiteSparseQR reports numerical rank below n",
            ),
            None,
        )
    evidence = inverse_one_norm_evidence(
        matrix, solve, solve_transpose, case_id=case_id
    )
    raw = max(1.0, _matrix_one_norm(matrix) * max(evidence.estimates))
    guarded = raw * ESTIMATOR_GUARD_FACTOR
    floor = minimum_safe_tolerance(guarded)
    supported = tuple(tolerance for tolerance in TOLERANCE_TIERS if floor <= tolerance)
    return (
        {
            "status": "estimated",
            "method": CONDITION_METHOD,
            "rank_estimate": rank_estimate,
            "rank_tolerance": rank_tolerance,
            "condition_norm": CONDITION_NORM,
            "condition_estimate_raw": raw,
            "estimator_guard_factor": ESTIMATOR_GUARD_FACTOR,
            "condition_estimate": guarded,
            "roundoff_budget": CONDITION_SAFETY_FACTOR
            * guarded
            * FLOAT64_EPSILON,
            "minimum_safe_tolerance": floor,
            "supported_tolerances": list(supported),
            "tightest_supported_tolerance": min(supported) if supported else None,
            "reason": None,
        },
        evidence,
    )


def qualify_matrix(
    matrix: sparse.spmatrix, *, case_id: str
) -> tuple[dict[str, object], InverseNormEvidence | None]:
    """Factor a matrix with SPQR and return uniform guarded 1-norm evidence."""
    try:
        from sksparse.spqr import spqr_factor
    except ImportError as exc:
        raise RuntimeError(
            "condition qualification requires scikit-sparse with SPQR"
        ) from exc
    csc = sparse.csc_array(matrix, dtype=np.float64, copy=True)
    csc.sum_duplicates()
    csc.eliminate_zeros()
    csc.sort_indices()
    rank_tolerance = spqr_rank_tolerance(csc)
    factor = spqr_factor(
        csc, use_singletons=True, order="colamd", tol=rank_tolerance
    )
    return condition_from_factor(
        csc,
        case_id=case_id,
        rank_estimate=int(factor.rank),
        rank_tolerance=rank_tolerance,
        solve=lambda value: factor.solve(value, transpose=False),
        solve_transpose=lambda value: factor.solve(value, transpose=True),
    )


def qualification_document(
    *,
    catalogue_sha256: str,
    input_manifest_sha256: str,
    expected_case_count: int,
    results: Sequence[Mapping[str, object]],
    toolchain: Mapping[str, object],
) -> dict[str, object]:
    """Build the content-identified aggregate v2 qualification artifact."""
    ordered = sorted(
        (dict(result) for result in results), key=lambda item: item["index"]
    )
    indices = tuple(result.get("index") for result in ordered)
    if (
        type(expected_case_count) is not int
        or expected_case_count <= 0
        or any(type(index) is not int for index in indices)
        or any(not 0 <= index < expected_case_count for index in indices)
        or len(set(indices)) != len(indices)
    ):
        raise ValueError("qualification result indices are invalid")
    statuses = Counter(str(result["condition"]["status"]) for result in ordered)
    methods = Counter(str(result["condition"]["method"]) for result in ordered)
    body = {
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "kind": QUALIFICATION_KIND,
        "catalogue_sha256": catalogue_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "expected_case_count": expected_case_count,
        "received_case_count": len(ordered),
        "complete": len(ordered) == expected_case_count,
        "policy": {
            "condition_norm": CONDITION_NORM,
            "method": CONDITION_METHOD,
            "rank_method": "SuiteSparseQR",
            "condition_safety_factor": CONDITION_SAFETY_FACTOR,
            "forward_error_factor": FORWARD_ERROR_FACTOR,
            "float64_epsilon": FLOAT64_EPSILON,
            "minimum_safe_tolerance_formula": (
                "100*guarded_condition*float64_epsilon/5"
            ),
            "tolerance_tiers": list(TOLERANCE_TIERS),
            "condition_ceilings": {
                format(value, ".17g"): condition_ceiling(value)
                for value in TOLERANCE_TIERS
            },
            "estimator": {
                "guard_factor": ESTIMATOR_GUARD_FACTOR,
                "repeats": ESTIMATOR_REPEATS,
                "onenormest_t": ONENORMEST_T,
                "onenormest_itmax": ONENORMEST_ITMAX,
                "solve_residual_limit": SOLVE_RESIDUAL_LIMIT,
            },
        },
        "toolchain": dict(toolchain),
        "status_counts": dict(sorted(statuses.items())),
        "method_counts": dict(sorted(methods.items())),
        "results": ordered,
    }
    return {**body, "condition_results_sha256": identity_sha256(body)}
