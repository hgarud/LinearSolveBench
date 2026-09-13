"""Versioned manufactured inputs and independent offline NS qualification."""

from __future__ import annotations

import hashlib
import hmac
import math
from collections.abc import Mapping
from decimal import Decimal, localcontext

import numpy as np
import scipy
from scipy.sparse.linalg import LinearOperator, gmres, spilu, splu

from .dataset import canonical_json
from .manifests import (
    ITERATIVE_QUALIFICATION_CONFIG,
    ITERATIVE_QUALIFICATION_METHOD,
    QUALIFICATION_LIMITS,
    RHS_SCHEME,
    validate_qualification,
)
from .models import CsrMatrix, EvaluationSystem, MatrixInput
from .verify import accuracy_metrics


def matrix_digest(matrix: CsrMatrix) -> str:
    """Hash canonical numerical arrays, independent of NPZ compression metadata."""
    value = hashlib.sha256(b"linear-solver-bench-csr-float64-v2\0")
    value.update(matrix.n.to_bytes(8, "little"))
    value.update(np.asarray(matrix.row_offsets, dtype="<u8").tobytes())
    value.update(np.asarray(matrix.column_indices, dtype="<u4").tobytes())
    value.update(np.asarray(matrix.values, dtype="<f8").tobytes())
    return value.hexdigest()


def vector_digest(vector: np.ndarray) -> str:
    value = hashlib.sha256(b"linear-solver-bench-vector-float64-v2\0")
    value.update(len(vector).to_bytes(8, "little"))
    value.update(np.asarray(vector, dtype="<f8").tobytes())
    return value.hexdigest()


def system_digest(system: EvaluationSystem) -> str:
    body = {
        "case_id": system.case_id,
        "matrix_sha256": matrix_digest(system.public.matrix),
        "b_sha256": vector_digest(system.public.b),
        "x0_sha256": vector_digest(system.public.x0),
        "tolerance": system.public.tolerance,
        "reference_kind": system.reference_kind,
        "reference_sha256": None
        if system.x_star is None
        else vector_digest(system.x_star),
    }
    return hashlib.sha256(canonical_json(body).encode("ascii")).hexdigest()


def expected_flash_system_digest(case: Mapping[str, object]) -> str:
    """Bind prepared replay inputs to the arrays frozen in the public release.

    The pilot's four-file capture format contains no numerical reference array.
    Its complete system identity can therefore be reconstructed from the release
    without trusting a prepared archive's self-declared hash.
    """
    source = case["source"]
    body = {
        "case_id": case["case_id"],
        "matrix_sha256": source["matrix_sha256"],
        "b_sha256": source["b_sha256"],
        "x0_sha256": source["x0_sha256"],
        "tolerance": float(case["tolerance"]),
        "reference_kind": "none",
        "reference_sha256": None,
    }
    return hashlib.sha256(canonical_json(body).encode("ascii")).hexdigest()


def rademacher_target(
    *,
    release_id: str,
    family: str,
    matrix_sha256: str,
    n: int,
    draw_index: int,
    key: bytes,
) -> np.ndarray:
    """Generate +/-1 directly from HMAC bits; no library RNG or version dependence.

    Each SHA-256 block supplies 256 bits, least significant bit first per byte.
    A one bit maps to +1 and a zero bit to -1. Every target has exactly unit RMS.
    """
    if not isinstance(key, bytes) or len(key) < 16:
        raise ValueError("RHS key must contain at least 16 bytes")
    if type(n) is not int or n <= 0 or type(draw_index) is not int or draw_index < 0:
        raise ValueError("invalid manufactured target dimensions or draw index")
    domain = canonical_json(
        {
            "scheme": RHS_SCHEME,
            "release_id": release_id,
            "family": family,
            "matrix_sha256": matrix_sha256,
            "rhs_kind": "rademacher",
            "draw_index": draw_index,
        }
    ).encode("ascii")
    raw = bytearray()
    for block in range((n + 255) // 256):
        raw.extend(
            hmac.digest(key, domain + b"\0" + block.to_bytes(8, "little"), "sha256")
        )
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")[:n]
    return bits.astype(np.float64) * 2 - 1


def manufacture_ns(
    case: Mapping[str, object],
    release: Mapping[str, object],
    key: bytes,
    matrix: CsrMatrix,
) -> EvaluationSystem:
    if hashlib.sha256(key).hexdigest() != release["rhs"]["key_id"]:
        raise ValueError("RHS key does not match the release commitment")
    truth = rademacher_target(
        release_id=release["release_id"],
        family=release["family"],
        matrix_sha256=matrix_digest(matrix),
        n=matrix.n,
        draw_index=release["rhs"]["draw_index"],
        key=key,
    )
    return EvaluationSystem(
        case_id=case["case_id"],
        public=MatrixInput(
            matrix=matrix,
            b=matrix.matvec(truth),
            x0=np.zeros(matrix.n),
            tolerance=case["tolerance"],
        ),
        x_star=truth,
        reference_kind="manufactured",
    )


def _extended_residual(
    matrix: CsrMatrix, rhs: np.ndarray, vector: np.ndarray
) -> np.ndarray:
    """Evaluate b-Ax with extra precision; this is not interval arithmetic."""
    residual = np.empty(matrix.n, dtype=np.longdouble)
    if np.finfo(np.longdouble).eps < np.finfo(np.float64).eps:
        extended_rhs = rhs.astype(np.longdouble)
        extended_vector = vector.astype(np.longdouble)
        for row in range(matrix.n):
            start, stop = int(matrix.row_offsets[row]), int(matrix.row_offsets[row + 1])
            dot = np.sum(
                matrix.values[start:stop].astype(np.longdouble)
                * extended_vector[matrix.column_indices[start:stop]],
                dtype=np.longdouble,
            )
            residual[row] = extended_rhs[row] - dot
    else:
        # Decimal.from_float preserves the binary64 operands exactly; products
        # and sums then use 80 decimal digits. The result remains empirical
        # higher-precision evidence, not an outward-rounded certificate.
        decimal_vector = [Decimal.from_float(float(value)) for value in vector]
        with localcontext() as context:
            context.prec = 80
            for row in range(matrix.n):
                start, stop = (
                    int(matrix.row_offsets[row]),
                    int(matrix.row_offsets[row + 1]),
                )
                dot = sum(
                    (
                        Decimal.from_float(float(value)) * decimal_vector[column]
                        for value, column in zip(
                            matrix.values[start:stop],
                            matrix.column_indices[start:stop],
                            strict=True,
                        )
                    ),
                    Decimal(0),
                )
                residual[row] = np.longdouble(
                    str(Decimal.from_float(float(rhs[row])) - dot)
                )
    return residual


def qualify_ns(system: EvaluationSystem, *, refine: bool = False) -> dict:
    """Establish empirical feasibility against every gate with a tenfold margin.

    Sparse LU is an offline reference, not the timing reference or a condition
    certificate. refine=True always performs exactly one correction using an
    extended-precision residual and the same LU factors, then checks the result.
    It never changes the target, matrix, RHS, or tolerances. The distinct method
    ID records that fixed procedure. RHS formation is independently audited with
    the same higher-precision evaluator; this does not certify forward error.
    """
    if type(refine) is not bool:
        raise ValueError("refine must be a boolean")
    if system.x_star is None or system.reference_kind != "manufactured":
        raise ValueError("NS qualification requires the manufactured target")
    matrix = system.public.matrix
    try:
        factor = splu(matrix.to_scipy().tocsc())
        reference = factor.solve(system.public.b)
        if not np.all(np.isfinite(reference)):
            raise ValueError("independent reference solution is not finite")
        if refine:
            residual = _extended_residual(matrix, system.public.b, reference)
            with np.errstate(over="ignore", invalid="ignore"):
                correction_rhs = np.asarray(residual, dtype=np.float64)
            if not np.all(np.isfinite(correction_rhs)):
                raise ValueError("refinement residual is not finite float64")
            with np.errstate(over="ignore", invalid="ignore"):
                reference = reference + factor.solve(correction_rhs)
            if not np.all(np.isfinite(reference)):
                raise ValueError("refined reference solution is not finite")
        result = accuracy_metrics(system, reference)
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        raise ValueError(
            "independent NS reference solution could not be computed"
        ) from exc
    metrics = {}
    for name in QUALIFICATION_LIMITS:
        value = getattr(result, name)
        if value is None:
            raise ValueError(f"independent NS qualification produced invalid {name}")
        metrics[name] = float(value)
    formation = _extended_residual(matrix, system.public.b, system.x_star)
    maximum = np.max(np.abs(formation), initial=np.longdouble(0))
    norm_b = np.max(
        np.abs(system.public.b.astype(np.longdouble)), initial=np.longdouble(0)
    )
    relative = (
        float(maximum / norm_b) if norm_b else (0.0 if maximum == 0 else float("inf"))
    )
    evidence = {
        "method": "independent-sparse-lu-refined-v1"
        if refine
        else "independent-sparse-lu-v1",
        "qualified": True,
        "system_sha256": system_digest(system),
        "metrics": metrics,
        "formation": {
            "method": "extended-precision-residual-v1",
            "relative_linf_residual": relative,
            "verified_forward_bound": None,
        },
        "uncertainty": "empirical-feasibility-only",
    }
    validate_qualification(evidence, system_digest(system))
    return evidence


def qualify_ns_iterative(system: EvaluationSystem) -> dict:
    """Qualify a fixed target with an independent, bounded-fill offline reference.

    The reference solves a row/column-equilibrated system from zero using ILU
    and GMRES, followed by exactly two extended-residual correction solves.
    Only the original stored A,b and manufactured target determine acceptance.
    The target is never a solver input. This is empirical feasibility evidence,
    without an inverse-norm bound or a candidate performance claim.

    ILU fill controls do not limit every temporary allocation. Operators must
    run this expensive offline procedure in a process with explicit memory and
    wall-time limits. Library versions and the complete fixed configuration
    accompany successful evidence; unsuccessful solves do not admit a case.
    """
    if system.x_star is None or system.reference_kind != "manufactured":
        raise ValueError("NS qualification requires the manufactured target")
    config = ITERATIVE_QUALIFICATION_CONFIG
    matrix = system.public.matrix
    scaled = matrix.to_scipy().copy()
    row_maximum = np.asarray(abs(scaled).max(axis=1).toarray()).ravel()
    if np.any(row_maximum == 0):
        raise ValueError("independent reference cannot equilibrate an empty row")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        row_scale = 1.0 / row_maximum
        scaled.data *= np.repeat(row_scale, np.diff(scaled.indptr))
        column_maximum = np.asarray(abs(scaled).max(axis=0).toarray()).ravel()
        column_scale = 1.0 / column_maximum
        scaled.data *= column_scale[scaled.indices]
    if (
        not np.all(np.isfinite(row_scale))
        or not np.all(np.isfinite(column_scale))
        or not np.all(np.isfinite(scaled.data))
        or np.any(scaled.data == 0)
    ):
        raise ValueError(
            "independent reference equilibration is not finite and nonzero"
        )
    try:
        factor = spilu(
            scaled.tocsc(),
            drop_tol=config["drop_tol"],
            fill_factor=config["fill_factor"],
            drop_rule=config["drop_rule"],
            permc_spec=config["permc_spec"],
            diag_pivot_thresh=config["diag_pivot_thresh"],
            options={"Equil": config["superlu_equilibration"]},
        )
        preconditioner = LinearOperator(scaled.shape, factor.solve, dtype=np.float64)
        solves = []

        def solve(rhs: np.ndarray) -> np.ndarray:
            iterations = 0

            def count_iteration(_residual: float) -> None:
                nonlocal iterations
                iterations += 1

            with np.errstate(over="ignore", invalid="ignore"):
                scaled_rhs = row_scale * rhs
            if not np.all(np.isfinite(scaled_rhs)):
                raise ValueError("scaled reference RHS is not finite")
            vector, info = gmres(
                scaled,
                scaled_rhs,
                x0=np.zeros(matrix.n),
                rtol=config["rtol"],
                atol=config["atol"],
                restart=config["restart"],
                maxiter=config["max_restart_cycles"],
                M=preconditioner,
                callback=count_iteration,
                callback_type="pr_norm",
            )
            solves.append({"info": int(info), "inner_iterations": iterations})
            with np.errstate(over="ignore", invalid="ignore"):
                vector = column_scale * vector
            if info != 0 or not np.all(np.isfinite(vector)):
                raise ValueError("independent GMRES reference did not converge")
            return vector

        reference = solve(system.public.b)
        for _ in range(config["refinement_steps"]):
            with np.errstate(over="ignore", invalid="ignore"):
                residual = np.asarray(
                    _extended_residual(matrix, system.public.b, reference),
                    dtype=np.float64,
                )
                reference = reference + solve(residual)
            if not np.all(np.isfinite(reference)):
                raise ValueError("refined reference solution is not finite")
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        raise ValueError(
            "independent iterative NS reference could not be computed"
        ) from exc

    result = accuracy_metrics(system, reference)
    metrics = {}
    for name in QUALIFICATION_LIMITS:
        value = getattr(result, name)
        if value is None:
            raise ValueError(f"independent NS qualification produced invalid {name}")
        metrics[name] = float(value)
    formation = _extended_residual(matrix, system.public.b, system.x_star)
    maximum = np.max(np.abs(formation), initial=np.longdouble(0))
    norm_b = np.max(
        np.abs(system.public.b.astype(np.longdouble)), initial=np.longdouble(0)
    )
    relative = (
        float(maximum / norm_b) if norm_b else (0.0 if maximum == 0 else float("inf"))
    )
    evidence = {
        "method": ITERATIVE_QUALIFICATION_METHOD,
        "qualified": True,
        "system_sha256": system_digest(system),
        "metrics": metrics,
        "formation": {
            "method": "extended-precision-residual-v1",
            "relative_linf_residual": relative,
            "verified_forward_bound": None,
        },
        "uncertainty": "empirical-feasibility-only",
        "reference_solver": {
            "configuration": dict(config),
            "implementation": {"numpy": np.__version__, "scipy": scipy.__version__},
            "solves": solves,
        },
    }
    validate_qualification(evidence, system_digest(system))
    return evidence


def qualify_ns_pyamg(system: EvaluationSystem) -> dict:
    """Run the optional fixed multigrid offline reference without eager imports."""
    from .reference_pyamg import qualify_ns_pyamg as qualify

    return qualify(system)


def qualify_ns_pyamg_inexact(system: EvaluationSystem) -> dict:
    """Run the optional v2 reference with fixed inexact inner corrections."""
    from .reference_pyamg import qualify_ns_pyamg_inexact as qualify

    return qualify(system)


def _binary64_units(value: float) -> int:
    """Represent a finite binary64 number exactly in units of 2**-1074."""
    numerator, denominator = float(value).as_integer_ratio()
    return numerator << (1074 - (denominator.bit_length() - 1))


def _rounded_ratio(numerator: int, denominator: int, *, upward: bool) -> float:
    """Round an exact nonnegative rational outwards, checking the direction."""
    if numerator < 0 or denominator <= 0:
        raise ValueError("certificate ratio requires a nonnegative finite value")
    try:
        value = numerator / denominator
    except OverflowError as exc:
        raise ValueError("certificate bound cannot be represented in float64") from exc
    if not math.isfinite(value):
        raise ValueError("certificate bound cannot be represented in float64")
    rounded_numerator, rounded_denominator = value.as_integer_ratio()
    comparison = rounded_numerator * denominator - numerator * rounded_denominator
    if (upward and comparison < 0) or (not upward and comparison > 0):
        value = math.nextafter(value, math.inf if upward else -math.inf)
    if not math.isfinite(value):
        raise ValueError("certificate bound cannot be represented in float64")
    bounded_numerator, bounded_denominator = value.as_integer_ratio()
    check = bounded_numerator * denominator - numerator * bounded_denominator
    if (upward and check < 0) or (not upward and check > 0):
        raise ValueError("could not round certificate bound in the required direction")
    return value


def qualify_ns_dominance(system: EvaluationSystem) -> dict:
    """Certify stored-system consistency for a strictly row-dominant NS case.

    Every coefficient is an exact integer multiple of 2**-1074. With a +/-1
    manufactured target, row dot products and formation residuals are therefore
    accumulated exactly using integers, including subnormal coefficients.

    Let d = min_i (|a_ii| - sum_{j != i}|a_ij|) > 0 and e = b - A x_target.
    Taking the largest component of any vector proves ||A y||inf >= d||y||inf.
    Thus A is invertible and ||A^-1||inf <= 1/d (the row-dominance bound):
    https://doi.org/10.1016/0024-3795(75)90112-3
    The exact stored-system solution differs from x_target by at most
    ||e||inf/d in Linf. Since ||x_target||inf=1 and ||x_target||2=sqrt(n), the
    same bound covers both relative forward discrepancies.

    Required metric values below are *measured* by the public binary64 verifier
    at x_target, an explicit feasible witness. They are not verified rounding
    bounds for the verifier's floating-point operations or an independent LU
    solve. The separate exact-arithmetic certificate proves stored-system
    forward consistency. Both witness metrics and that bound need a tenfold
    margin. This offline path does not make x_target available to candidates.
    """
    target = system.x_star
    if (
        system.reference_kind != "manufactured"
        or target is None
        or not np.all((target == 1.0) | (target == -1.0))
    ):
        raise ValueError("dominance qualification requires a Rademacher target")
    matrix = system.public.matrix
    minimum_margin = None
    maximum_error = maximum_rhs = 0
    for row in range(matrix.n):
        start, stop = int(matrix.row_offsets[row]), int(matrix.row_offsets[row + 1])
        diagonal = row_sum = dot = 0
        for column, value in zip(
            matrix.column_indices[start:stop], matrix.values[start:stop], strict=True
        ):
            integer = _binary64_units(value)
            row_sum += abs(integer)
            dot += integer if target[column] > 0 else -integer
            if column == row:
                diagonal = abs(integer)
        margin = 2 * diagonal - row_sum
        if margin <= 0:
            raise ValueError(
                f"matrix is not strictly row diagonally dominant at row {row}"
            )
        minimum_margin = (
            margin if minimum_margin is None else min(minimum_margin, margin)
        )
        rhs = _binary64_units(system.public.b[row])
        maximum_error = max(maximum_error, abs(rhs - dot))
        maximum_rhs = max(maximum_rhs, abs(rhs))
    margin_lower = _rounded_ratio(minimum_margin, 1 << 1074, upward=False)
    error_upper = _rounded_ratio(maximum_error, 1 << 1074, upward=True)
    # Build the reported bound from the outward-rounded summaries as well, so
    # their exact rational relation can be checked without loading the matrix.
    forward_bound = _rounded_ratio(
        _binary64_units(error_upper), _binary64_units(margin_lower), upward=True
    )
    relative_formation = (
        _rounded_ratio(maximum_error, maximum_rhs, upward=True)
        if maximum_rhs
        else 0.0
        if maximum_error == 0
        else math.inf
    )
    observed = accuracy_metrics(system, target)
    metrics = {}
    for name in QUALIFICATION_LIMITS:
        value = getattr(observed, name)
        if value is None:
            raise ValueError(f"dominance witness produced invalid {name}")
        metrics[name] = float(value)
    evidence = {
        "method": "strict-row-diagonal-dominance-v1",
        "qualified": True,
        "system_sha256": system_digest(system),
        "metrics": metrics,
        "formation": {
            "method": "exact-binary64-residual-v1",
            "relative_linf_residual": relative_formation,
            "verified_forward_bound": forward_bound,
        },
        "certificate": {
            "arithmetic": "exact-binary64-integer-v1",
            "witness": "manufactured-target",
            "metric_evaluation": "binary64",
            "minimum_row_margin": margin_lower,
            "maximum_formation_error": error_upper,
        },
        "uncertainty": "verified-stored-system-forward-bound",
    }
    validate_qualification(evidence, system_digest(system))
    return evidence
