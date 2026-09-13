"""Versioned manufactured inputs and independent offline NS qualification."""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Mapping
from decimal import Decimal, localcontext

import numpy as np
from scipy.sparse.linalg import splu

from .dataset import canonical_json
from .manifests import QUALIFICATION_LIMITS, RHS_SCHEME, validate_qualification
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


def qualify_ns(system: EvaluationSystem) -> dict:
    """Establish empirical feasibility against every gate with a tenfold margin.

    Sparse LU is an offline reference, not the timing reference or a condition
    certificate. Extended-precision residuals audit RHS formation; their size
    alone does not certify forward error for an ill-conditioned system.
    """
    if system.x_star is None or system.reference_kind != "manufactured":
        raise ValueError("NS qualification requires the manufactured target")
    try:
        factor = splu(system.public.matrix.to_scipy().tocsc())
        reference = factor.solve(system.public.b)
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
    # Evaluate row dot products with the platform's extended floating-point
    # type. A platform without extra precision cannot provide this evidence.
    matrix = system.public.matrix
    extended = np.finfo(np.longdouble).eps < np.finfo(np.float64).eps
    rhs = system.public.b.astype(np.longdouble)
    truth = system.x_star.astype(np.longdouble)
    maximum = np.longdouble(0)
    for row in range(matrix.n):
        start, stop = int(matrix.row_offsets[row]), int(matrix.row_offsets[row + 1])
        if extended:
            dot = np.sum(
                matrix.values[start:stop].astype(np.longdouble)
                * truth[matrix.column_indices[start:stop]],
                dtype=np.longdouble,
            )
            difference = abs(rhs[row] - dot)
        else:
            # On platforms where longdouble equals float64, use portable decimal
            # accumulation. This is empirical higher-precision evidence, not an
            # interval bound; that distinction is explicit in the result.
            with localcontext() as context:
                context.prec = 80
                dot = sum(
                    (
                        Decimal.from_float(float(value))
                        * Decimal.from_float(float(system.x_star[column]))
                        for value, column in zip(
                            matrix.values[start:stop],
                            matrix.column_indices[start:stop],
                            strict=True,
                        )
                    ),
                    Decimal(0),
                )
                difference = np.longdouble(
                    str(abs(Decimal.from_float(float(system.public.b[row])) - dot))
                )
        maximum = max(maximum, difference)
    norm_b = np.max(np.abs(rhs), initial=np.longdouble(0))
    relative = (
        float(maximum / norm_b) if norm_b else (0.0 if maximum == 0 else float("inf"))
    )
    evidence = {
        "method": "independent-sparse-lu-v1",
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
