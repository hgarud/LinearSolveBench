"""Stable trusted archive codec; archives never enter candidate workspaces."""

from __future__ import annotations

import io

import numpy as np

from .models import CsrMatrix, EvaluationSystem, MatrixInput

ARCHIVE_KEYS = frozenset(
    {
        "b",
        "case_id",
        "column_indices",
        "row_offsets",
        "tolerance",
        "values",
        "x0",
        "x_star",
    }
)


def encode_system(system: EvaluationSystem) -> bytes:
    buffer = io.BytesIO()
    public = system.public
    np.savez_compressed(
        buffer,
        case_id=np.asarray(system.case_id),
        row_offsets=public.matrix.row_offsets,
        column_indices=public.matrix.column_indices,
        values=public.matrix.values,
        b=public.b,
        x0=public.x0,
        tolerance=np.asarray(public.tolerance, dtype=np.float64),
        x_star=system.x_star,
    )
    return buffer.getvalue()


def decode_system(payload: bytes) -> EvaluationSystem:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != ARCHIVE_KEYS:
                raise ValueError("system archive schema mismatch")
            case = np.asarray(archive["case_id"])
            tolerance = np.asarray(archive["tolerance"])
            if case.shape != () or case.dtype.kind not in "SU":
                raise ValueError("archive case_id must be a string scalar")
            if tolerance.shape != () or tolerance.dtype != np.float64:
                raise ValueError("archive tolerance must be a float64 scalar")
            rows = np.asarray(archive["row_offsets"])
            matrix = CsrMatrix(
                n=int(rows.size - 1),
                row_offsets=rows,
                column_indices=np.asarray(archive["column_indices"]),
                values=np.asarray(archive["values"]),
            )
            return EvaluationSystem(
                case_id=str(case),
                public=MatrixInput(
                    matrix=matrix,
                    b=np.asarray(archive["b"]),
                    x0=np.asarray(archive["x0"]),
                    tolerance=float(tolerance),
                ),
                x_star=np.asarray(archive["x_star"]),
            )
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("cannot read system archive") from exc
