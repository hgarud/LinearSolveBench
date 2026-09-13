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
ARCHIVE_SCHEMA_VERSION = 2
V2_REQUIRED_KEYS = (ARCHIVE_KEYS - {"x_star"}) | {
    "schema_version",
    "reference_kind",
}


def encode_system(
    system: EvaluationSystem, *, schema_version: int | None = None
) -> bytes:
    """Encode a trusted system; existing manufactured callers retain v1 bytes.

    Pilot preparation must explicitly request schema_version=2, including for
    manufactured targets. Systems with optional numerical references use v2 by
    default because v1 cannot express their reference semantics.
    """
    if schema_version is None:
        schema_version = 1 if system.reference_kind == "manufactured" else 2
    if type(schema_version) is not int or schema_version not in (1, 2):
        raise ValueError("unsupported system archive schema version")
    if schema_version == 1 and system.reference_kind != "manufactured":
        raise ValueError("legacy archives require a manufactured reference")
    buffer = io.BytesIO()
    public = system.public
    fields = dict(
        case_id=np.asarray(system.case_id),
        row_offsets=public.matrix.row_offsets,
        column_indices=public.matrix.column_indices,
        values=public.matrix.values,
        b=public.b,
        x0=public.x0,
        tolerance=np.asarray(public.tolerance, dtype=np.float64),
    )
    if schema_version == 2:
        fields["schema_version"] = np.asarray(2, dtype=np.uint32)
        fields["reference_kind"] = np.asarray(system.reference_kind)
    if system.x_star is not None:
        fields["x_star"] = system.x_star
    np.savez_compressed(buffer, **fields)
    return buffer.getvalue()


def _string_scalar(archive: object, key: str) -> str:
    value = np.asarray(archive[key])
    if value.shape != () or value.dtype.kind not in "SU":
        raise ValueError(f"archive {key} must be a string scalar")
    scalar = value.item()
    if isinstance(scalar, bytes):
        try:
            return scalar.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError(f"archive {key} must be ASCII") from exc
    return scalar


def decode_system(payload: bytes) -> EvaluationSystem:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            keys = set(archive.files)
            if len(keys) != len(archive.files):
                raise ValueError("system archive contains duplicate fields")
            reference_kind = "manufactured"
            if "schema_version" in keys:
                version = np.asarray(archive["schema_version"])
                if (
                    version.shape != ()
                    or version.dtype != np.uint32
                    or int(version) != ARCHIVE_SCHEMA_VERSION
                ):
                    raise ValueError("unsupported system archive schema version")
                if not V2_REQUIRED_KEYS <= keys:
                    raise ValueError("system archive schema mismatch")
                reference_kind = _string_scalar(archive, "reference_kind")
                expected = V2_REQUIRED_KEYS | (
                    {"x_star"} if reference_kind != "none" else set()
                )
            else:
                expected = ARCHIVE_KEYS
            if keys != expected:
                raise ValueError("system archive schema mismatch")
            case = _string_scalar(archive, "case_id")
            tolerance = np.asarray(archive["tolerance"])
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
                case_id=case,
                public=MatrixInput(
                    matrix=matrix,
                    b=np.asarray(archive["b"]),
                    x0=np.asarray(archive["x0"]),
                    tolerance=float(tolerance),
                ),
                x_star=(
                    np.asarray(archive["x_star"]) if reference_kind != "none" else None
                ),
                reference_kind=reference_kind,
            )
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("cannot read system archive") from exc
