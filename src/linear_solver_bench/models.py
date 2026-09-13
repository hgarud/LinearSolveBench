"""Small, strict value types shared by preparation and verification."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import sparse

NATIVE_INDEX_MAX = int(np.iinfo(np.int32).max)
REFERENCE_KINDS = frozenset({"manufactured", "numerical", "none"})


def _readonly_1d(value: object, dtype: object, name: str) -> np.ndarray:
    raw = np.asarray(value)
    expected = np.dtype(dtype)
    if raw.ndim != 1 or raw.dtype != expected:
        raise ValueError(f"{name} must be a one-dimensional {expected.name} array")
    result = np.array(raw, dtype=expected, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, eq=False)
class CsrMatrix:
    n: int
    row_offsets: np.ndarray
    column_indices: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        if type(self.n) is not int or not 0 < self.n <= NATIVE_INDEX_MAX:
            raise ValueError("n must be a positive integer within native int32 bounds")
        rows = _readonly_1d(self.row_offsets, np.uint64, "row_offsets")
        columns = _readonly_1d(self.column_indices, np.uint32, "column_indices")
        values = _readonly_1d(self.values, np.float64, "values")
        if rows.size != self.n + 1 or rows[0] != 0:
            raise ValueError("row_offsets must contain n + 1 entries and start at zero")
        if np.any(rows[1:] < rows[:-1]):
            raise ValueError("row_offsets must be monotone")
        if int(rows[-1]) != values.size or columns.size != values.size:
            raise ValueError("CSR arrays disagree on nnz")
        if values.size > NATIVE_INDEX_MAX:
            raise ValueError("nnz exceeds native int32 bounds")
        if columns.size and int(columns.max()) >= self.n:
            raise ValueError("column index lies outside the matrix")
        if not np.all(np.isfinite(values)):
            raise ValueError("matrix values must be finite")
        for row in range(self.n):
            start, stop = int(rows[row]), int(rows[row + 1])
            if stop - start > 1 and np.any(
                columns[start + 1 : stop] <= columns[start : stop - 1]
            ):
                raise ValueError("column indices must be strictly sorted in each row")
        object.__setattr__(self, "row_offsets", rows)
        object.__setattr__(self, "column_indices", columns)
        object.__setattr__(self, "values", values)

    @property
    def nnz(self) -> int:
        return int(self.values.size)

    @classmethod
    def from_scipy(cls, matrix: sparse.spmatrix) -> CsrMatrix:
        if not sparse.issparse(matrix) or matrix.ndim != 2:
            raise ValueError("matrix must be sparse and two-dimensional")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be square")
        if not 0 < matrix.shape[0] <= NATIVE_INDEX_MAX:
            raise ValueError("matrix dimension exceeds native int32 bounds")
        if matrix.nnz > NATIVE_INDEX_MAX:
            raise ValueError("nnz exceeds native int32 bounds")
        # Validate before converting sparse formats or narrowing native indices.
        source = matrix.copy()
        if hasattr(source, "check_format"):
            source.check_format(full_check=True)
        source = source.tocoo()
        values = source.data
        if values.dtype.kind not in "fiu" or not np.all(np.isfinite(values)):
            raise ValueError("matrix values must be finite real numbers")
        with np.errstate(over="ignore", invalid="ignore"):
            converted = values.astype(np.float64)
        if not np.all(np.isfinite(converted)):
            raise ValueError("matrix values cannot be represented in float64")
        if values.dtype.kind in "iu":
            # Integer-to-float comparisons would coerce the integer and hide loss.
            large = values > 2**53
            if values.dtype.kind == "i":
                large |= values < -(2**53)
            exact = all(int(value) == int(float(value)) for value in values[large])
        else:
            exact = bool(np.all(converted.astype(values.dtype) == values))
        if not exact:
            raise ValueError("matrix conversion to float64 would lose precision")
        # Sum duplicate coordinates only after conversion, avoiding integer wrap.
        csr = sparse.coo_matrix(
            (converted, (source.row, source.col)), shape=source.shape
        ).tocsr()
        csr.sum_duplicates()
        csr.eliminate_zeros()
        csr.sort_indices()
        return cls(
            n=int(csr.shape[0]),
            row_offsets=np.asarray(csr.indptr, dtype=np.uint64),
            column_indices=np.asarray(csr.indices, dtype=np.uint32),
            values=np.asarray(csr.data, dtype=np.float64),
        )

    def to_scipy(self) -> sparse.csr_matrix:
        return sparse.csr_matrix(
            (self.values, self.column_indices, self.row_offsets),
            shape=(self.n, self.n),
            copy=False,
        )

    def matvec(self, vector: object) -> np.ndarray:
        value = np.asarray(vector, dtype=np.float64)
        if value.shape != (self.n,):
            raise ValueError(f"vector must have shape ({self.n},)")
        return np.asarray(self.to_scipy() @ value, dtype=np.float64)


@dataclass(frozen=True, eq=False)
class MatrixInput:
    matrix: CsrMatrix
    b: np.ndarray
    x0: np.ndarray
    tolerance: float

    def __post_init__(self) -> None:
        if not isinstance(self.matrix, CsrMatrix):
            raise TypeError("matrix must be CsrMatrix")
        b = _readonly_1d(self.b, np.float64, "b")
        x0 = _readonly_1d(self.x0, np.float64, "x0")
        if b.shape != (self.matrix.n,) or x0.shape != (self.matrix.n,):
            raise ValueError("b and x0 must match the matrix dimension")
        if not np.all(np.isfinite(b)) or not np.all(np.isfinite(x0)):
            raise ValueError("input vectors must be finite")
        tolerance = float(self.tolerance)
        if not math.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
            raise ValueError("tolerance must lie strictly between zero and one")
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "x0", x0)
        object.__setattr__(self, "tolerance", tolerance)


@dataclass(frozen=True, eq=False)
class EvaluationSystem:
    case_id: str
    public: MatrixInput
    x_star: np.ndarray | None = None
    reference_kind: str = "manufactured"

    def __post_init__(self) -> None:
        if not self.case_id or not self.case_id.isascii():
            raise ValueError("case_id must be nonempty ASCII")
        if not isinstance(self.public, MatrixInput):
            raise TypeError("public must be MatrixInput")
        if self.reference_kind not in REFERENCE_KINDS:
            raise ValueError("unknown reference_kind")
        if self.reference_kind == "none":
            if self.x_star is not None:
                raise ValueError("reference_kind 'none' must not contain x_star")
            return
        if self.x_star is None:
            raise ValueError(f"{self.reference_kind} reference requires x_star")
        truth = _readonly_1d(self.x_star, np.float64, "x_star")
        if truth.shape != (self.public.matrix.n,) or not np.all(np.isfinite(truth)):
            raise ValueError("x_star must be finite and match the matrix")
        object.__setattr__(self, "x_star", truth)
