"""The small set of values shared by the benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .protocol import DIAGNOSTIC_LIMIT


def _array(value: object, dtype: object, name: str) -> np.ndarray:
    """Copy a one-dimensional array and make it immutable."""
    raw = np.asarray(value)
    expected = np.dtype(dtype)
    if raw.ndim != 1 or raw.dtype != expected:
        raise ValueError(f"{name} must be a one-dimensional {expected.name} array")
    result = np.array(raw, dtype=expected, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, eq=False)
class CsrMatrix:
    """A canonical square float64 CSR matrix accepted by the native driver."""

    n: int
    row_offsets: np.ndarray
    column_indices: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        if type(self.n) is not int or not 0 < self.n <= np.iinfo(np.int32).max:
            raise ValueError("n must be a positive native int32")
        rows = _array(self.row_offsets, np.uint64, "row_offsets")
        columns = _array(self.column_indices, np.uint32, "column_indices")
        values = _array(self.values, np.float64, "values")
        if rows.size != self.n + 1 or rows[0] != 0:
            raise ValueError("row_offsets must have n + 1 entries and start at zero")
        if np.any(rows[1:] < rows[:-1]):
            raise ValueError("row_offsets must be monotone")
        if int(rows[-1]) != values.size or columns.size != values.size:
            raise ValueError("CSR arrays disagree on nnz")
        if values.size > np.iinfo(np.int32).max:
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
                raise ValueError("columns must be strictly sorted within each row")
        object.__setattr__(self, "row_offsets", rows)
        object.__setattr__(self, "column_indices", columns)
        object.__setattr__(self, "values", values)

    @property
    def nnz(self) -> int:
        return int(self.values.size)

    @classmethod
    def from_scipy(cls, matrix: sparse.spmatrix) -> CsrMatrix:
        """Canonicalize a real sparse matrix without silently losing precision."""
        if not sparse.issparse(matrix) or matrix.ndim != 2:
            raise ValueError("matrix must be sparse and two-dimensional")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be square")
        source = matrix.copy()
        if hasattr(source, "check_format"):
            source.check_format(full_check=True)
        source = source.tocoo()
        if source.data.dtype.kind not in "fiu" or not np.all(np.isfinite(source.data)):
            raise ValueError("matrix values must be finite real numbers")
        converted = source.data.astype(np.float64)
        if not np.all(np.isfinite(converted)):
            raise ValueError("matrix values cannot be represented as float64")
        if source.data.dtype.kind in "iu":
            exact = all(int(value) == int(float(value)) for value in source.data)
        else:
            exact = bool(np.all(converted.astype(source.data.dtype) == source.data))
        if not exact:
            raise ValueError("matrix conversion to float64 would lose precision")
        csr = sparse.coo_matrix(
            (converted, (source.row, source.col)), shape=source.shape
        ).tocsr()
        csr.sum_duplicates()
        csr.eliminate_zeros()
        csr.sort_indices()
        return cls(
            int(csr.shape[0]),
            np.asarray(csr.indptr, dtype=np.uint64),
            np.asarray(csr.indices, dtype=np.uint32),
            np.asarray(csr.data, dtype=np.float64),
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
class CaseInput:
    """Everything made visible to a candidate solver."""

    matrix: CsrMatrix
    b: np.ndarray
    x0: np.ndarray
    tolerance: float

    def __post_init__(self) -> None:
        if not isinstance(self.matrix, CsrMatrix):
            raise TypeError("matrix must be CsrMatrix")
        b = _array(self.b, np.float64, "b")
        x0 = _array(self.x0, np.float64, "x0")
        if b.shape != (self.matrix.n,) or x0.shape != (self.matrix.n,):
            raise ValueError("b and x0 must match the matrix")
        if not np.all(np.isfinite(b)) or not np.all(np.isfinite(x0)):
            raise ValueError("input vectors must be finite")
        tolerance = float(self.tolerance)
        if not 0 < tolerance < 1:
            raise ValueError("tolerance must be between zero and one")
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "x0", x0)
        object.__setattr__(self, "tolerance", tolerance)


@dataclass(frozen=True, eq=False)
class BenchmarkCase:
    """A candidate input plus trusted verification data kept outside that input."""

    case_id: str
    input: CaseInput
    target: np.ndarray | None = None
    input_sha256: str = ""
    role: str = "scored"

    def __post_init__(self) -> None:
        if not self.case_id or not self.case_id.isascii():
            raise ValueError("case_id must be nonempty ASCII")
        if not isinstance(self.input, CaseInput):
            raise TypeError("input must be CaseInput")
        if self.target is not None:
            target = _array(self.target, np.float64, "target")
            if target.shape != (self.input.matrix.n,) or not np.all(
                np.isfinite(target)
            ):
                raise ValueError("target must be finite and match the matrix")
            object.__setattr__(self, "target", target)
        if self.input_sha256 and (
            len(self.input_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.input_sha256)
        ):
            raise ValueError("input_sha256 must be a lowercase SHA-256 digest")
        if self.role not in {"scored", "control"}:
            raise ValueError("role must be scored or control")


@dataclass(frozen=True)
class RawExecution:
    """Untrusted output returned by a local process or Modal sandbox."""

    returncode: int
    wall_seconds: float
    diagnostics: str
    output: bytes | None

    def __post_init__(self) -> None:
        if type(self.returncode) is not int:
            raise TypeError("returncode must be an integer")
        if not math.isfinite(self.wall_seconds) or self.wall_seconds < 0:
            raise ValueError("wall_seconds must be finite and nonnegative")
        if not isinstance(self.diagnostics, str):
            raise TypeError("diagnostics must be text")
        object.__setattr__(self, "diagnostics", self.diagnostics[-DIAGNOSTIC_LIMIT:])
