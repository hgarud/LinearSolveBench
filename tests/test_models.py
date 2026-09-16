import numpy as np
import pytest
from scipy import sparse

from linear_solver_bench.models import BenchmarkCase, CaseInput, CsrMatrix, RawExecution


def test_csr_matrix_is_canonical_and_immutable():
    source = sparse.coo_matrix(
        ([1.0, 2.0, 3.0, 1.0], ([0, 0, 1, 0], [1, 0, 1, 1])),
        shape=(2, 2),
    )
    matrix = CsrMatrix.from_scipy(source)

    assert matrix.nnz == 3
    np.testing.assert_allclose(matrix.matvec([2.0, 4.0]), [12.0, 12.0])
    assert not matrix.values.flags.writeable
    with pytest.raises(ValueError):
        matrix.values[0] = 0


def test_models_reject_malformed_values():
    with pytest.raises(ValueError, match="row_offsets"):
        CsrMatrix(
            2,
            np.array([1, 1, 1], dtype=np.uint64),
            np.array([], dtype=np.uint32),
            np.array([], dtype=np.float64),
        )

    matrix = CsrMatrix.from_scipy(sparse.eye(2, format="csr"))
    with pytest.raises(ValueError, match="match the matrix"):
        CaseInput(matrix, np.ones(1), np.zeros(2), 1e-8)


def test_target_is_separate_from_candidate_input():
    matrix = CsrMatrix.from_scipy(sparse.eye(2, format="csr"))
    value = CaseInput(matrix, np.ones(2), np.zeros(2), 1e-8)
    case = BenchmarkCase("case-1", value, np.ones(2))

    assert not hasattr(case.input, "target")
    assert not case.target.flags.writeable


def test_raw_execution_bounds_diagnostics():
    raw = RawExecution(0, 0.5, "x" * 9000, b"result")
    assert len(raw.diagnostics) == 8000
    with pytest.raises(ValueError, match="nonnegative"):
        RawExecution(0, -1, "", None)
