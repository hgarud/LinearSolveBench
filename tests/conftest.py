from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from linear_solver_bench.accuracy import AccuracyThresholds
from linear_solver_bench.models import CsrMatrix, EvaluationSystem, MatrixInput
from linear_solver_bench.protocol import DriverOutput


@pytest.fixture
def small_system() -> EvaluationSystem:
    matrix = CsrMatrix.from_scipy(
        sparse.csr_matrix(np.asarray([[4.0, 1.0], [2.0, 3.0]]))
    )
    truth = np.asarray([1.0, -1.0], dtype=np.float64)
    return EvaluationSystem(
        case_id="test-small",
        public=MatrixInput(
            matrix=matrix,
            b=matrix.matvec(truth),
            x0=np.zeros(2, dtype=np.float64),
            tolerance=1.0e-4,
        ),
        x_star=truth,
    )


@pytest.fixture
def driver_output(small_system):
    return (
        DriverOutput(
            status=0,
            solution=small_system.x_star,
            elapsed_s=0.01,
            create_s=0.001,
            setup_s=0.003,
            solve_s=0.006,
            input_mutated=False,
        ),
        AccuracyThresholds.from_tolerance(1.0e-4),
    )
