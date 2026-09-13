from __future__ import annotations

import numpy as np

from linear_solver_bench.accuracy import AccuracyThresholds
from linear_solver_bench.verify import verify_solution


def test_exact_solution_passes_all_gates(small_system) -> None:
    result = verify_solution(small_system, small_system.x_star)
    assert result.ok
    assert result.failure is None
    assert result.metrics is not None
    assert result.metrics.normwise_backward_error == 0.0
    assert result.metrics.forward_error == 0.0


def test_bad_solution_and_driver_failures_are_closed(small_system) -> None:
    bad = verify_solution(small_system, np.zeros(2, dtype=np.float64))
    assert not bad.ok
    assert "forward_error" in bad.failure
    assert verify_solution(small_system, small_system.x_star, status=1).failure == (
        "nonzero_status"
    )
    assert (
        verify_solution(small_system, small_system.x_star, input_mutated=True).failure
        == "input_mutation"
    )
    assert verify_solution(small_system, np.asarray([np.nan, 0.0])).failure == (
        "nonfinite_solution"
    )


def test_accuracy_threshold_contract() -> None:
    thresholds = AccuracyThresholds.from_tolerance(1.0e-4)
    assert thresholds.normwise_backward_error == 1.0e-4
    assert thresholds.componentwise_backward_error == 5.0e-4
    assert thresholds.forward_error == 5.0e-4
    ill_conditioned = AccuracyThresholds.from_tolerance(4.0e-2)
    assert ill_conditioned.forward_error == 2.0e-1


def _system(matrix, truth, *, tolerance=1e-12, reference_kind="manufactured", b=None):
    from scipy import sparse

    from linear_solver_bench.models import CsrMatrix, EvaluationSystem, MatrixInput

    matrix = CsrMatrix.from_scipy(
        sparse.csr_matrix(np.asarray(matrix, dtype=np.float64))
    )
    truth = None if truth is None else np.asarray(truth, dtype=np.float64)
    return EvaluationSystem(
        case_id="numerical-test",
        public=MatrixInput(
            matrix=matrix,
            b=matrix.matvec(truth) if b is None else np.asarray(b, dtype=np.float64),
            x0=np.zeros(matrix.n, dtype=np.float64),
            tolerance=tolerance,
        ),
        x_star=truth,
        reference_kind=reference_kind,
    )


def test_ns_mesh_uses_fixed_gates_and_diagnostic_residual() -> None:
    system = _system([[1e10, -1e10], [0, 1]], [1, 1])
    result = verify_solution(
        system, np.asarray([1 + 1e-12, 1.0]), contract_id="ns-mesh-accuracy-v1"
    )
    assert result.ok
    assert result.metrics.relative_residual > 1e-8
    assert result.thresholds.normwise_backward_error == 1e-10
    assert result.thresholds.componentwise_backward_error == 1e-8
    assert result.thresholds.forward_error == 1e-5


def test_ns_mesh_requires_manufactured_truth(small_system) -> None:
    from dataclasses import replace

    for system in (
        replace(small_system, reference_kind="none", x_star=None),
        replace(small_system, reference_kind="numerical"),
    ):
        result = verify_solution(
            system, small_system.x_star, contract_id="ns-mesh-accuracy-v1"
        )
        assert result.failure == "missing_manufactured_reference"


def test_flash_weak_row_and_numerical_forward_errors_are_diagnostic() -> None:
    system = _system(
        [[1, 0], [0, 1e-20]],
        [100, 100],
        b=[1, 1e-20],
        reference_kind="numerical",
    )
    result = verify_solution(
        system, np.asarray([1.0, 0.0]), contract_id="flash-replay-accuracy-v1"
    )
    assert result.ok
    assert result.metrics.componentwise_backward_error == 1.0
    assert result.metrics.forward_error == 1.0
    assert result.thresholds.componentwise_backward_error is None
    assert result.thresholds.forward_error is None


def test_flash_missing_reference_serializes_null_with_reason() -> None:
    import json

    system = _system([[1]], None, b=[1], reference_kind="none")
    result = verify_solution(
        system, np.asarray([1.0]), contract_id="flash-replay-accuracy-v1"
    )
    assert result.ok
    encoded = result.to_dict()
    assert encoded["metrics"]["relative_l2_forward_error"] is None
    assert encoded["metrics"]["diagnostic_reasons"]["relative_l2_forward_error"] == (
        "reference_unavailable"
    )
    json.dumps(encoded, allow_nan=False)


def test_flash_zero_rhs_residual_definition() -> None:
    import json

    system = _system([[1]], None, b=[0], reference_kind="none")
    exact = verify_solution(
        system, np.asarray([0.0]), contract_id="flash-replay-accuracy-v1"
    )
    bad = verify_solution(
        system, np.asarray([1e-300]), contract_id="flash-replay-accuracy-v1"
    )
    assert exact.ok
    assert exact.metrics.relative_residual == 0
    assert not bad.ok
    assert bad.metrics.relative_residual is None
    assert bad.metrics.diagnostic_reasons["relative_residual"] == (
        "zero_rhs_with_nonzero_residual"
    )
    json.dumps(bad.to_dict(), allow_nan=False)


def test_overflowing_denominator_cannot_appear_accurate() -> None:
    import json

    system = _system([[1e308]], [1.0])
    result = verify_solution(
        system, np.asarray([1.0]), contract_id="ns-mesh-accuracy-v1"
    )
    assert not result.ok
    assert result.failure == "nonfinite_metric"
    assert result.metrics.normwise_backward_error is None
    json.dumps(result.to_dict(), allow_nan=False)


def test_l2_ratio_is_stable_for_large_and_tiny_vectors() -> None:
    import pytest

    from linear_solver_bench.verify import accuracy_metrics

    for scale in (1e-300, 1.3e308):
        system = _system([[1e-300, 0], [0, 1e-300]], [scale, scale])
        metrics = accuracy_metrics(system, system.x_star * (1 + 1e-4))
        assert metrics.relative_l2_forward_error == pytest.approx(1e-4, rel=1e-10)
        assert metrics.relative_linf_forward_error == pytest.approx(1e-4, rel=1e-10)


def test_unknown_accuracy_contract_is_rejected(small_system) -> None:
    import pytest

    with pytest.raises(ValueError, match="unknown accuracy contract"):
        verify_solution(
            small_system, small_system.x_star, contract_id="future-contract"
        )
