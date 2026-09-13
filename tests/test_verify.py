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
