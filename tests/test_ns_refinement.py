from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import splu

from linear_solver_bench.manifests import QUALIFICATION_LIMITS, validate_qualification
from linear_solver_bench.models import CsrMatrix, EvaluationSystem, MatrixInput
from linear_solver_bench.verify import accuracy_metrics
from linear_solver_bench.workloads import qualify_ns, system_digest


@pytest.mark.parametrize("decimal_fallback", [False, True])
def test_one_refinement_improves_an_ill_scaled_exact_system(
    monkeypatch, decimal_fallback
):
    import linear_solver_bench.workloads as workloads

    # Integer Vandermonde entries and +/-1 targets make RHS formation exact.
    # Ill-conditioning makes the initial float64 solve measurably inaccurate.
    n = 10
    matrix = CsrMatrix.from_scipy(
        sparse.csr_matrix(
            np.vander(np.arange(1, n + 1, dtype=float), N=n, increasing=True)
        )
    )
    target = np.where(np.arange(n) % 2, -1.0, 1.0)
    system = EvaluationSystem(
        "integer-vandermonde",
        MatrixInput(matrix, matrix.matvec(target), np.zeros(n), 1e-12),
        target,
    )
    initial = accuracy_metrics(
        system, splu(matrix.to_scipy().tocsc()).solve(system.public.b)
    )
    assert initial.relative_linf_forward_error > 1e-9
    original_digest = system_digest(system)
    solve_calls = []

    class CountedFactor:
        def __init__(self, value):
            self.factor = splu(value)

        def solve(self, rhs):
            solve_calls.append(rhs.copy())
            return self.factor.solve(rhs)

    monkeypatch.setattr(workloads, "splu", CountedFactor)
    if decimal_fallback:
        double_info = np.finfo(np.float64)
        monkeypatch.setattr(workloads.np, "finfo", lambda dtype: double_info)
    evidence = qualify_ns(system, refine=True)
    assert len(solve_calls) == 2
    assert evidence["method"] == "independent-sparse-lu-refined-v1"
    assert evidence["metrics"]["relative_linf_forward_error"] < (
        initial.relative_linf_forward_error * 0.01
    )
    for name, limit in QUALIFICATION_LIMITS.items():
        assert evidence["metrics"][name] <= limit
    assert evidence["formation"]["relative_linf_residual"] == 0
    assert evidence["formation"]["verified_forward_bound"] is None
    assert evidence["uncertainty"] == "empirical-feasibility-only"
    assert system_digest(system) == original_digest
    validate_qualification(evidence, original_digest)


def test_plain_lu_keeps_existing_method_identity(small_system):
    evidence = qualify_ns(small_system)
    assert evidence["method"] == "independent-sparse-lu-v1"
    assert evidence["formation"]["verified_forward_bound"] is None
    with pytest.raises(ValueError, match="boolean"):
        qualify_ns(small_system, refine="yes")
