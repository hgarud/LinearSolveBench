from __future__ import annotations

import copy
import json
from fractions import Fraction

import numpy as np
import pytest
from scipy import sparse

from linear_solver_bench.manifests import QUALIFICATION_LIMITS, validate_qualification
from linear_solver_bench.models import CsrMatrix, EvaluationSystem, MatrixInput
from linear_solver_bench.workloads import qualify_ns_dominance, system_digest


def _system(values, target, b=None):
    matrix = CsrMatrix.from_scipy(
        sparse.csr_matrix(np.asarray(values, dtype=np.float64))
    )
    target = np.asarray(target, dtype=np.float64)
    return EvaluationSystem(
        "dominance-fixture",
        MatrixInput(
            matrix,
            matrix.matvec(target) if b is None else np.asarray(b, dtype=np.float64),
            np.zeros(matrix.n),
            1e-12,
        ),
        target,
    )


def test_dominance_bound_covers_exact_stored_solution(monkeypatch):
    import linear_solver_bench.workloads as workloads

    def no_factorization(*args, **kwargs):
        pytest.fail("the certificate must not invoke LU")

    monkeypatch.setattr(workloads, "splu", no_factorization)
    system = _system([[4.1, 0.3], [0.7, 3.2]], [1, -1])
    evidence = qualify_ns_dominance(system)
    a, b, c, d = map(Fraction.from_float, system.public.matrix.values)
    r, s = map(Fraction.from_float, system.public.b)
    determinant = a * d - b * c
    exact = ((d * r - b * s) / determinant, (a * s - c * r) / determinant)
    errors = [abs(exact[0] - 1), abs(exact[1] + 1)]
    bound = Fraction.from_float(evidence["formation"]["verified_forward_bound"])
    assert max(errors) <= bound
    assert sum(value * value for value in errors) / 2 <= bound * bound
    assert bound > 0
    assert evidence["certificate"]["witness"] == "manufactured-target"
    assert evidence["certificate"]["metric_evaluation"] == "binary64"
    assert evidence["uncertainty"] == "verified-stored-system-forward-bound"
    for name, limit in QUALIFICATION_LIMITS.items():
        assert evidence["metrics"][name] <= limit
    validate_qualification(json.loads(json.dumps(evidence)), system_digest(system))


@pytest.mark.parametrize("matrix", [[[1, 2], [3, 1]], [[1, 1], [1, 1]]])
def test_non_strict_row_dominance_is_rejected(matrix):
    with pytest.raises(ValueError, match="not strictly row"):
        qualify_ns_dominance(_system(matrix, [1, -1]))


def test_exact_margin_does_not_hide_small_off_diagonal_entries():
    # Each tiny entry is lost when naively added to 1.0, but the exact row is
    # weakly dominant, not strictly dominant.
    diagonal = np.nextafter(1.0, np.inf)
    tiny = 2.0**-53
    system = _system(
        [[diagonal, 1.0, tiny, tiny], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]],
        [1, -1, 1, -1],
    )
    with pytest.raises(ValueError, match="not strictly row"):
        qualify_ns_dominance(system)


def test_small_positive_exact_margin_is_supported():
    value = np.nextafter(1.0, np.inf)
    evidence = qualify_ns_dominance(_system([[value, 1], [0, 1]], [1, -1]))
    assert evidence["certificate"]["minimum_row_margin"] == value - 1
    assert evidence["formation"]["verified_forward_bound"] == 0


def test_zero_measured_metrics_cannot_hide_large_formation_uncertainty():
    # b=fl(A*[1,1]) has zero recomputed residual, but rounding RHS formation
    # relative to a tiny dominance margin cannot support the forward limit.
    value = np.nextafter(1.0, np.inf)
    system = _system([[value, 1], [1, value]], [1, 1])
    with pytest.raises(ValueError, match="forward bound lacks the tenfold"):
        qualify_ns_dominance(system)


def test_subnormal_coefficients_are_accumulated_exactly():
    tiny = np.nextafter(0.0, 1.0)
    evidence = qualify_ns_dominance(_system([[2 * tiny]], [1]))
    assert evidence["certificate"]["minimum_row_margin"] == 2 * tiny
    assert evidence["certificate"]["maximum_formation_error"] == 0
    assert evidence["formation"]["verified_forward_bound"] == 0


def test_dominance_requires_rademacher_and_all_witness_gates():
    with pytest.raises(ValueError, match="Rademacher"):
        qualify_ns_dominance(_system([[2]], [0.5]))
    with pytest.raises(ValueError, match="tenfold"):
        qualify_ns_dominance(_system([[2]], [1], b=[2.01]))


def test_certificate_rejects_inconsistent_bound_and_identity():
    system = _system([[4.1, 0.3], [0.7, 3.2]], [1, -1])
    evidence = qualify_ns_dominance(system)
    bad = copy.deepcopy(evidence)
    bad["formation"]["verified_forward_bound"] = 0.0
    with pytest.raises(ValueError, match="understates"):
        validate_qualification(bad)
    bad = copy.deepcopy(evidence)
    bad["certificate"]["minimum_row_margin"] = -1.0
    with pytest.raises(ValueError):
        validate_qualification(bad)
    with pytest.raises(ValueError, match="different numerical inputs"):
        validate_qualification(evidence, "0" * 64)
