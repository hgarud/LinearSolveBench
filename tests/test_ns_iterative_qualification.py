from __future__ import annotations

import copy

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import gmres

from linear_solver_bench.manifests import (
    ITERATIVE_QUALIFICATION_CONFIG,
    ITERATIVE_QUALIFICATION_METHOD,
    QUALIFICATION_LIMITS,
    validate_qualification,
)
from linear_solver_bench.models import CsrMatrix, EvaluationSystem, MatrixInput
from linear_solver_bench.workloads import qualify_ns_iterative, system_digest


def _system(*, empty_row=False):
    operator = sparse.diags([-np.ones(7), np.full(8, 4.0), np.full(7, 0.5)], [-1, 0, 1])
    operator = sparse.diags(np.logspace(-4, 4, 8)) @ operator
    operator = operator @ sparse.diags(np.logspace(3, -3, 8))
    if empty_row:
        operator = operator.tolil()
        operator[3, :] = 0
    matrix = CsrMatrix.from_scipy(operator)
    target = np.where(np.arange(8) % 2, -1.0, 1.0)
    return EvaluationSystem(
        "iterative-qualification-fixture",
        MatrixInput(matrix, matrix.matvec(target), np.full(8, 7.0), 1e-12),
        target,
    )


def test_fixed_iterative_reference_preserves_inputs_and_starts_every_solve_from_zero(
    monkeypatch,
):
    import linear_solver_bench.workloads as workloads

    system = _system()
    original = system_digest(system)
    calls = []

    def counted_gmres(matrix, rhs, **kwargs):
        assert np.array_equal(kwargs["x0"], np.zeros(system.public.matrix.n))
        assert kwargs["callback_type"] == "pr_norm"
        calls.append(rhs.copy())
        return gmres(matrix, rhs, **kwargs)

    monkeypatch.setattr(workloads, "gmres", counted_gmres)
    evidence = qualify_ns_iterative(system)
    assert len(calls) == 3
    assert evidence["method"] == ITERATIVE_QUALIFICATION_METHOD
    assert (
        evidence["reference_solver"]["configuration"] == ITERATIVE_QUALIFICATION_CONFIG
    )
    assert all(value["info"] == 0 for value in evidence["reference_solver"]["solves"])
    assert evidence["formation"]["verified_forward_bound"] is None
    assert evidence["uncertainty"] == "empirical-feasibility-only"
    for metric, limit in QUALIFICATION_LIMITS.items():
        assert evidence["metrics"][metric] <= limit
    assert system_digest(system) == original
    validate_qualification(evidence, original)


@pytest.mark.parametrize("result", ["nonconverged", "nonfinite"])
def test_failed_reference_never_produces_admission(monkeypatch, result):
    import linear_solver_bench.workloads as workloads

    def failing_gmres(matrix, rhs, **kwargs):
        if result == "nonconverged":
            return np.zeros(len(rhs)), 250
        return np.full(len(rhs), np.nan), 0

    monkeypatch.setattr(workloads, "gmres", failing_gmres)
    with pytest.raises(ValueError, match="could not be computed"):
        qualify_ns_iterative(_system())


def test_iterative_reference_rejects_empty_row_and_inaccurate_target():
    with pytest.raises(ValueError, match="empty row"):
        qualify_ns_iterative(_system(empty_row=True))
    system = _system()
    invalid = EvaluationSystem(system.case_id, system.public, -system.x_star)
    with pytest.raises(ValueError, match="tenfold margin"):
        qualify_ns_iterative(invalid)


@pytest.mark.parametrize(
    "change",
    ["budget", "boolean_budget", "missing_correction", "failure", "bound", "library"],
)
def test_iterative_evidence_rejects_configuration_or_semantic_drift(change):
    evidence = copy.deepcopy(qualify_ns_iterative(_system()))
    reference = evidence["reference_solver"]
    if change == "budget":
        reference["configuration"]["restart"] = 100
    elif change == "boolean_budget":
        reference["configuration"]["diag_pivot_thresh"] = True
    elif change == "missing_correction":
        reference["solves"].pop()
    elif change == "failure":
        reference["solves"][1]["info"] = 250
    elif change == "bound":
        evidence["formation"]["verified_forward_bound"] = 1e-8
    elif change == "library":
        reference["implementation"]["scipy"] = "/some/local/path"
    with pytest.raises(ValueError):
        validate_qualification(evidence)
