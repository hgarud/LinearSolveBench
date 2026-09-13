from __future__ import annotations

import copy

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import gmres

from linear_solver_bench.manifests import validate_qualification
from linear_solver_bench.models import CsrMatrix, EvaluationSystem, MatrixInput
from linear_solver_bench.workloads import qualify_ns_pyamg, system_digest


@pytest.fixture
def system():
    pytest.importorskip("pyamg")
    side = 35
    one = np.ones(side)
    horizontal = sparse.diags([-1.2 * one[:-1], 4.1 * one, -0.8 * one[:-1]], [-1, 0, 1])
    vertical = sparse.diags([-one[:-1], -one[:-1]], [-1, 1], shape=(side, side))
    matrix = CsrMatrix.from_scipy(
        sparse.kron(sparse.eye(side), horizontal)
        + sparse.kron(vertical, sparse.eye(side))
    )
    target = np.where(np.arange(matrix.n) % 3, 1.0, -1.0)
    return EvaluationSystem(
        "synthetic-nonsymmetric-grid",
        MatrixInput(matrix, matrix.matvec(target), np.full(matrix.n, 7.0), 1e-12),
        target,
    )


def test_fixed_multigrid_reference_is_independent_and_reproducible(system, monkeypatch):
    import linear_solver_bench.reference_pyamg as reference

    original = system_digest(system)
    calls = []

    def observed(*args, **kwargs):
        np.testing.assert_array_equal(kwargs["x0"], np.zeros(system.public.matrix.n))
        calls.append(1)
        return gmres(*args, **kwargs)

    state = np.random.get_state()
    monkeypatch.setattr(reference, "gmres", observed)
    evidence = qualify_ns_pyamg(system)
    assert len(calls) == 3
    assert len(evidence["reference_solver"]["hierarchy"]) > 1
    assert system_digest(system) == original
    after = np.random.get_state()
    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]
    assert qualify_ns_pyamg(system) == evidence
    assert evidence["formation"]["verified_forward_bound"] is None
    validate_qualification(evidence, original)


def test_inaccurate_target_and_nonconvergence_cannot_admit(system, monkeypatch):
    import linear_solver_bench.reference_pyamg as reference

    with pytest.raises(ValueError, match="tenfold margin"):
        qualify_ns_pyamg(
            EvaluationSystem(system.case_id, system.public, -system.x_star)
        )
    monkeypatch.setattr(
        reference,
        "gmres",
        lambda *args, **kwargs: (np.zeros(system.public.matrix.n), 250),
    )
    with pytest.raises(ValueError, match="did not converge"):
        qualify_ns_pyamg(system)


def test_nonfinite_reference_cannot_admit(system, monkeypatch):
    import linear_solver_bench.reference_pyamg as reference

    monkeypatch.setattr(
        reference,
        "gmres",
        lambda *args, **kwargs: (np.full(system.public.matrix.n, np.nan), 0),
    )
    with pytest.raises(ValueError, match="did not converge"):
        qualify_ns_pyamg(system)


@pytest.mark.parametrize(
    "change",
    [
        "nested_type",
        "implementation",
        "empty_hierarchy",
        "dimensions",
        "nnz",
        "bound",
        "iterations",
    ],
)
def test_optional_reference_evidence_is_strict(system, change):
    evidence = copy.deepcopy(qualify_ns_pyamg(system))
    reference = evidence["reference_solver"]
    if change == "nested_type":
        reference["configuration"]["smooth"][1]["degree"] = True
    elif change == "implementation":
        reference["implementation"]["pyamg"] = "9.0.0"
    elif change == "empty_hierarchy":
        reference["hierarchy"] = []
    elif change == "dimensions":
        reference["hierarchy"][1]["n"] = reference["hierarchy"][0]["n"]
    elif change == "nnz":
        reference["hierarchy"][0]["nnz"] = -1
    elif change == "bound":
        evidence["formation"]["verified_forward_bound"] = 1e-8
    elif change == "iterations":
        reference["solves"][0]["inner_iterations"] = 12501
    with pytest.raises(ValueError):
        validate_qualification(evidence)
