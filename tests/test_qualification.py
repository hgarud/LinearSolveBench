from __future__ import annotations

import numpy as np
from scipy import sparse

from linear_solver_bench.qualification import (
    CONDITION_METHOD,
    ESTIMATOR_GUARD_FACTOR,
    FLOAT64_EPSILON,
    condition_from_factor,
    qualification_document,
)


def test_uniform_one_norm_condition_from_factor() -> None:
    diagonal = np.arange(1.0, 7.0)
    matrix = sparse.diags(diagonal, format="csc")
    dense = matrix.toarray()
    condition, evidence = condition_from_factor(
        matrix,
        case_id="ss-test",
        rank_estimate=6,
        rank_tolerance=1.0e-12,
        solve=lambda value: np.linalg.solve(dense, value),
        solve_transpose=lambda value: np.linalg.solve(dense.T, value),
    )
    assert evidence is not None
    assert condition["status"] == "estimated"
    assert condition["method"] == CONDITION_METHOD
    assert condition["condition_norm"] == "1"
    assert np.isclose(condition["condition_estimate_raw"], 6.0)
    assert np.isclose(condition["condition_estimate"], 6.0 * ESTIMATOR_GUARD_FACTOR)
    assert np.isclose(
        condition["minimum_safe_tolerance"],
        20.0 * 6.0 * ESTIMATOR_GUARD_FACTOR * FLOAT64_EPSILON,
    )


def test_rank_deficient_factor_has_no_condition_claim() -> None:
    matrix = sparse.eye(6, format="csc")
    condition, evidence = condition_from_factor(
        matrix,
        case_id="ss-test",
        rank_estimate=5,
        rank_tolerance=1.0e-12,
        solve=lambda value: value,
        solve_transpose=lambda value: value,
    )
    assert evidence is None
    assert condition["status"] == "rank_deficient"
    assert condition["condition_norm"] is None
    assert condition["condition_estimate"] is None


def test_qualification_document_binds_uniform_policy() -> None:
    condition = {
        "status": "rank_deficient",
        "method": CONDITION_METHOD,
    }
    document = qualification_document(
        catalogue_sha256="a" * 64,
        input_manifest_sha256="b" * 64,
        expected_case_count=1,
        results=[{"index": 0, "condition": condition}],
        toolchain={"python": "test"},
    )
    assert document["complete"] is True
    assert document["policy"]["condition_norm"] == "1"
    assert document["method_counts"] == {CONDITION_METHOD: 1}
    assert len(document["condition_results_sha256"]) == 64
