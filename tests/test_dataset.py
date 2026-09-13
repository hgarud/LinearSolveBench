from __future__ import annotations

import numpy as np
from scipy import sparse

from linear_solver_bench.dataset import (
    dataset_summary,
    load_split,
    materialize_system,
    matrix_sha256,
    rademacher_x_star,
)
from linear_solver_bench.models import CsrMatrix


def test_frozen_dataset_counts_and_identity() -> None:
    assert dataset_summary() == {
        "screened": 350,
        "full_rank": 213,
        "rank_deficient": 137,
        "condition_unsupported": 10,
        "ranked": 203,
        "development": 8,
        "tolerance_tiers": [1.0e-4, 1.0e-3, 1.0e-2, 4.0e-2],
    }
    ranked = load_split("ranked")
    development = load_split("dev")
    assert ranked["manifest_sha256"] == (
        "85ad37de2d7fc4b787e7eb0d5248eaebccf7ccaa04fd6f24e58637d37d6ef69f"
    )
    assert development["manifest_sha256"] == (
        "b743e8cf87f7c286b84d3ddac567d90460228b194df43e49c9e7fa8398cea454"
    )
    assert len(ranked["cases"]) == 203
    assert {case["condition_norm"] for case in ranked["cases"]} == {"1"}
    assert sorted({case["tolerance"] for case in ranked["cases"]}) == [
        1.0e-4,
        1.0e-3,
        1.0e-2,
        4.0e-2,
    ]
    assert len({case["group"] for case in development["cases"]}) == 8
    assert {case["case_id"] for case in development["cases"]} <= {
        case["case_id"] for case in ranked["cases"]
    }


def test_rhs_derivation_is_deterministic_and_keyed() -> None:
    first = rademacher_x_star("case-a", 100, b"a" * 32)
    again = rademacher_x_star("case-a", 100, b"a" * 32)
    other = rademacher_x_star("case-a", 100, b"b" * 32)
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)
    assert set(first) == {-1.0, 1.0}
    assert first[:16].astype(int).tolist() == [
        -1,
        1,
        1,
        1,
        1,
        -1,
        -1,
        -1,
        1,
        1,
        1,
        -1,
        1,
        1,
        -1,
        1,
    ]


def test_materialization_manufactures_a_consistent_rhs() -> None:
    matrix = sparse.csr_matrix(np.asarray([[3.0, 1.0], [-2.0, 4.0]]))
    canonical = CsrMatrix.from_scipy(matrix)
    case = {
        "case_id": "tiny-case",
        "n": 2,
        "nnz": 4,
        "matrix_sha256": matrix_sha256(canonical),
        "tolerance": 1.0e-4,
    }
    system = materialize_system(case, matrix, b"operator-test-key")
    np.testing.assert_array_equal(
        system.public.b, system.public.matrix.matvec(system.x_star)
    )
    np.testing.assert_array_equal(system.public.x0, np.zeros(2))
