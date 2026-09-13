from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from linear_solver_bench.dataset import identity_sha256
from linear_solver_bench.models import CsrMatrix

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location(
        "audit_ns_split", ROOT / "tools/audit_ns_split.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matrix_diagnostics_match_analytic_operator_and_preserve_input(audit):
    original = np.array([[2.0, 1.0], [0.0, 3.0]])
    matrix = CsrMatrix.from_scipy(sparse.csr_matrix(original))
    features = audit.matrix_features(matrix)
    assert features["relative_nonsymmetry"] == pytest.approx(np.sqrt(2 / 14))
    assert features["zero_diagonal_fraction"] == 0
    assert features["weakly_diagonally_dominant_row_fraction"] == 1
    assert np.array_equal(matrix.to_scipy().toarray(), original)
    scaled = CsrMatrix.from_scipy(sparse.csr_matrix(original * -(2.0**400)))
    assert audit.matrix_features(scaled) == pytest.approx(features)


def test_exact_zero_diagonal_count_survives_extreme_dynamic_range(audit):
    matrix = CsrMatrix.from_scipy(sparse.diags([1e-300, 1e300], format="csr"))
    assert audit.matrix_features(matrix)["zero_diagonal_fraction"] == 0


def test_small_mean_distance_does_not_erase_single_group_coverage_gap(audit):
    features = {name: 0.25 for name in audit.FEATURES}
    features["log10_n"] = 2.0
    cases = [
        {"split": "dev", "n": 100, "provenance_group": "small", "features": features},
    ]
    cases.extend(
        {
            "split": "ranked",
            "n": 100,
            "provenance_group": "other-small",
            "features": dict(features),
        }
        for _ in range(31)
    )
    cases.append(
        {
            "split": "ranked",
            "n": 1_100_000,
            "provenance_group": "large",
            "features": {**features, "log10_n": float(np.log10(1_100_000))},
        }
    )
    result = audit.compare_splits(cases)
    assert 0 < result["mean_normalized_marginal_distance"] < 0.01
    assert result["size_bands"][-1]["counts"] == {"dev": 0, "ranked": 1}
    assert not result["size_bands"][-1]["can_cover_both_sides_with_distinct_groups"]
    cases[1]["provenance_group"] = "small"
    with pytest.raises(ValueError, match="crosses"):
        audit.compare_splits(cases)


def test_unknown_method_does_not_count_as_known_coverage(audit):
    features = {name: 0.0 for name in audit.FEATURES}
    cases = [
        {
            "split": split,
            "n": 100,
            "provenance_group": split,
            "features": features,
            "discretization_tags": ["unknown"],
        }
        for split in ("dev", "ranked")
    ]
    result = audit.compare_splits(cases)
    row = result["tag_coverage"]["discretization_tags"]["unknown"]
    assert row["at_least_two_source_groups"]
    assert not row["known_regime_present_in_both_splits"]


def test_published_audit_matches_frozen_sources_and_descriptor_review(audit):
    record = json.loads((ROOT / "docs/analysis/ns-split-audit.json").read_text())
    digest = record.pop("audit_sha256")
    assert identity_sha256(record) == digest
    descriptors = json.loads(
        (ROOT / "docs/analysis/ns-source-descriptors.json").read_text()
    )
    assert identity_sha256(descriptors) == record["descriptor_sha256"]
    cases = audit.add_descriptors(record["cases"], descriptors)
    assert audit.compare_splits(cases) == record["comparison"]
    releases = [
        json.loads((ROOT / f"data/ns-mesh-pilot-{split}.json").read_text())
        for split in ("dev", "ranked")
    ]
    assert (
        sorted(r["manifest_sha256"] for r in releases)
        == record["source_release_sha256"]
    )
    matrices = {c["source"]["matrix_id"]: c for r in releases for c in r["cases"]}
    assert len(cases) == len(matrices) == 43
    for case in cases:
        source = matrices[case["matrix_id"]]
        assert case["matrix_sha256"] == source["source"]["matrix_sha256"]
        assert (case["n"], case["nnz"]) == (source["n"], source["nnz"])
    changed = copy.deepcopy(descriptors)
    changed["matrices"][0]["provenance_group"] = "substituted"
    with pytest.raises(ValueError, match="identity"):
        audit.add_descriptors(cases, changed)
