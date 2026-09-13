from __future__ import annotations

import copy
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

from linear_solver_bench.dataset import identity_sha256

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def selector():
    spec = importlib.util.spec_from_file_location(
        "ns_cohort_selector", ROOT / "tools/select_ns_cohort.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def policy():
    return json.loads((ROOT / "docs/analysis/ns-cohort-v2-policy.json").read_text())


def seal(inventory):
    body = {k: v for k, v in inventory.items() if k != "inventory_sha256"}
    return {**body, "inventory_sha256": identity_sha256(body)}


def make_inventory(count=12, *, constant=False):
    cases, sources = [], []
    for i in range(count):
        n = 100 if constant else 100 * (i + 1)
        nnz = n * (5 if constant else 1 + i % 3)
        identity = f"{i + 1:064x}"
        cases.append(
            {
                "matrix_id": i + 1,
                "name": f"Fixture/m{i}",
                "matrix_sha256": identity,
                "provenance_group": f"g{i:02d}",
                "n": n,
                "nnz": nnz,
                "features": {
                    "log10_n": math.log10(n),
                    "log10_nnz_per_row": math.log10(nnz / n),
                    "relative_nonsymmetry": 1.0 if constant else 0.1 + (i % 4) / 10,
                    "zero_diagonal_fraction": 0.0 if constant else (0, 0.25, 1)[i % 3],
                    "weakly_diagonally_dominant_row_fraction": (
                        1.0 if constant else (0, 0.5, 1)[(i // 2) % 3]
                    ),
                },
                "application_tags": ["flow"],
                "discretization_tags": [
                    "finite_element" if constant or i % 2 else "finite_difference"
                ],
            }
        )
        sources.append(
            {
                "kind": "suitesparse",
                "matrix_id": i + 1,
                "group": "Fixture",
                "name": f"m{i}",
                "url": f"https://sparse.tamu.edu/MM/Fixture/m{i}.tar.gz",
                "archive_bytes": 1,
                "archive_sha256": f"{i + 100:064x}",
                "matrix_sha256": identity,
            }
        )
    return seal(
        {
            "schema_version": 1,
            "kind": "ns-cohort-source-inventory",
            "cases": cases,
            "sources": sources,
            "descriptor_sha256": "a" * 64,
        }
    )


def brute_force(inventory, policy):
    """Independent scalar enumeration with SciPy's marginal W1 implementation."""
    cases = sorted(inventory["cases"], key=lambda c: c["matrix_id"])
    groups = sorted({c["provenance_group"] for c in cases})
    support = defaultdict(set)
    for case in cases:
        group = case["provenance_group"]
        size = sum(case["n"] >= n for n in policy["coverage"]["size_boundaries"])
        support[("size", size)].add(group)
        for f in policy["coverage"]["endpoint_features"]:
            v = case["features"][f]
            support[(f, v if v in (0, 1) else "interior")].add(group)
        for f in policy["coverage"]["known_tag_fields"]:
            for tag in case[f]:
                if tag != "unknown":
                    support[(f, tag)].add(group)
        for method in case["discretization_tags"]:
            if method != "unknown":
                support[("joint", size, method)].add(group)
    fields = policy["objective"]["features"]
    x = np.array([[c["features"][f] for f in fields] for c in cases])
    ranges = np.ptp(x, axis=0)
    scaled = np.divide(
        x - x.min(axis=0), ranges, out=np.zeros_like(x), where=ranges != 0
    )
    distances = cdist(scaled, scaled)
    tags = sorted(
        {
            (f, t)
            for c in cases
            for f in policy["coverage"]["known_tag_fields"]
            for t in c[f]
            if t != "unknown"
        }
    )
    feasible = []
    for mask in range(1 << len(groups)):
        dev_groups = tuple(g for i, g in enumerate(groups) if mask & (1 << i))
        dev_set = set(dev_groups)
        a = np.array([c["provenance_group"] in dev_set for c in cases])
        if not 0.25 <= a.mean() <= 0.5 or len(groups) - len(dev_set) < 8:
            continue
        if any(
            (len(gs) > 1 and not (gs & dev_set and gs - dev_set))
            or (len(gs) == 1 and key[0] != "joint" and not gs <= dev_set)
            for key, gs in support.items()
        ):
            continue
        b = ~a
        w1 = np.mean(
            [
                wasserstein_distance(x[a, j], x[b, j]) / ranges[j] if ranges[j] else 0
                for j in range(len(fields))
            ]
        )
        energy = (
            2 * distances[np.ix_(a, b)].mean()
            - distances[np.ix_(a, a)].mean()
            - distances[np.ix_(b, b)].mean()
        )
        energy = max(0, energy) / (2 * distances.max()) if distances.max() else 0
        incidence = (
            np.mean(
                [
                    abs(
                        np.mean(
                            [
                                t in c[f]
                                for c, take in zip(cases, a, strict=True)
                                if take
                            ]
                        )
                        - np.mean(
                            [
                                t in c[f]
                                for c, take in zip(cases, b, strict=True)
                                if take
                            ]
                        )
                    )
                    for f, t in tags
                ]
            )
            if tags
            else 0
        )
        components = [w1, energy, incidence, abs(a.mean() - 1 / 3)]
        feasible.append(
            ((round(float(np.mean(components)), 12), dev_groups), components)
        )
    return sorted(feasible, key=lambda entry: entry[0])


def test_objective_and_feasible_count_match_independent_brute_force(selector, policy):
    inventory = make_inventory()
    expected = brute_force(inventory, policy)
    result = selector.select_cohort(inventory, policy)
    assert result["enumerated_partition_count"] == 4096
    assert result["feasible_partition_count"] == len(expected)
    assert tuple(result["groups"]["dev"]) == expected[0][0][1]
    assert result["rounded_score"] == expected[0][0][0]
    assert list(result["score_components"].values()) == pytest.approx(expected[0][1])
    assert len(result["alternatives"]) == 5
    for rank, (actual, reference) in enumerate(
        zip(result["alternatives"], expected[1:6], strict=True), start=2
    ):
        assert actual["selection_rank"] == rank
        assert tuple(actual["groups"]["dev"]) == reference[0][1]
        assert actual["rounded_score"] == reference[0][0]
        assert list(actual["score_components"].values()) == pytest.approx(reference[1])
        assert sum(actual["case_counts"].values()) == len(inventory["cases"])
    body = {k: v for k, v in result.items() if k != "selection_sha256"}
    assert identity_sha256(body) == result["selection_sha256"]


def test_constant_features_have_zero_distance_and_lexicographic_tie(selector, policy):
    result = selector.select_cohort(make_inventory(constant=True), policy)
    assert result["groups"]["dev"] == ["g00", "g01", "g02", "g03"]
    assert result["feasible_partition_count"] == 715
    assert result["score"] == 0
    assert all(row["distance"] == 0 for row in result["nearest_dev_feature_neighbors"])
    expected = brute_force(make_inventory(constant=True), policy)
    assert [tuple(a["groups"]["dev"]) for a in result["alternatives"]] == [
        row[0][1] for row in expected[1:6]
    ]


def test_partition_and_score_batch_boundaries_preserve_top_six(
    selector, policy, monkeypatch
):
    inventory = make_inventory()
    expected = selector.select_cohort(inventory, policy)
    monkeypatch.setattr(selector, "PARTITION_CHUNK", 37)
    monkeypatch.setattr(selector, "SCORE_CHUNK", 7)
    actual = selector.select_cohort(inventory, policy)
    assert actual["matrix_ids"] == expected["matrix_ids"]
    assert actual["feasible_partition_count"] == expected["feasible_partition_count"]
    assert [a["groups"] for a in actual["alternatives"]] == [
        a["groups"] for a in expected["alternatives"]
    ]
    assert [a["rounded_score"] for a in actual["alternatives"]] == [
        a["rounded_score"] for a in expected["alternatives"]
    ]


def test_group_integrity_and_input_permutation_do_not_change_partition(
    selector, policy
):
    inventory = make_inventory()
    inventory["cases"][1]["provenance_group"] = "g00"
    inventory = seal(inventory)
    before = copy.deepcopy(inventory)
    first = selector.select_cohort(inventory, policy)
    assert inventory == before
    inventory["cases"].reverse()
    inventory["sources"].reverse()
    second = selector.select_cohort(seal(inventory), policy)
    for key in ("matrix_ids", "groups", "score_components", "feasible_partition_count"):
        assert first[key] == second[key]
    for ids in first["matrix_ids"].values():
        assert (1 in ids) == (2 in ids)
    assert sorted(sum(first["matrix_ids"].values(), [])) == list(range(1, 13))


def test_single_group_known_regime_is_dev_only_and_unknown_is_excluded(
    selector, policy
):
    inventory = make_inventory(constant=True)
    inventory["cases"][0]["application_tags"].append("rare_application")
    inventory["cases"][1]["discretization_tags"] = ["unknown"]
    result = selector.select_cohort(seal(inventory), policy)
    assert "g00" in result["groups"]["dev"]
    rare = next(
        r for r in result["regime_coverage"] if r["stratum"] == "rare_application"
    )
    assert rare["required_placement"] == "dev-only" and rare["dev_only"]
    assert rare["counts"] == {"dev": 1, "ranked": 0}
    assert not any(r["stratum"] == "unknown" for r in result["regime_coverage"])


def test_shared_size_method_joints_are_hard_constraints(selector, policy):
    inventory = make_inventory(constant=True)
    for i, case in enumerate(inventory["cases"]):
        case["discretization_tags"] = [
            "finite_element" if i % 2 else "finite_difference"
        ]
        if i >= 6:
            case.update(n=20_000, nnz=100_000)
            case["features"]["log10_n"] = math.log10(20_000)
    inventory = seal(inventory)
    expected = brute_force(inventory, policy)
    result = selector.select_cohort(inventory, policy)
    assert result["feasible_partition_count"] == len(expected)
    assert tuple(result["groups"]["dev"]) == expected[0][0][1]
    joint = [r for r in result["regime_coverage"] if r["kind"] == "size_method_joint"]
    assert len(joint) == 4
    assert all(r["counts"]["dev"] and r["counts"]["ranked"] for r in joint)
    assert result["joint_size_method_gaps"] == []


def test_single_group_joint_can_remain_ranked_and_is_reported(selector, policy):
    inventory = make_inventory(constant=True)
    for i, case in enumerate(inventory["cases"]):
        if i in {5, 11}:
            case["discretization_tags"] = ["finite_difference"]
        if i >= 6:
            case.update(n=20_000, nnz=100_000)
            case["features"]["log10_n"] = math.log10(20_000)
    inventory["cases"][5]["application_tags"].append("rare_application")
    inventory = seal(inventory)
    expected = brute_force(inventory, policy)
    result = selector.select_cohort(inventory, policy)
    assert tuple(result["groups"]["dev"]) == expected[0][0][1]
    assert "g05" in result["groups"]["dev"]
    assert "g11" in result["groups"]["ranked"]
    joint = next(
        r
        for r in result["regime_coverage"]
        if r["kind"] == "size_method_joint"
        and r["stratum"] == "10000<=n<100000 / finite_difference"
    )
    assert joint["required_placement"] == "report-only"
    assert not joint["dev_only"]
    assert joint["counts"] == {"dev": 0, "ranked": 1}
    assert any(
        r["size_band"] == "10000<=n<100000"
        and r["spatial_method"] == "finite_difference"
        and not r["hard_constraint"]
        for r in result["joint_size_method_gaps"]
    )


def test_three_opposite_split_constraints_form_an_infeasible_odd_cycle(
    selector, policy
):
    inventory = make_inventory(constant=True)
    for group_indices, tag in (
        ((0, 1), "edge-a"),
        ((1, 2), "edge-b"),
        ((0, 2), "edge-c"),
    ):
        for i in group_indices:
            inventory["cases"][i]["application_tags"].append(tag)
    with pytest.raises(ValueError, match="No feasible cohort partition"):
        selector.select_cohort(seal(inventory), policy)
    # Removing any edge makes these constraints compatible with the budget too.
    for tag in ("edge-a", "edge-b", "edge-c"):
        reduced = copy.deepcopy(inventory)
        for case in reduced["cases"]:
            case["application_tags"] = [t for t in case["application_tags"] if t != tag]
        assert selector.select_cohort(seal(reduced), policy)["feasible_partition_count"]


def test_infeasibility_and_incompatible_inputs_fail_explicitly(selector, policy):
    inventory = make_inventory(constant=True)
    for i, case in enumerate(inventory["cases"]):
        case["application_tags"] = [f"unique-{i}"]
    with pytest.raises(ValueError, match="No feasible cohort partition"):
        selector.select_cohort(seal(inventory), policy)
    changed_policy = copy.deepcopy(policy)
    changed_policy["budget"]["maximum_dev_fraction"] = 0.75
    with pytest.raises(ValueError, match="incompatible selection policy"):
        selector.select_cohort(make_inventory(), changed_policy)
    inventory = make_inventory()
    inventory["cases"][0]["solver_passed"] = True
    with pytest.raises(ValueError, match="outcome inputs are forbidden"):
        selector.select_cohort(seal(inventory), policy)
    with pytest.raises(ValueError, match="at most 26"):
        selector.select_cohort(make_inventory(count=27, constant=True), policy)
