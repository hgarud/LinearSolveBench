"""Bind the replacement cohort to its public selection and numerical evidence."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from linear_solver_bench.dataset import identity_sha256
from linear_solver_bench.manifests import (
    load_release,
    validate_qualification,
    validate_release_splits,
)

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text())


def test_cohort_preserves_operators_and_qualifies_fresh_workloads():
    releases = [
        load_release(ROOT / f"data/ns-mesh-cohort-v2-{split}.json", official=True)
        for split in ("dev", "ranked")
    ]
    validate_release_splits(releases)
    original = {
        case["source"]["matrix_id"]: case
        for split in ("dev", "ranked")
        for case in read(f"data/ns-mesh-pilot-{split}.json")["cases"]
    }
    cases = {c["source"]["matrix_id"]: c for r in releases for c in r["cases"]}
    assert len(cases) == 49
    assert set(cases) == set(original) | {231, 917, 2647, 2649, 2815, 2825}
    contract = read("data/ns-mesh-pilot-dev.json")
    configured = tomllib.loads((ROOT / "benchmark.toml").read_text())["pilot"][
        "ns_mesh"
    ]
    assert configured["development_manifest"] == "ns-mesh-cohort-v2-dev"
    assert configured["ranked_manifest"] == "ns-mesh-cohort-v2-ranked"
    for release in releases:
        assert release["execution"] == contract["execution"]
        assert release["contracts"] == contract["contracts"]
        assert release["rhs"]["draw_index"] == 0
        assert (release["rhs"]["public_key_hex"] is None) == (
            release["split"] == "ranked"
        )
        for case in release["cases"]:
            validate_qualification(case["qualification"])
            assert case["tolerance"] == 1e-12
            old = original.get(case["source"]["matrix_id"])
            if old is not None:
                assert case["source"] == old["source"]
                assert (
                    case["qualification"]["system_sha256"]
                    != old["qualification"]["system_sha256"]
                )


def test_releases_match_frozen_scientific_selection():
    inventory = read("docs/analysis/ns-cohort-v2-inventory.json")
    selection = read("docs/analysis/ns-cohort-v2-selection.json")
    policy = read("docs/analysis/ns-cohort-v2-policy.json")
    descriptors = read("docs/analysis/ns-cohort-v2-descriptors.json")
    assert inventory["descriptor_sha256"] == identity_sha256(descriptors)
    assert inventory["inventory_sha256"] == identity_sha256(
        {k: v for k, v in inventory.items() if k != "inventory_sha256"}
    )
    assert selection["inventory_sha256"] == inventory["inventory_sha256"]
    assert selection["policy_sha256"] == identity_sha256(policy)
    assert selection["selection_sha256"] == identity_sha256(
        {k: v for k, v in selection.items() if k != "selection_sha256"}
    )
    source_cases = {c["matrix_id"]: c for c in inventory["cases"]}
    sources = {s["matrix_id"]: s for s in inventory["sources"]}
    for split in ("dev", "ranked"):
        release = read(f"data/ns-mesh-cohort-v2-{split}.json")
        assert [c["source"]["matrix_id"] for c in release["cases"]] == selection[
            "matrix_ids"
        ][split]
        assert len(release["cases"]) == selection["case_counts"][split]
        groups = sorted({c["provenance_group"] for c in release["cases"]})
        assert groups == selection["groups"][split]
        assert len(groups) == selection["group_counts"][split]
        for case in release["cases"]:
            source = source_cases[case["source"]["matrix_id"]]
            assert case["source"] == sources[source["matrix_id"]]
            assert case["provenance_group"] == source["provenance_group"]
            assert case["source"]["matrix_sha256"] == source["matrix_sha256"]
            assert (case["n"], case["nnz"]) == (source["n"], source["nnz"])
    assert not set(selection["groups"]["dev"]) & set(selection["groups"]["ranked"])
    for regime in selection["regime_coverage"]:
        counts = regime["counts"]
        if regime["required_placement"] == "both-splits":
            assert counts["dev"] > 0 and counts["ranked"] > 0
        elif regime["required_placement"] == "dev-only":
            assert counts["dev"] > 0 and counts["ranked"] == 0
    assert all(
        not gap["supported_by_multiple_groups"]
        for gap in selection["joint_size_method_gaps"]
    )


def test_minimum_dependencies_reproduce_every_fresh_system():
    receipt = read("data/releases/ns-cohort-v2-preparation-reproduction.json")
    assert receipt["receipt_sha256"] == identity_sha256(
        {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    )
    assert (
        receipt["inventory_sha256"]
        == read("docs/analysis/ns-cohort-v2-inventory.json")["inventory_sha256"]
    )
    assert receipt["case_count"] == 49 and receipt["all_match"] is True
    assert len(receipt["cases"]) == 49
    assert receipt["numpy_version"] == "2.0.2"
    assert receipt["scipy_version"] == "1.14.1"
    cases = {
        case["source"]["matrix_id"]: case
        for split in ("dev", "ranked")
        for case in read(f"data/ns-mesh-cohort-v2-{split}.json")["cases"]
    }
    assert {r["matrix_id"] for r in receipt["cases"]} == set(cases)
    for row in receipt["cases"]:
        case = cases[row["matrix_id"]]
        assert row["matrix_matches"] and row["system_matches"]
        assert row["matrix_sha256"] == case["source"]["matrix_sha256"]
        assert row["system_sha256"] == case["qualification"]["system_sha256"]


def test_full_venue_reports_bind_every_case_and_preserve_failures():
    from linear_solver_bench.pilot_scoring import score_pilot_report

    for split in ("dev", "ranked"):
        release = read(f"data/ns-mesh-cohort-v2-{split}.json")
        report = read(f"data/releases/ns-cohort-v2-{split}-validation.json")
        score = score_pilot_report(report)
        assert score["official"] is True
        assert report["release"] == release
        assert report["case_count"] == len(release["cases"])
        assert report["repetitions"] == 1
        assert [c["case_id"] for c in report["cases"]] == [
            c["case_id"] for c in release["cases"]
        ]
        assert all(c["execution"]["index"] == 0 for c in report["cases"])
        assert report["venue"] == {
            "id": "modal-sandbox-pilot-cpu-v2",
            "cpus": 2,
            "memory_bytes": 4 * 1024**3,
            "limits_enforced": True,
        }
        assert (
            report["runtime_manifest_sha256"]
            == read("data/releases/cpu-runtime-v2.json")["manifest_sha256"]
        )
        assert report["solved_count"] == sum(c["passed"] for c in report["cases"])
    ranked = read("data/releases/ns-cohort-v2-ranked-validation.json")
    timed_out = next(
        c for c in ranked["cases"] if c["case_id"] == "ss-2825-rademacher-0"
    )
    assert timed_out["execution"]["outcome"] == "timeout"
    assert timed_out["passed"] is False


def test_offline_reference_attempts_preserve_fixed_inputs_and_prior_failures():
    history = read("data/releases/ns-cohort-v2-offline-reference-attempts.json")
    assert history["manifest_sha256"] == identity_sha256(
        {k: v for k, v in history.items() if k != "manifest_sha256"}
    )
    assert (
        history["inventory_sha256"]
        == read("docs/analysis/ns-cohort-v2-inventory.json")["inventory_sha256"]
    )
    assert (
        history["selection_sha256"]
        == read("docs/analysis/ns-cohort-v2-selection.json")["selection_sha256"]
    )
    expected = {
        2647: ["reference_construction_failure", "qualified"],
        2649: ["offline_time_limit", "operator_cancelled", "qualified"],
        2815: ["qualified"],
    }
    assert {row["matrix_id"] for row in history["cases"]} == set(expected)
    for row in history["cases"]:
        report = read(f"data/releases/ns-cohort-v2-{row['split']}-validation.json")
        prepared = next(
            c
            for c in report["prepared_manifest"]["cases"]
            if c["case_id"] == row["case_id"]
        )
        assert row["system_sha256"] == prepared["system_sha256"]
        assert row["input_archive_sha256"] == prepared["sha256"]
        attempts = row["attempts"]
        assert [a["outcome"] for a in attempts] == expected[row["matrix_id"]]
        assert [a["attempt_index"] for a in attempts] == list(
            range(1, len(attempts) + 1)
        )
        for attempt in attempts:
            if attempt["outcome"] == "qualified":
                assert attempt["qualification_sha256"] == identity_sha256(
                    prepared["qualification"]
                )
            else:
                assert attempt["qualification_sha256"] is None
