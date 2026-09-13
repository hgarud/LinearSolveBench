#!/usr/bin/env python3
"""Derive immutable ranked and public-development manifests from evidence."""

from __future__ import annotations

import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TOLERANCE_TIERS = (1.0e-4, 1.0e-3, 1.0e-2, 4.0e-2)
CONDITION_METHOD = "spqr-guarded-onenormest-v1"
PUBLIC_DEV_KEY = hashlib.sha256(b"linear-solver-bench-public-dev-v1").digest()


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode("ascii")).hexdigest()


def case_tolerance(result: dict[str, object]) -> float:
    minimum = float(result["condition"]["minimum_safe_tolerance"])
    for tolerance in TOLERANCE_TIERS:
        if minimum <= tolerance:
            return tolerance
    raise RuntimeError("full-rank case exceeds the v1 tolerance tiers")


def case_record(index: int, result: dict[str, object]) -> dict[str, object]:
    condition = result["condition"]
    return {
        "index": index,
        "case_id": result["case_id"],
        "matrix_id": result["matrix_id"],
        "group": result["group"],
        "name": result["name"],
        "n": result["n"],
        "nnz": result["nnz"],
        "source_url": f"https://sparse.tamu.edu/MM/{result['group']}/{result['name']}.tar.gz",
        "archive_sha256": result["archive_sha256"],
        "matrix_sha256": result["matrix_sha256"],
        "condition_method": condition["method"],
        "condition_norm": condition["condition_norm"],
        "condition_estimate": condition["condition_estimate"],
        "minimum_safe_tolerance": condition["minimum_safe_tolerance"],
        "tolerance": case_tolerance(result),
    }


def write_manifest(filename: str, body: dict[str, object]) -> None:
    value = {**body, "manifest_sha256": digest(body)}
    (DATA / filename).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    qualification = json.loads((DATA / "qualification.json").read_text())
    eligible = []
    for result in qualification["results"]:
        condition = result["condition"]
        if (
            condition["status"] == "estimated"
            and condition["tightest_supported_tolerance"] is not None
        ):
            eligible.append(result)
    eligible.sort(key=lambda item: item["index"])
    ranked_cases = [case_record(index, result) for index, result in enumerate(eligible)]
    if len(ranked_cases) != 203:
        raise RuntimeError("ranked v1 selection must contain 203 cases")
    ranked_body = {
        "schema_version": 1,
        "benchmark_id": "linear-solver-bench-cpu-v1",
        "split": "ranked",
        "catalogue_sha256": qualification["catalogue_sha256"],
        "qualification_sha256": qualification["condition_results_sha256"],
        "selection": "spqr-full-rank-and-condition-supported-v1",
        "rhs_scheme": "rademacher-hmac-pcg64-v1",
        "tolerance_policy": {
            "id": "condition-aware-four-tier-v1",
            "condition_norm": "1",
            "condition_method": CONDITION_METHOD,
            "tiers": list(TOLERANCE_TIERS),
            "criterion": "minimum_safe_tolerance<=case_tolerance",
        },
        "case_count": len(ranked_cases),
        "cases": ranked_cases,
    }
    write_manifest("ranked-v1.json", ranked_body)

    # Smallest case from each distinct group, then stable matrix identity.
    ordered = sorted(ranked_cases, key=lambda case: (case["n"], case["matrix_id"]))
    dev_source = []
    groups = set()
    for case in ordered:
        if case["group"] not in groups:
            groups.add(case["group"])
            dev_source.append(case)
        if len(dev_source) == 8:
            break
    dev_cases = [{**case, "index": index} for index, case in enumerate(dev_source)]
    dev_body = {
        "schema_version": 1,
        "benchmark_id": "linear-solver-bench-cpu-v1",
        "split": "dev",
        "source_ranked_manifest_sha256": digest(ranked_body),
        "selection": "eight-smallest-distinct-groups-v1",
        "rhs_scheme": "rademacher-hmac-pcg64-v1",
        "public_rhs_key_hex": PUBLIC_DEV_KEY.hex(),
        "tolerance_policy": ranked_body["tolerance_policy"],
        "case_count": len(dev_cases),
        "cases": dev_cases,
    }
    write_manifest("dev-v1.json", dev_body)


if __name__ == "__main__":
    main()
