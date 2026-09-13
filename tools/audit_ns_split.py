"""Describe NS split balance from public matrix properties, without solver runs.

This is a descriptive audit, not a certificate of equal solver difficulty. It
does not change releases, inspect solver results, or choose a new official split.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

from linear_solver_bench.dataset import identity_sha256
from linear_solver_bench.manifests import load_release, validate_release_splits
from linear_solver_bench.models import CsrMatrix
from linear_solver_bench.paths import cache_dir
from linear_solver_bench.pilot_dataset import (
    iter_pilot_prepared,
    load_pilot_prepared_manifest,
)
from linear_solver_bench.sources.download import fetch_archive
from linear_solver_bench.sources.suitesparse import load_suitesparse

FEATURES = (
    "log10_n",
    "log10_nnz_per_row",
    "relative_nonsymmetry",
    "zero_diagonal_fraction",
    "weakly_diagonally_dominant_row_fraction",
)
SIZE_BANDS = (10_000, 100_000, 1_000_000)


def matrix_features(matrix: CsrMatrix) -> dict[str, float]:
    """Compute scale-invariant diagnostics; dominance here is not certified."""
    if matrix.nnz == 0:
        raise ValueError("split diagnostics require a nonzero operator")
    a = matrix.to_scipy().copy()
    zero_diagonal_fraction = float(np.mean(a.diagonal() == 0))
    a.data /= float(np.max(np.abs(a.data)))
    difference = a - a.T
    nonsymmetry = float(np.linalg.norm(difference.data) / np.linalg.norm(a.data))
    diagonal = np.abs(a.diagonal())
    a.data = np.abs(a.data)
    row_sums = np.asarray(a.sum(axis=1)).ravel()
    return {
        "log10_n": math.log10(matrix.n),
        "log10_nnz_per_row": math.log10(matrix.nnz / matrix.n),
        "relative_nonsymmetry": nonsymmetry,
        "zero_diagonal_fraction": zero_diagonal_fraction,
        "weakly_diagonally_dominant_row_fraction": float(
            np.mean(diagonal >= row_sums - diagonal)
        ),
    }


def _record(case: dict, matrix: CsrMatrix, split: str) -> dict:
    source = case["source"]
    return {
        "matrix_id": source["matrix_id"],
        "name": source["group"] + "/" + source["name"],
        "matrix_sha256": source["matrix_sha256"],
        "provenance_group": case["provenance_group"],
        "split": split,
        "n": case["n"],
        "nnz": case["nnz"],
        "features": matrix_features(matrix),
    }


def collect_releases(
    paths: list[Path], cache: Path, *, offline: bool = False
) -> tuple[list[dict], list[str]]:
    """Acquire only public A arrays; ranked RHS keys are not needed."""
    releases = [load_release(path) for path in paths]
    validate_release_splits(releases)
    if any(r["family"] != "ns-mesh-pde" for r in releases):
        raise ValueError("audit requires NS releases")
    cases = []
    for release in releases:
        for case in release["cases"]:
            archive = fetch_archive(case["source"], cache, offline=offline)
            matrix = load_suitesparse(archive, case)
            cases.append(_record(case, matrix, release["split"]))
            del matrix
    return sorted(cases, key=lambda c: c["matrix_id"]), sorted(
        r["manifest_sha256"] for r in releases
    )


def collect_cases(roots: list[Path]) -> tuple[list[dict], list[str]]:
    cases, releases, identities = [], [], set()
    for root in roots:
        prepared = load_pilot_prepared_manifest(root)
        if prepared["family"] != "ns-mesh-pde" or not prepared["complete_split"]:
            raise ValueError("audit requires complete prepared NS splits")
        release = prepared["source_release"]
        releases.append(release["manifest_sha256"])
        for system, case in zip(
            iter_pilot_prepared(root), release["cases"], strict=True
        ):
            identity = case["source"]["matrix_id"]
            if identity in identities:
                raise ValueError("an operator occurs in more than one input split")
            identities.add(identity)
            cases.append(_record(case, system.public.matrix, release["split"]))
            del system
    return sorted(cases, key=lambda c: c["matrix_id"]), sorted(releases)


def add_descriptors(cases: list[dict], descriptors: dict) -> list[dict]:
    records = descriptors["matrices"]
    by_id = {row["matrix_id"]: row for row in records}
    if len(by_id) != len(records) or set(by_id) != {c["matrix_id"] for c in cases}:
        raise ValueError("descriptor inventory differs from matrix inventory")
    result = []
    for case in cases:
        record = by_id[case["matrix_id"]]
        if (
            record["full_name"] != case["name"]
            or record["provenance_group"] != case["provenance_group"]
        ):
            raise ValueError("descriptor source identity differs from matrix")
        tags = {}
        for field, definition in (
            ("application_tags", "application_tag_definitions"),
            ("discretization_tags", "discretization_tag_definitions"),
        ):
            values = record[field]
            if (
                not values
                or len(set(values)) != len(values)
                or not set(values) <= set(descriptors[definition])
            ):
                raise ValueError("invalid descriptor tags")
            tags[field] = sorted(values)
        result.append({**case, **tags})
    return result


def compare_splits(cases: list[dict]) -> dict:
    """Report marginal distances and strata impossible to share across groups."""
    splits = {
        split: [c for c in cases if c["split"] == split] for split in ("dev", "ranked")
    }
    if any(not rows for rows in splits.values()) or sum(
        map(len, splits.values())
    ) != len(cases):
        raise ValueError("audit requires nonempty dev and ranked splits only")
    groups = {
        split: {c["provenance_group"] for c in rows} for split, rows in splits.items()
    }
    if groups["dev"] & groups["ranked"]:
        raise ValueError("a provenance group crosses the split boundary")
    tag_coverage = {}
    for field in ("application_tags", "discretization_tags"):
        if not all(field in case for case in cases):
            continue
        tag_coverage[field] = {}
        for tag in sorted({tag for case in cases for tag in case[field]}):
            selected = [case for case in cases if tag in case[field]]
            counts = {
                split: sum(c["split"] == split for c in selected) for split in splits
            }
            source_groups = sorted({c["provenance_group"] for c in selected})
            tag_coverage[field][tag] = {
                "counts": counts,
                "provenance_groups": source_groups,
                "known_regime_present_in_both_splits": tag != "unknown"
                and all(counts.values()),
                "at_least_two_source_groups": len(source_groups) >= 2,
            }
    features = {}
    for feature in FEATURES:
        values = {
            split: np.asarray([c["features"][feature] for c in rows])
            for split, rows in splits.items()
        }
        pooled = np.concatenate(list(values.values()))
        if not np.all(np.isfinite(pooled)):
            raise ValueError("matrix features must be finite")
        spread = float(np.ptp(pooled))
        distance = float(wasserstein_distance(values["dev"], values["ranked"]))
        features[feature] = {
            "normalized_wasserstein_distance": distance / spread if spread else 0.0,
            **{
                split: {
                    "min": float(v.min()),
                    "median": float(np.median(v)),
                    "max": float(v.max()),
                }
                for split, v in values.items()
            },
        }
    bands = []
    for low, high in zip((0, *SIZE_BANDS), (*SIZE_BANDS, None), strict=True):
        selected = [
            c for c in cases if c["n"] >= low and (high is None or c["n"] < high)
        ]
        distinct = sorted({c["provenance_group"] for c in selected})
        bands.append(
            {
                "minimum_n_inclusive": low,
                "maximum_n_exclusive": high,
                "counts": {
                    split: sum(c["split"] == split for c in selected)
                    for split in splits
                },
                "provenance_groups": distinct,
                "can_cover_both_sides_with_distinct_groups": len(distinct) >= 2,
            }
        )
    return {
        "case_counts": {split: len(rows) for split, rows in splits.items()},
        "group_counts": {split: len(values) for split, values in groups.items()},
        "groups": {
            split: dict(sorted(Counter(c["provenance_group"] for c in rows).items()))
            for split, rows in splits.items()
        },
        "size_bands": bands,
        "tag_coverage": tag_coverage,
        "features": features,
        "mean_normalized_marginal_distance": float(
            np.mean([v["normalized_wasserstein_distance"] for v in features.values()])
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prepared", type=Path, action="append")
    inputs.add_argument("--release", type=Path, action="append")
    parser.add_argument("--cache", type=Path, default=cache_dir() / "downloads")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--descriptors", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    cases, releases = (
        collect_cases(args.prepared)
        if args.prepared
        else collect_releases(args.release, args.cache, offline=args.offline)
    )
    descriptors = json.loads(args.descriptors.read_text()) if args.descriptors else None
    if descriptors is not None:
        cases = add_descriptors(cases, descriptors)
    body = {
        "schema_version": 1,
        "kind": "ns-split-descriptive-audit",
        "source_release_sha256": releases,
        "descriptor_sha256": identity_sha256(descriptors)
        if descriptors is not None
        else None,
        "method": "matrix-only-marginal-wasserstein-range-normalized-v1",
        "limitations": [
            "Marginal feature similarity does not establish joint similarity "
            "or equal solver difficulty.",
            "No condition-number, nonnormality, or solver-success guarantee "
            "is inferred.",
            "Dominance and nonsymmetry are floating-point descriptive "
            "diagnostics, not mathematical certificates.",
            "Application and discretization tags reflect the cited source "
            "review; unknown labels do not establish regime coverage.",
            "No universal acceptable-distance threshold is specified by this audit.",
        ],
        "cases": cases,
        "comparison": compare_splits(cases),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {**body, "audit_sha256": identity_sha256(body)}, indent=2, sort_keys=True
        )
        + "\n"
    )
    print(json.dumps(body["comparison"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
