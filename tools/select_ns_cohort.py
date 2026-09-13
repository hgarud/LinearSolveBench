"""Select indivisible NS source groups using a frozen matrix-only policy.

The supported contract is docs/analysis/ns-cohort-v2-policy.json. Every group
partition is considered; hard coverage and budget rules precede the objective.
No solver outcomes, reference solutions, random draws, or matrix arrays enter.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from linear_solver_bench.dataset import identity_sha256
from linear_solver_bench.manifests import digest, https_url, positive_int

SUPPORTED_POLICY_SHA256 = (
    "e5c7bfd1f224621c12101fb38a07a7cba1d1f19304901e235c2826f8a4d9188c"
)
FEATURES = (
    "log10_n",
    "log10_nnz_per_row",
    "relative_nonsymmetry",
    "zero_diagonal_fraction",
    "weakly_diagonally_dominant_row_fraction",
)
SIZE_LABELS = ("n<10000", "10000<=n<100000", "100000<=n<1000000", "n>=1000000")
CASE_FIELDS = {
    "matrix_id",
    "name",
    "matrix_sha256",
    "provenance_group",
    "n",
    "nnz",
    "features",
    "application_tags",
    "discretization_tags",
}
SOURCE_FIELDS = {
    "kind",
    "matrix_id",
    "group",
    "name",
    "url",
    "archive_bytes",
    "archive_sha256",
    "matrix_sha256",
}
PARTITION_CHUNK = 65_536
SCORE_CHUNK = 4_096


def _validate(inventory: dict, policy: dict) -> list[dict]:
    if (
        not isinstance(policy, dict)
        or identity_sha256(policy) != SUPPORTED_POLICY_SHA256
    ):
        raise ValueError("incompatible selection policy schema or values")
    fields = {
        "schema_version",
        "kind",
        "cases",
        "sources",
        "descriptor_sha256",
        "inventory_sha256",
    }
    if (
        not isinstance(inventory, dict)
        or set(inventory) != fields
        or type(inventory["schema_version"]) is not int
        or inventory["schema_version"] != 1
        or inventory["kind"] != "ns-cohort-source-inventory"
    ):
        raise ValueError("invalid cohort source inventory schema")
    body = {k: v for k, v in inventory.items() if k != "inventory_sha256"}
    if inventory["inventory_sha256"] != identity_sha256(body):
        raise ValueError("cohort inventory digest mismatch")
    digest(inventory["descriptor_sha256"], "descriptor_sha256")
    if not isinstance(inventory["cases"], list) or not inventory["cases"]:
        raise ValueError("cohort inventory requires cases")
    if not isinstance(inventory["sources"], list):
        raise ValueError("cohort inventory requires source records")
    sources = {}
    for source in inventory["sources"]:
        if not isinstance(source, dict) or set(source) != SOURCE_FIELDS:
            raise ValueError("invalid inventory source fields")
        matrix_id = positive_int(source["matrix_id"], "matrix_id")
        if source["kind"] != "suitesparse" or matrix_id in sources:
            raise ValueError("invalid or duplicate inventory source")
        for key in ("group", "name"):
            if not isinstance(source[key], str) or not source[key]:
                raise ValueError("source names must be nonempty strings")
        https_url(source["url"])
        positive_int(source["archive_bytes"], "archive_bytes")
        digest(source["archive_sha256"], "archive_sha256")
        digest(source["matrix_sha256"], "matrix_sha256")
        sources[matrix_id] = source
    ids, identities = set(), set()
    for case in inventory["cases"]:
        if not isinstance(case, dict) or set(case) != CASE_FIELDS:
            raise ValueError(
                "invalid case fields; split and outcome inputs are forbidden"
            )
        matrix_id = positive_int(case["matrix_id"], "matrix_id")
        n = positive_int(case["n"], "n")
        nnz = positive_int(case["nnz"], "nnz")
        identity = digest(case["matrix_sha256"], "matrix_sha256")
        if matrix_id in ids or identity in identities:
            raise ValueError("duplicate cohort matrix identity")
        ids.add(matrix_id)
        identities.add(identity)
        source = sources.get(matrix_id)
        if source is None or (
            case["name"] != source["group"] + "/" + source["name"]
            or identity != source["matrix_sha256"]
        ):
            raise ValueError("case and source identities differ")
        if (
            not isinstance(case["provenance_group"], str)
            or not case["provenance_group"]
        ):
            raise ValueError("case requires a provenance group")
        if not isinstance(case["features"], dict) or set(case["features"]) != set(
            FEATURES
        ):
            raise ValueError("case requires exactly the five matrix-only features")
        for name, value in case["features"].items():
            if type(value) not in {int, float} or not math.isfinite(value):
                raise ValueError("matrix features must be finite numbers")
            if name.endswith("_fraction") and not 0 <= value <= 1:
                raise ValueError("fraction features must lie in [0, 1]")
        if case["features"]["relative_nonsymmetry"] < 0:
            raise ValueError("relative nonsymmetry must be nonnegative")
        for name, expected in (
            ("log10_n", math.log10(n)),
            ("log10_nnz_per_row", math.log10(nnz / n)),
        ):
            if not math.isclose(
                case["features"][name], expected, abs_tol=1e-12, rel_tol=1e-12
            ):
                raise ValueError("matrix log features disagree with dimensions")
        for field in policy["coverage"]["known_tag_fields"]:
            tags = case[field]
            if (
                not isinstance(tags, list)
                or not tags
                or any(not isinstance(tag, str) or not tag for tag in tags)
                or len(tags) != len(set(tags))
            ):
                raise ValueError("case tags must be nonempty unique strings")
    if ids != set(sources):
        raise ValueError("source and case inventories differ")
    return sorted(inventory["cases"], key=lambda case: case["matrix_id"])


def _size_label(n: int) -> str:
    return SIZE_LABELS[sum(n >= boundary for boundary in (10_000, 100_000, 1_000_000))]


def _endpoint(value: float) -> str:
    return "0" if value == 0 else "1" if value == 1 else "(0,1)"


def _strata(cases: list[dict], policy: dict) -> dict[tuple[str, str], list[int]]:
    strata = defaultdict(list)
    excluded = set(policy["coverage"]["excluded_tags"])
    for i, case in enumerate(cases):
        strata[("size_band", _size_label(case["n"]))].append(i)
        for feature in policy["coverage"]["endpoint_features"]:
            strata[(feature, _endpoint(case["features"][feature]))].append(i)
        for field in policy["coverage"]["known_tag_fields"]:
            for tag in sorted(set(case[field]) - excluded):
                strata[(field, tag)].append(i)
        for method in sorted(set(case["discretization_tags"]) - excluded):
            strata[
                ("size_method_joint", _size_label(case["n"]) + " / " + method)
            ].append(i)
    return dict(sorted(strata.items()))


def _count_table(weights: np.ndarray) -> np.ndarray:
    table = np.zeros(1 << len(weights), dtype=np.int64)
    for i, weight in enumerate(weights):
        start = 1 << i
        table[start : 2 * start] = table[:start] + weight
    return table


class _Objective:
    """Preaggregate per-group CDF counts, tag incidence, and pair distances."""

    def __init__(self, cases: list[dict], group_ids: np.ndarray, group_count: int):
        self.n = len(cases)
        self.group_count = group_count
        self.membership = np.eye(group_count)[group_ids]
        features = np.array([[c["features"][f] for f in FEATURES] for c in cases])
        spread = np.ptp(features, axis=0)
        self.features = np.divide(
            features - features.min(axis=0),
            spread,
            out=np.zeros_like(features),
            where=spread != 0,
        )
        self.distances = cdist(self.features, self.features)
        self.maximum_distance = float(self.distances.max())
        self.group_distances = self.membership.T @ self.distances @ self.membership
        self.group_distance_totals = self.group_distances.sum(axis=1)
        self.total_distance = float(self.group_distances.sum())
        cdfs, gaps = [], []
        for feature in range(len(FEATURES)):
            order = np.argsort(self.features[:, feature], kind="stable")
            cdfs.append(np.cumsum(self.membership[order], axis=0)[:-1].T)
            gaps.append(np.diff(self.features[order, feature]) / len(FEATURES))
        self.group_cdfs = np.concatenate(cdfs, axis=1)
        self.cdf_totals = self.group_cdfs.sum(axis=0)
        self.cdf_gaps = np.concatenate(gaps)
        tags = sorted(
            {
                (field, tag)
                for c in cases
                for field in ("application_tags", "discretization_tags")
                for tag in c[field]
                if tag != "unknown"
            }
        )
        incidence = np.array([[tag in c[field] for field, tag in tags] for c in cases])
        self.group_tags = self.membership.T @ incidence
        self.tag_totals = self.group_tags.sum(axis=0)

    def score(self, masks: np.ndarray, dev_counts: np.ndarray) -> np.ndarray:
        chosen = (masks[:, None] >> np.arange(self.group_count, dtype=np.uint32)) & 1
        chosen = chosen.astype(np.float64)
        dev = dev_counts.astype(np.float64)
        ranked = self.n - dev
        cdf = chosen @ self.group_cdfs
        marginal = (
            np.abs(cdf / dev[:, None] - (self.cdf_totals - cdf) / ranked[:, None])
            @ self.cdf_gaps
        )
        within_dev = np.sum((chosen @ self.group_distances) * chosen, axis=1)
        cross = chosen @ self.group_distance_totals - within_dev
        within_ranked = self.total_distance - within_dev - 2 * cross
        energy = (
            2 * cross / (dev * ranked) - within_dev / dev**2 - within_ranked / ranked**2
        )
        energy = (
            np.maximum(energy, 0) / (2 * self.maximum_distance)
            if self.maximum_distance
            else np.zeros_like(dev)
        )
        if self.group_tags.shape[1]:
            counts = chosen @ self.group_tags
            tags = np.mean(
                np.abs(
                    counts / dev[:, None] - (self.tag_totals - counts) / ranked[:, None]
                ),
                axis=1,
            )
        else:
            tags = np.zeros_like(dev)
        budget = np.abs(dev / self.n - 1 / 3)
        return np.column_stack((marginal, energy, tags, budget))


def _coverage_report(cases: list[dict], strata: dict, dev_ids: set[int]) -> list[dict]:
    rows = []
    for (kind, label), indices in strata.items():
        groups = sorted({cases[i]["provenance_group"] for i in indices})
        dev = sum(cases[i]["matrix_id"] in dev_ids for i in indices)
        rows.append(
            {
                "kind": kind,
                "stratum": label,
                "supporting_groups": groups,
                "required_placement": (
                    "both-splits"
                    if len(groups) >= 2
                    else "report-only"
                    if kind == "size_method_joint"
                    else "dev-only"
                ),
                "counts": {"dev": dev, "ranked": len(indices) - dev},
                "dev_only": len(groups) == 1 and kind != "size_method_joint",
            }
        )
    return rows


def _joint_gaps(cases: list[dict], dev_ids: set[int]) -> list[dict]:
    joint = defaultdict(list)
    for case in cases:
        for method in set(case["discretization_tags"]) - {"unknown"}:
            joint[(_size_label(case["n"]), method)].append(case)
    gaps = []
    for (size, method), rows in sorted(joint.items()):
        dev = sum(c["matrix_id"] in dev_ids for c in rows)
        if dev in {0, len(rows)}:
            groups = sorted({c["provenance_group"] for c in rows})
            gaps.append(
                {
                    "size_band": size,
                    "spatial_method": method,
                    "counts": {"dev": dev, "ranked": len(rows) - dev},
                    "supporting_groups": groups,
                    "supported_by_multiple_groups": len(groups) >= 2,
                    "hard_constraint": len(groups) >= 2,
                }
            )
    return gaps


def _partition_summary(
    cases: list[dict], groups: list[str], mask: int, components: np.ndarray, policy: dict
) -> dict:
    dev_groups = {g for i, g in enumerate(groups) if mask & (1 << i)}
    by_split = {
        split: [
            c for c in cases if (c["provenance_group"] in dev_groups) == (split == "dev")
        ]
        for split in ("dev", "ranked")
    }
    return {
        "matrix_ids": {
            split: [c["matrix_id"] for c in rows] for split, rows in by_split.items()
        },
        "groups": {
            split: sorted({c["provenance_group"] for c in rows})
            for split, rows in by_split.items()
        },
        "case_counts": {split: len(rows) for split, rows in by_split.items()},
        "group_counts": {
            split: len({c["provenance_group"] for c in rows})
            for split, rows in by_split.items()
        },
        "score": float(np.mean(components)),
        "rounded_score": round(float(np.mean(components)), 12),
        "score_components": dict(
            zip(policy["objective"]["components"], map(float, components), strict=True)
        ),
    }


def select_cohort(inventory: dict, policy: dict) -> dict:
    """Return the best feasible partition, retaining every admitted input case."""
    cases = _validate(inventory, policy)
    groups = sorted({case["provenance_group"] for case in cases})
    if len(groups) > policy["search"]["maximum_groups"]:
        raise ValueError("exhaustive selection supports at most 26 provenance groups")
    group_lookup = {group: i for i, group in enumerate(groups)}
    group_ids = np.array([group_lookup[c["provenance_group"]] for c in cases])
    strata = _strata(cases, policy)
    shared_masks, forced_dev = [], 0
    for (kind, _), indices in strata.items():
        mask = sum(1 << int(i) for i in set(group_ids[indices]))
        if mask.bit_count() == 1 and kind != "size_method_joint":
            forced_dev |= mask
        elif mask.bit_count() >= 2:
            shared_masks.append(mask)
    weights = np.bincount(group_ids, minlength=len(groups))
    half = len(groups) // 2
    low, high = _count_table(weights[:half]), _count_table(weights[half:])
    low_mask = (1 << half) - 1
    objective = _Objective(cases, group_ids, len(groups))
    total_partitions = 1 << len(groups)
    feasible_count = 0
    # Keep the winner and five alternatives under exactly the same policy key.
    leaders = []
    for start in range(0, total_partitions, PARTITION_CHUNK):
        masks = np.arange(
            start, min(start + PARTITION_CHUNK, total_partitions), dtype=np.uint32
        )
        counts = low[masks & low_mask] + high[masks >> half]
        valid = (masks & forced_dev) == forced_dev
        valid &= (4 * counts >= len(cases)) & (2 * counts <= len(cases))
        valid &= np.bitwise_count(masks) <= len(groups) - 8
        masks, counts = masks[valid], counts[valid]
        for shared in sorted(set(shared_masks)):
            selected = masks & shared
            valid = (selected != 0) & (selected != shared)
            masks, counts = masks[valid], counts[valid]
        feasible_count += len(masks)
        for offset in range(0, len(masks), SCORE_CHUNK):
            selected = masks[offset : offset + SCORE_CHUNK]
            components = objective.score(
                selected, counts[offset : offset + SCORE_CHUNK]
            )
            scores = np.mean(components, axis=1)
            # Values more than one rounding unit above this batch's sixth raw
            # score cannot enter its top six. Keep all nearby rounding ties.
            sixth = min(5, len(scores) - 1)
            cutoff = float(np.partition(scores, sixth)[sixth]) + 1.001e-12
            contenders = np.flatnonzero(scores <= cutoff)
            batch_leaders = []
            for index in contenders:
                mask = int(selected[index])
                dev_groups = tuple(g for i, g in enumerate(groups) if mask & (1 << i))
                key = (round(float(scores[index]), 12), dev_groups)
                batch_leaders.append((key, mask, components[index].copy()))
            batch_leaders.sort(key=lambda entry: entry[0])
            leaders = sorted(
                leaders + batch_leaders[:6], key=lambda entry: entry[0]
            )[:6]
    if not leaders:
        forced = [g for i, g in enumerate(groups) if forced_dev & (1 << i)]
        raise ValueError(
            f"No feasible cohort partition among {total_partitions} group partitions: "
            "coverage requires shared strata on both sides, singleton strata in dev, "
            "25%-50% dev cases and at least 8 ranked groups; "
            f"singleton strata force these groups into dev: {forced}"
        )
    _, best_mask, best_components = leaders[0]
    selected_partition = _partition_summary(
        cases, groups, best_mask, best_components, policy
    )
    dev_ids = set(selected_partition["matrix_ids"]["dev"])
    dev_indices = [i for i, c in enumerate(cases) if c["matrix_id"] in dev_ids]
    nearest = []
    for i, case in enumerate(cases):
        if case["matrix_id"] in dev_ids:
            continue
        closest = dev_indices[int(np.argmin(objective.distances[i, dev_indices]))]
        nearest.append(
            {
                "ranked_matrix_id": case["matrix_id"],
                "dev_matrix_id": cases[closest]["matrix_id"],
                "distance": float(objective.distances[i, closest]),
            }
        )
    body = {
        "schema_version": 1,
        "kind": "ns-cohort-group-selection",
        "policy_sha256": identity_sha256(policy),
        "inventory_sha256": inventory["inventory_sha256"],
        "enumerated_partition_count": total_partitions,
        "feasible_partition_count": feasible_count,
        **selected_partition,
        "alternatives": [
            {
                "selection_rank": rank,
                **_partition_summary(cases, groups, mask, components, policy),
            }
            for rank, (_, mask, components) in enumerate(leaders[1:], start=2)
        ],
        "alternative_limit": 5,
        "regime_coverage": _coverage_report(cases, strata, dev_ids),
        "nearest_dev_feature_neighbors": nearest,
        "nearest_neighbor_metric": (
            "Euclidean distance in five pooled-range-normalized features"
        ),
        "joint_size_method_gaps": _joint_gaps(cases, dev_ids),
        "limitations": list(policy["limitations"])
        + [
            "Energy uses empirical V-statistic pair averages, including self-pairs; "
            "it is not the square root of that statistic.",
            "Constant feature ranges and an all-zero pair distance contribute zero; "
            "an empty known-tag inventory contributes zero to the tag block.",
            "Nearest-neighbor distances and single-source size-by-method joint gaps "
            "are descriptive diagnostics, not additional selection constraints.",
            "Alternatives are the next five feasible partitions after the winner, "
            "ordered by the same rounded-score and group-tuple policy key.",
        ],
    }
    return {**body, "selection_sha256": identity_sha256(body)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = select_cohort(
            json.loads(args.inventory.read_text()), json.loads(args.policy.read_text())
        )
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "groups": result["groups"],
                "case_counts": result["case_counts"],
                "score": result["score"],
                "feasible_partition_count": result["feasible_partition_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
