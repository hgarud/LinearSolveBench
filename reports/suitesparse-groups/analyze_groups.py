#!/usr/bin/env python3
"""Reproducible group-level analysis of the SuiteSparse Matrix Collection.

Inputs
------
* ``ssstats.csv`` downloaded from https://sparse.tamu.edu/files/ssstats.csv
* An optional checkout of the collection website repository:
  https://github.com/ScottKolo/suitesparse-matrix-collection-website

The first source supplies the canonical index and continuous symmetry scores.  The
second supplies the website's categorical ``rb_type`` and ``structure`` labels,
including the distinction between real, complex, integer, and binary matrices.

This script deliberately uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable


CSV_FIELDS = (
    "group",
    "name",
    "nrows",
    "ncols",
    "nnz",
    "is_real",
    "is_binary",
    "is_2d3d",
    "posdef_index",
    "pattern_symmetry",
    "numerical_symmetry",
    "kind",
    "nentries",
)

WEBSITE_FIELDS = (
    "rb_type",
    "structure",
    "positive_definite",
    "cholesky_candidate",
    "structural_full_rank",
    "website_kind",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument(
        "--website-repo",
        type=Path,
        help="Checkout containing db/collection_data/matrices (recommended)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_stats(path: Path) -> tuple[list[dict], str, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        declared_count = int(next(reader)[0])
        revision = next(reader)[0]
        matrices: list[dict] = []
        for matrix_id, values in enumerate(reader, 1):
            if len(values) != len(CSV_FIELDS):
                raise ValueError(
                    f"{path}: matrix {matrix_id}: expected {len(CSV_FIELDS)} "
                    f"columns, found {len(values)}"
                )
            item = dict(zip(CSV_FIELDS, values))
            item["id"] = matrix_id
            for key in (
                "nrows",
                "ncols",
                "nnz",
                "nentries",
                "is_real",
                "is_binary",
                "is_2d3d",
                "posdef_index",
            ):
                item[key] = int(item[key])
            for key in ("pattern_symmetry", "numerical_symmetry"):
                item[key] = float(item[key])
            matrices.append(item)
    if len(matrices) != declared_count:
        raise ValueError(
            f"{path}: declared {declared_count} matrices but contained {len(matrices)}"
        )
    return matrices, revision, declared_count


def ruby_scalar(text: str, key: str) -> str:
    """Extract one simple scalar from the website's Ruby hash fixture."""
    match = re.search(
        rf"^\s*{re.escape(key)}:\s*(?:'([^']*)'|\"([^\"]*)\"|([^,\n]*)),?\s*$",
        text,
        re.MULTILINE,
    )
    if not match:
        return "missing"
    value = next((part for part in match.groups() if part is not None), "")
    return value.strip()


def enrich_from_website(matrices: list[dict], repo: Path | None) -> dict:
    provenance = {
        "available": False,
        "repository": "https://github.com/ScottKolo/suitesparse-matrix-collection-website",
    }
    if repo is None:
        for item in matrices:
            for key in WEBSITE_FIELDS:
                item[key] = "missing"
        return provenance

    metadata_root = repo / "db" / "collection_data" / "matrices"
    if not metadata_root.is_dir():
        raise ValueError(f"website metadata directory not found: {metadata_root}")

    for item in matrices:
        path = metadata_root / item["group"] / f"{item['name']}.rb"
        if not path.is_file():
            raise ValueError(f"website metadata record not found: {path}")
        text = path.read_text(encoding="utf-8", errors="replace")
        if ruby_scalar(text, "matrix_id") != str(item["id"]):
            raise ValueError(f"matrix-id mismatch between index and {path}")
        for key in WEBSITE_FIELDS:
            ruby_key = "kind" if key == "website_kind" else key
            item[key] = ruby_scalar(text, ruby_key)

    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    provenance.update(
        {
            "available": True,
            "checkout": str(repo.resolve()),
            "commit": commit,
            "metadata_records": len(matrices),
        }
    )
    return provenance


def shape_class(item: dict) -> str:
    return "square" if item["nrows"] == item["ncols"] else "rectangular"


def exact_symmetry_class(item: dict) -> str:
    """Mutually exclusive class based on continuous ssstats symmetry metrics."""
    if item["nrows"] != item["ncols"]:
        return "rectangular"
    psym = item["pattern_symmetry"]
    nsym = item["numerical_symmetry"]
    if psym < 0 or nsym < 0:
        return "unknown"
    if math.isclose(nsym, 1.0, abs_tol=1e-12):
        return "exact_numerical_or_hermitian"
    if math.isclose(psym, 1.0, abs_tol=1e-12):
        return "pattern_symmetric_only"
    if math.isclose(psym, 0.0, abs_tol=1e-12):
        return "pattern_unsymmetric"
    return "partially_pattern_symmetric"


def posdef_class(item: dict) -> str:
    return {1: "yes", 0: "no", -1: "unknown"}.get(
        item["posdef_index"], f"other_{item['posdef_index']}"
    )


def size_class(item: dict) -> str:
    dimension = max(item["nrows"], item["ncols"])
    if dimension < 1_000:
        return "lt_1k"
    if dimension < 10_000:
        return "1k_to_10k"
    if dimension < 100_000:
        return "10k_to_100k"
    if dimension < 1_000_000:
        return "100k_to_1m"
    return "ge_1m"


def counts(items: Iterable[dict], classifier: str | Callable[[dict], str]) -> Counter:
    if isinstance(classifier, str):
        return Counter(str(item[classifier]) for item in items)
    return Counter(classifier(item) for item in items)


def counts_text(values: Counter) -> str:
    return "; ".join(f"{key}:{values[key]}" for key in sorted(values))


def dominant(values: Counter) -> tuple[str, float]:
    # Deterministic alphabetical tie breaking.
    key, count = sorted(values.items(), key=lambda pair: (-pair[1], pair[0]))[0]
    return key, count / sum(values.values())


def impurity(values: Counter) -> float:
    """Fraction outside the modal category; zero means categorical homogeneity."""
    return 1.0 - max(values.values()) / sum(values.values())


def log10_span(values: Iterable[int]) -> float:
    positive = [value for value in values if value > 0]
    if not positive:
        return 0.0
    return math.log10(max(positive) / min(positive))


def median_number(values: Iterable[int | float]) -> int | float:
    value = statistics.median(values)
    return int(value) if float(value).is_integer() else value


def numeric_summary(items: list[dict], key: str) -> dict:
    values = [item[key] for item in items]
    return {
        "min": min(values),
        "median": median_number(values),
        "max": max(values),
    }


def outlier_names(items: list[dict], key: str, largest: bool) -> str:
    target = (max if largest else min)(item[key] for item in items)
    return "; ".join(
        f"{item['group']}/{item['name']}={target}"
        for item in items
        if item[key] == target
    )


def group_row(group: str, items: list[dict]) -> dict:
    category_specs: list[tuple[str, str | Callable[[dict], str]]] = [
        ("shape", shape_class),
        ("structure", "structure"),
        ("entry_type", "rb_type"),
        ("exact_symmetry", exact_symmetry_class),
        ("positive_definite", posdef_class),
        ("kind", "kind"),
        ("structural_full_rank", "structural_full_rank"),
        ("is_2d3d", lambda item: "yes" if item["is_2d3d"] == 1 else "no"),
        ("cholesky_candidate", "cholesky_candidate"),
        ("size_band", size_class),
    ]
    categorical: dict[str, Counter] = {
        label: counts(items, classifier) for label, classifier in category_specs
    }
    dominants = {label: dominant(value) for label, value in categorical.items()}

    # The score is descriptive, not an official SuiteSparse metric.  It is the
    # mean fraction outside the modal category for six matrix attributes.  Kind
    # is included because a provenance group may span application domains.
    diversity_attributes = (
        "shape",
        "structure",
        "entry_type",
        "exact_symmetry",
        "positive_definite",
        "kind",
    )
    categorical_diversity = statistics.mean(
        impurity(categorical[label]) for label in diversity_attributes
    )

    max_dims = [max(item["nrows"], item["ncols"]) for item in items]
    dim_span = log10_span(max_dims)
    nnz_span = log10_span(item["nnz"] for item in items)
    nentries_span = log10_span(item["nentries"] for item in items)

    # Labels help readers screen the full table; the component metrics remain
    # the evidence.  Small groups are marked as insufficient rather than being
    # mistaken for strongly homogeneous populations.
    mixed_count = sum(
        dominants[label][1] < 0.90 for label in diversity_attributes
    )
    if len(items) < 5:
        diversity_label = "small_group_lt5"
    elif mixed_count >= 3:
        diversity_label = "categorically_heterogeneous"
    elif mixed_count == 0:
        diversity_label = (
            "type_homogeneous_size_diverse"
            if max(dim_span, nnz_span) >= 3.0
            else "homogeneous"
        )
    else:
        diversity_label = "mixed_some_attributes"

    row: dict[str, object] = {
        "group": group,
        "matrix_count": len(items),
        "diversity_label": diversity_label,
        "categorical_diversity_score": round(categorical_diversity, 6),
        "mixed_attribute_count_dominant_share_lt_0_90": mixed_count,
        "distinct_kinds": len(categorical["kind"]),
        "dimension_log10_span": round(dim_span, 6),
        "nnz_log10_span": round(nnz_span, 6),
        "nentries_log10_span": round(nentries_span, 6),
        "matrices_with_explicit_zeros": sum(
            item["nentries"] > item["nnz"] for item in items
        ),
        "explicit_zero_count_total": sum(
            item["nentries"] - item["nnz"] for item in items
        ),
    }
    for key in (
        "nrows",
        "ncols",
        "nnz",
        "nentries",
        "pattern_symmetry",
        "numerical_symmetry",
    ):
        summary = numeric_summary(items, key)
        row.update({f"{key}_{stat}": value for stat, value in summary.items()})
    for label, value in categorical.items():
        row[f"{label}_counts"] = counts_text(value)
        row[f"dominant_{label}"] = dominants[label][0]
        row[f"dominant_{label}_share"] = round(dominants[label][1], 6)
    for key in ("nrows", "ncols", "nnz", "nentries"):
        row[f"smallest_{key}_matrices"] = outlier_names(items, key, False)
        row[f"largest_{key}_matrices"] = outlier_names(items, key, True)
    return row


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    if fieldnames is None:
        fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_distribution_rows(
    matrices: list[dict], specifications: list[tuple[str, str | Callable[[dict], str]]]
) -> list[dict]:
    rows = []
    for attribute, classifier in specifications:
        value_counts = counts(matrices, classifier)
        for value, count in sorted(value_counts.items()):
            rows.append(
                {
                    "attribute": attribute,
                    "value": value,
                    "count": count,
                    "share": round(count / len(matrices), 8),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    matrices, revision, declared_count = load_stats(args.stats)
    website_provenance = enrich_from_website(matrices, args.website_repo)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    by_group: dict[str, list[dict]] = defaultdict(list)
    for item in matrices:
        by_group[item["group"]].append(item)
    group_rows = [group_row(group, by_group[group]) for group in sorted(by_group)]

    matrix_fields = ["id", *CSV_FIELDS, *WEBSITE_FIELDS]
    write_csv(output_dir / "matrix_metadata.csv", matrices, matrix_fields)
    write_csv(output_dir / "group_summary.csv", group_rows)
    write_csv(
        output_dir / "group_diversity_ranking.csv",
        sorted(
            group_rows,
            key=lambda row: (
                -float(row["categorical_diversity_score"]),
                -int(row["matrix_count"]),
                str(row["group"]),
            ),
        ),
    )

    group_size_counts = Counter(len(items) for items in by_group.values())
    group_size_rows = []
    for size, number_of_groups in sorted(group_size_counts.items()):
        matrices_at_size = size * number_of_groups
        group_size_rows.append(
            {
                "group_size": size,
                "number_of_groups": number_of_groups,
                "number_of_matrices": matrices_at_size,
                "share_of_groups": round(number_of_groups / len(by_group), 8),
                "share_of_matrices": round(matrices_at_size / len(matrices), 8),
            }
        )
    write_csv(output_dir / "group_size_distribution.csv", group_size_rows)

    distribution_specs: list[tuple[str, str | Callable[[dict], str]]] = [
        ("shape", shape_class),
        ("website_structure", "structure"),
        ("entry_type", "rb_type"),
        (
            "numeric_domain_isReal_flag",
            lambda item: "real_or_noncomplex" if item["is_real"] == 1 else "complex",
        ),
        (
            "binary_isBinary_flag",
            lambda item: "binary" if item["is_binary"] == 1 else "not_binary",
        ),
        ("exact_symmetry_class", exact_symmetry_class),
        ("positive_definite", posdef_class),
        ("is_2d3d", lambda item: str(item["is_2d3d"])),
        ("structural_full_rank", "structural_full_rank"),
        ("kind", "kind"),
        ("size_band_by_max_dimension", size_class),
    ]
    distribution_rows = make_distribution_rows(matrices, distribution_specs)
    write_csv(output_dir / "collection_attribute_distributions.csv", distribution_rows)

    largest_rows = []
    for key in ("nrows", "ncols", "nnz", "nentries"):
        for rank, item in enumerate(
            sorted(matrices, key=lambda value: (-value[key], value["id"]))[:20], 1
        ):
            largest_rows.append(
                {
                    "metric": key,
                    "rank": rank,
                    "id": item["id"],
                    "matrix": f"{item['group']}/{item['name']}",
                    "value": item[key],
                    "nrows": item["nrows"],
                    "ncols": item["ncols"],
                    "nnz": item["nnz"],
                    "nentries": item["nentries"],
                    "structure": item["structure"],
                    "entry_type": item["rb_type"],
                    "kind": item["kind"],
                }
            )
    write_csv(output_dir / "largest_matrices.csv", largest_rows)

    kind_mismatches = [
        {
            "id": item["id"],
            "matrix": f"{item['group']}/{item['name']}",
            "ssstats_kind": item["kind"],
            "website_repo_kind": item["website_kind"],
        }
        for item in matrices
        if item["website_kind"] != "missing" and item["website_kind"] != item["kind"]
    ]
    if kind_mismatches:
        write_csv(output_dir / "kind_source_mismatches.csv", kind_mismatches)

    group_sizes = [len(items) for items in by_group.values()]
    summary = {
        "source": {
            "ssstats_url": "https://sparse.tamu.edu/files/ssstats.csv",
            "ss_index_url": "https://sparse.tamu.edu/files/ss_index.mat",
            "ssstats_revision_embedded": revision,
            "website_metadata": website_provenance,
        },
        "collection": {
            "declared_matrix_count": declared_count,
            "parsed_matrix_count": len(matrices),
            "group_count": len(by_group),
            "group_size": {
                "min": min(group_sizes),
                "median": median_number(group_sizes),
                "mean": statistics.mean(group_sizes),
                "max": max(group_sizes),
                "singleton_groups": sum(value == 1 for value in group_sizes),
                "groups_lt_5": sum(value < 5 for value in group_sizes),
            },
            "nrows": numeric_summary(matrices, "nrows"),
            "ncols": numeric_summary(matrices, "ncols"),
            "nnz": numeric_summary(matrices, "nnz"),
            "nentries": numeric_summary(matrices, "nentries"),
            "matrices_with_explicit_zeros": sum(
                item["nentries"] > item["nnz"] for item in matrices
            ),
            "explicit_zero_count_total": sum(
                item["nentries"] - item["nnz"] for item in matrices
            ),
            "pattern_symmetry": numeric_summary(matrices, "pattern_symmetry"),
            "numerical_symmetry": numeric_summary(matrices, "numerical_symmetry"),
            "pattern_symmetry_missing_count": sum(
                item["pattern_symmetry"] < 0 for item in matrices
            ),
            "numerical_symmetry_missing_count": sum(
                item["numerical_symmetry"] < 0 for item in matrices
            ),
            "ssstats_distinct_kind_count": len({item["kind"] for item in matrices}),
            "website_distinct_kind_count": len(
                {
                    item["website_kind"]
                    for item in matrices
                    if item["website_kind"] != "missing"
                }
            ),
            "ssstats_vs_website_kind_mismatch_count": len(kind_mismatches),
        },
        "definitions_and_cautions": {
            "group": (
                "A provenance/source label, typically a person, organization, or "
                "earlier collection; it is not a homogeneity guarantee."
            ),
            "nnz": "Numerically nonzero entries in A; explicit stored zeros are excluded.",
            "nentries": (
                "All stored pattern entries, equal to nnz plus explicit zeros supplied "
                "by the matrix author."
            ),
            "entry_type": (
                "The website's mutually exclusive Rutherford-Boeing entry-type label: "
                "real, complex, integer, or binary. In the ssstats flags, integer and "
                "binary matrices are also real/non-complex; isBinary is a separate flag."
            ),
            "pattern_symmetry": (
                "Matched off-diagonal structural entries divided by all off-diagonal "
                "entries; rectangular matrices are assigned 0."
            ),
            "numerical_symmetry": (
                "Matched conjugate off-diagonal values divided by all off-diagonal "
                "entries; this measures Hermitian symmetry and rectangular matrices "
                "are assigned 0."
            ),
            "posdef_index": (
                "1 known positive definite; 0 known not positive definite; -1 symmetric/"
                "Hermitian with unknown definiteness. No -1 values occur in this snapshot."
            ),
            "categorical_diversity_score": (
                "Unofficial descriptive metric: the mean fraction outside the modal "
                "category across shape, website structure, entry type, exact symmetry "
                "class, positive-definite status, and problem kind."
            ),
            "diversity_label": (
                "Unofficial screening rule. Groups with fewer than five matrices are "
                "not classified; heterogeneous means at least three attributes have a "
                "modal share below 90%; homogeneous means none do."
            ),
            "kind_source_discrepancy": (
                "Group statistics use the ssstats.csv kind. Four Chevron matrices are "
                "'other problem' in ssstats.csv but '2D/3D problem' in the website "
                "repository; see kind_source_mismatches.csv."
            ),
        },
    }
    (output_dir / "collection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
