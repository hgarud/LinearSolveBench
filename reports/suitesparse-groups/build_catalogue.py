#!/usr/bin/env python3
"""Build the readable all-group catalogue from the verified analysis outputs.

The numeric inputs are produced by ``analyze_groups.py``.  Group descriptions
come from the official SuiteSparse Matrix Collection website repository at the
commit recorded in ``data/provenance.json``.
"""

from __future__ import annotations

import argparse
import csv
import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-summary", type=Path, required=True)
    parser.add_argument("--matrix-metadata", type=Path, required=True)
    parser.add_argument("--website-repo", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalize_kind(kind: str) -> str:
    for prefix in ("subsequent ", "duplicate "):
        if kind.startswith(prefix):
            kind = kind[len(prefix) :]
    if kind.endswith(" sequence"):
        kind = kind[: -len(" sequence")]
    return kind


def group_description(repo: Path, group: str) -> str:
    path = repo / "db" / "collection_data" / "groups" / f"{group}.rb"
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"\bnotes:\s*'(.*)'\s*,\s*\n\s*}\s*$", text, re.DOTALL)
    if not match:
        raise ValueError(f"could not parse group notes: {path}")
    notes = match.group(1).replace("\\'", "'")
    lines = [
        line.strip()
        for line in notes.splitlines()
        if line.strip() and not set(line.strip()) <= {"=", "-"}
    ]
    description = lines[0] if lines else ""
    # Some official first lines deliberately continue after a comma/colon.
    if description.endswith((",", ":")) and len(lines) > 1:
        description += " " + lines[1]
    return textwrap.shorten(description, width=125, placeholder="…")


def count_text(counts: Counter[str], limit: int | None = None) -> str:
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    shown = ordered if limit is None else ordered[:limit]
    result = "; ".join(f"{name} ({count})" for name, count in shown)
    if limit is not None and len(ordered) > limit:
        result += f"; +{len(ordered) - limit} more"
    return result


def parse_count_field(value: str) -> Counter[str]:
    result: Counter[str] = Counter()
    for part in value.split("; "):
        name, count = part.rsplit(":", 1)
        result[name] = int(count)
    return result


def size_profile(dimensions: list[int]) -> str:
    ratio = max(dimensions) / min(dimensions)
    if ratio == 1:
        return "one dimension"
    if ratio <= 4:
        return "narrow (≤4×)"
    if ratio <= 100:
        return "moderate (4–100×)"
    return "wide (>100×)"


def assessment(summary: dict[str, str], n: int, dimensions: list[int]) -> str:
    label = summary["diversity_label"]
    if n == 1:
        categorical = "singleton"
    elif label == "small_group_lt5":
        score = float(summary["categorical_diversity_score"])
        categorical = "small; observed mix" if score > 0 else "small; uniform observed types"
    elif label == "categorically_heterogeneous":
        categorical = "high categorical mix"
    elif label == "mixed_some_attributes":
        categorical = "some categorical mix"
    else:
        categorical = "categorically uniform"
    return f"{categorical}; {size_profile(dimensions)}"


def main() -> None:
    args = parse_args()
    summaries = {row["group"]: row for row in read_csv(args.group_summary)}
    matrices = read_csv(args.matrix_metadata)

    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for matrix in matrices:
        by_group[matrix["group"]].append(matrix)

    rows: list[dict[str, str | int]] = []
    for group in sorted(by_group):
        items = by_group[group]
        summary = summaries[group]
        dimensions = [max(int(item["nrows"]), int(item["ncols"])) for item in items]
        kinds = Counter(
            normalize_kind(item["website_kind"] if item["website_kind"] != "missing" else item["kind"])
            for item in items
        )
        shapes = Counter("square" if item["nrows"] == item["ncols"] else "rectangular" for item in items)
        structures = Counter(item["structure"] for item in items)
        entry_types = Counter(item["rb_type"] for item in items)
        pd_counts = Counter(item["positive_definite"] for item in items)
        rows.append(
            {
                "group": group,
                "matrix_count": len(items),
                "source_or_meaning": group_description(args.website_repo, group),
                "normalized_kind_counts": count_text(kinds),
                "dimension_min": min(dimensions),
                "dimension_max": max(dimensions),
                "size_profile": size_profile(dimensions),
                "shape_counts": count_text(shapes),
                "structure_counts": count_text(structures),
                "entry_type_counts": count_text(entry_types),
                "known_positive_definite": pd_counts.get("yes", 0),
                "assessment": assessment(summary, len(items), dimensions),
                "categorical_diversity_score": summary["categorical_diversity_score"],
                "source_url": f"https://sparse.tamu.edu/{group}",
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# SuiteSparse group catalogue",
        "",
        "This appendix covers all 203 groups in the 2,904-problem public snapshot.",
        "Group descriptions are condensed from each official group page; click a group",
        "name for its full provenance notes and matrix list. `Kind` is the current website",
        "label after collapsing only `subsequent`, `duplicate`, and `sequence` modifiers.",
        "`Dimension` is `max(rows, columns)`. Entry types are the mutually exclusive",
        "Rutherford–Boeing categories `real`, `binary`, `integer`, and `complex`.",
        "The assessment is an unofficial screening aid computed from six categorical fields,",
        "including the uncollapsed kind string; the component counts are the evidence.",
        "",
        "| Group | N | Source / meaning | Normalized kind(s) | Dimension | Shape | Structure | Entry type | Known PD | Assessment |",
        "|---|---:|---|---|---:|---|---|---|---:|---|",
    ]
    for row in rows:
        group = str(row["group"])
        kind_counts = Counter()
        for part in str(row["normalized_kind_counts"]).split("; "):
            match = re.match(r"(.*) \((\d+)\)$", part)
            if match:
                kind_counts[match.group(1)] = int(match.group(2))
        compact_kinds = count_text(kind_counts, limit=4)
        lines.append(
            "| "
            + " | ".join(
                [
                    f"[{group}](https://sparse.tamu.edu/{group})",
                    str(row["matrix_count"]),
                    str(row["source_or_meaning"]).replace("|", "\\|"),
                    compact_kinds.replace("|", "\\|"),
                    f"{int(row['dimension_min']):,}–{int(row['dimension_max']):,}",
                    str(row["shape_counts"]),
                    str(row["structure_counts"]),
                    str(row["entry_type_counts"]),
                    str(row["known_positive_definite"]),
                    str(row["assessment"]),
                ]
            )
            + " |"
        )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
