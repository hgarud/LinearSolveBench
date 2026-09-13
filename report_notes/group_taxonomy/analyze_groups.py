#!/usr/bin/env python3
"""Build a reproducible group-level view of the SuiteSparse CSV index.

The public ssstats.csv lacks column headers; the first twelve fields are
documented at https://sparse.tamu.edu/statistics.  A thirteenth current field
is retained as ``nentries`` but is not needed by this analysis.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GROUP_SOURCE = Path("/tmp/ssweb.o4KLNI/db/collection_data/groups")
MATRIX_SOURCE = Path("/tmp/ssweb.o4KLNI/db/collection_data/matrices")


def clean_kind(kind: str) -> str:
    kind = re.sub(r"^(subsequent|duplicate)\s+", "", kind.strip())
    kind = re.sub(r"\s+sequence$", "", kind)
    return kind


def family(kind: str) -> str:
    k = clean_kind(kind).lower()
    if "power network" in k:
        return "electrical/circuit/device"
    if "biochemical network" in k:
        return "chemistry/biology"
    if re.search(r"\b(?:graph|multigraph)\b", k) or "term/document" in k:
        return "graph/network data"
    if any(x in k for x in ("linear programming", "optimization", "optimal control", "combinatorial")):
        return "optimization/control"
    if any(x in k for x in ("structural", "materials")):
        return "structures/materials"
    if any(x in k for x in ("circuit", "semiconductor")):
        return "electrical/circuit/device"
    if any(x in k for x in ("fluid", "oceanography")):
        return "fluids/transport"
    if any(x in k for x in ("electromagnet", "acoustic")):
        return "electromagnetics/acoustics"
    if any(x in k for x in ("chemistry", "chemical")):
        return "chemistry/biology"
    if any(x in k for x in ("model reduction", "eigenvalue")):
        return "eigenvalue/model reduction"
    if any(x in k for x in ("graphics", "vision", "robotics", "tomography")):
        return "imaging/graphics/robotics"
    if any(x in k for x in ("2d/3d", "thermal")):
        return "generic spatial PDE/thermal"
    if any(x in k for x in ("statistical", "counter-example", "mathematical")):
        return "mathematics/statistics"
    if "data analytics" in k:
        return "data analytics"
    return "other"


def official_notes() -> dict[str, str]:
    result = {}
    for p in sorted(GROUP_SOURCE.glob("*.rb")):
        text = p.read_text(errors="replace")
        match = re.search(r"notes:\s*'(.*)'\s*,?\s*\n?\s*}\s*$", text, re.S)
        note = match.group(1) if match else ""
        note = note.replace("\\'", "'")
        result[p.stem] = note.strip()
    return result


def current_website_metadata() -> dict[tuple[str, str], dict[str, str]]:
    """Read current curator labels from the official website source.

    At analysis time these differ from the downloadable 2023 CSV for only the
    four Chevron matrices (current: 2D/3D problem; CSV: other problem).
    """
    result = {}
    if not MATRIX_SOURCE.exists():
        return result
    for p in MATRIX_SOURCE.glob("*/*.rb"):
        text = p.read_text(errors="replace")
        name_match = re.search(r"^\s*name:\s*'([^']+)'", text, re.M)
        fields = {}
        for field in ("kind", "rb_type", "structure"):
            match = re.search(rf"^\s*{field}:\s*'([^']+)'", text, re.M)
            if match:
                fields[field] = match.group(1)
        if name_match and "kind" in fields:
            result[(p.parent.name, name_match.group(1))] = fields
    return result


def synopsis(note: str, limit: int = 210) -> str:
    lines = [re.sub(r"\s+", " ", line.strip()) for line in note.splitlines() if line.strip()]
    text = " ".join(lines[:2])
    text = re.sub(r"https?://\S+", "", text).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def label_span(lo: int, hi: int) -> str:
    if lo == hi:
        return "one size"
    ratio = hi / max(lo, 1)
    if ratio <= 4:
        return "narrow (≤4×)"
    if ratio <= 100:
        return "moderate (4–100×)"
    return "wide (>100×)"


def main() -> None:
    raw = list(csv.reader((ROOT / "ssstats.csv").open(newline="")))
    declared_count, revision = int(raw[0][0]), raw[1][0]
    rows = raw[2:]
    assert len(rows) == declared_count
    website_metadata = current_website_metadata()

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        group, name, nrows, ncols, nnz = row[:5]
        current = website_metadata.get((group, name), {})
        fallback_rb_type = "binary" if int(row[6]) else "real" if int(row[5]) else "complex"
        fallback_structure = "rectangular" if nrows != ncols else "symmetric/Hermitian" if float(row[10]) == 1 else "unsymmetric"
        grouped[group].append(
            {
                "name": name,
                "nrows": int(nrows),
                "ncols": int(ncols),
                "nnz": int(nnz),
                "is_real": int(row[5]),
                "is_binary": int(row[6]),
                "is_nd": int(row[7]),
                "posdef": int(row[8]),
                "pattern_symmetry": float(row[9]),
                "numerical_symmetry": float(row[10]),
                "kind": current.get("kind", row[11]),
                "rb_type": current.get("rb_type", fallback_rb_type),
                "structure": current.get("structure", fallback_structure),
            }
        )

    notes = official_notes()
    output = []
    for group, members in grouped.items():
        n = len(members)
        dims = [max(x["nrows"], x["ncols"]) for x in members]
        nnzs = [x["nnz"] for x in members]
        square = [x for x in members if x["nrows"] == x["ncols"]]
        rect = n - len(square)
        # The official statistic compares A with A' (conjugate transpose in
        # MATLAB), so it means symmetric for real matrices and Hermitian for
        # complex matrices.
        exact_sym = sum(x["numerical_symmetry"] == 1 for x in square)
        structurally_sym = sum(x["pattern_symmetry"] == 1 for x in square)
        kinds = Counter(clean_kind(str(x["kind"])) for x in members)
        families = Counter(family(str(x["kind"])) for x in members)
        rb_types = Counter(str(x["rb_type"]) for x in members)
        structures = Counter(str(x["structure"]) for x in members)
        family_names = sorted(families)
        shape_profile = "square only" if not rect else "rectangular only" if not square else "mixed"
        if not square:
            symmetry_profile = "N/A (rectangular only)"
        elif exact_sym == len(square):
            symmetry_profile = "all square matrices symmetric/Hermitian"
        elif exact_sym == 0:
            symmetry_profile = "no square matrix exactly symmetric/Hermitian"
        else:
            symmetry_profile = "mixed among square matrices"
        value_profile = (
            "real only" if all(x["is_real"] for x in members)
            else "complex only" if all(not x["is_real"] for x in members)
            else "real + complex"
        )
        form_mixed = len(structures) > 1 or len(rb_types) > 1
        if n == 1:
            cohesion = "single matrix"
        elif len(families) > 1:
            cohesion = "multi-domain/mixed"
        elif form_mixed or len(kinds) > 2:
            cohesion = "one broad domain; mixed matrix forms"
        else:
            cohesion = "coherent family/series"

        dominant_family, dominant_count = families.most_common(1)[0]
        note = notes.get(group, "")
        output.append(
            {
                "group": group,
                "count": n,
                "documented_meaning": synopsis(note),
                "source_url": f"https://sparse.tamu.edu/{group}",
                "coarse_kind_families_interpretive": "; ".join(family_names),
                "dominant_coarse_family_interpretive": dominant_family,
                "dominant_coarse_family_share": dominant_count / n,
                "normalized_kind_count": len(kinds),
                "normalized_kinds": "; ".join(f"{k} ({v})" for k, v in kinds.most_common()),
                "curator_value_type_count": len(rb_types),
                "curator_value_types": "; ".join(f"{k} ({v})" for k, v in rb_types.most_common()),
                "curator_structure_count": len(structures),
                "curator_structures": "; ".join(f"{k} ({v})" for k, v in structures.most_common()),
                "min_dimension": min(dims),
                "max_dimension": max(dims),
                "dimension_ratio": max(dims) / max(min(dims), 1),
                "size_span": label_span(min(dims), max(dims)),
                "min_nnz": min(nnzs),
                "max_nnz": max(nnzs),
                "shape_profile": shape_profile,
                "square": len(square),
                "rectangular": rect,
                "symmetry_profile": symmetry_profile,
                "exact_symmetric_square": exact_sym,
                "structurally_symmetric_square": structurally_sym,
                "known_spd": sum(x["posdef"] == 1 for x in members),
                "value_profile": value_profile,
                "binary": sum(x["is_binary"] for x in members),
                "geometric_2d3d": sum(x["is_nd"] for x in members),
                "cohesion": cohesion,
                "official_notes": note,
            }
        )

    fields = list(output[0])
    with (ROOT / "group_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(output)
    (ROOT / "group_metrics.json").write_text(json.dumps({"revision": revision, "matrix_count": declared_count, "groups": output}, indent=2))

    with (ROOT / "group_catalogue.md").open("w") as f:
        f.write("# SuiteSparse group catalogue\n\n")
        f.write(f"Generated from the public `ssstats.csv` index ({declared_count:,} matrices; revision {revision}), current curator kind labels in the official website source, and official group notes.\n\n")
        f.write("`Numerically self-adjoint` means numerical symmetry = 1 in the official index (A=Aᵀ for real A; A=A* for complex A); rectangular matrices are excluded from that denominator. The exact curator `structure` column separately preserves complex-symmetric and skew-symmetric classes. `Dimension` is max(rows, columns). Kinds have only the index's `subsequent`, `duplicate`, and `sequence` qualifiers collapsed.\n\n")
        f.write("| Group | N | Documented meaning/source | Exact curator kind(s) | Coarse kind family (interpretive) | Dimension range | Curator structure | Numerically self-adjoint | Curator value type | Assessment (heuristic) |\n")
        f.write("|---|---:|---|---|---|---:|---|---:|---|---|\n")
        for x in sorted(output, key=lambda x: x["group"].lower()):
            desc = str(x["documented_meaning"]).replace("|", "\\|")
            link = f"[{x['group']}]({x['source_url']})"
            f.write(
                f"| {link} | {x['count']} | {desc} | {x['normalized_kinds']} | {x['coarse_kind_families_interpretive']} | "
                f"{x['min_dimension']:,}–{x['max_dimension']:,} ({x['size_span']}) | "
                f"{x['curator_structures']} | {x['exact_symmetric_square']}/{x['square']} square | "
                f"{x['curator_value_types']} | {x['cohesion']} |\n"
            )

    summary = {
        "revision": revision,
        "matrices": declared_count,
        "groups": len(output),
        "group_size": Counter(x["size_span"] for x in output),
        "group_shape": Counter(x["shape_profile"] for x in output),
        "group_symmetry": Counter(x["symmetry_profile"] for x in output),
        "group_values": Counter(x["value_profile"] for x in output),
        "groups_mixing_curator_value_type": sum(x["curator_value_type_count"] > 1 for x in output),
        "groups_mixing_curator_structure": sum(x["curator_structure_count"] > 1 for x in output),
        "group_cohesion": Counter(x["cohesion"] for x in output),
        "single_matrix_groups": sum(x["count"] == 1 for x in output),
        "multi_kind_groups": sum(x["normalized_kind_count"] > 1 for x in output),
        "multi_family_groups_interpretive": sum(";" in x["coarse_kind_families_interpretive"] for x in output),
    }
    (ROOT / "aggregate_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
