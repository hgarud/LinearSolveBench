#!/usr/bin/env python3
"""Build frozen FLASH pilot manifests from a public, commit-pinned catalogue.

Supply the dataset commit returned by publication as --revision. This tool does
not upload data or register the resulting manifests as trusted releases.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import Counter, defaultdict
from urllib.parse import quote

from linear_solver_bench.dataset import sha256_file
from linear_solver_bench.families import resolve_track
from linear_solver_bench.manifests import (
    seal_manifest,
    validate_release,
    validate_release_splits,
)
from linear_solver_bench.sources.flash import load_flash

RELEASE_ID = "flash-replay-pilot"
DEFAULT_REPOSITORY = "hgarud/LinearSolveBench"
FLASH_DIFFUSION_URL = (
    "https://flash.rochester.edu/site/flashcode/user_support/"
    "flash_ug_devel/node128.html"
)
PILOT_EXECUTION = {
    "repetitions": 1,
    "cpus": 2,
    "memory_bytes": 4 * 1024**3,
    "case_timeout_seconds": 90.0,
    "max_iterations": 5000,
}


def _archive_path(value: object) -> pathlib.PurePosixPath:
    if not isinstance(value, str):
        raise ValueError("catalogue archive path must be a string")
    path = pathlib.PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or ".." in path.parts
        or "\\" in value
        or path.suffix != ".zip"
    ):
        raise ValueError("catalogue archive paths must be relative ZIP paths")
    return path


def build_releases(
    catalogue: dict,
    *,
    revision: str,
    repository: str = DEFAULT_REPOSITORY,
    archive_dir: pathlib.Path | None = None,
) -> dict[str, dict]:
    """Validate both splits, optionally inspect all archives, and return manifests."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full lowercase 40-character commit hash")
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", repository
    ):
        raise ValueError("repository must have the form owner/dataset")
    if (
        catalogue.get("schema_version") != 1
        or catalogue.get("kind") != "flash-replay-public-catalogue"
        or catalogue.get("family") != "magnetic_diffusion_flash"
    ):
        raise ValueError("unsupported public FLASH catalogue schema")
    rows, run_rows = catalogue.get("cases"), catalogue.get("source_runs")
    if not isinstance(rows, list) or not rows or not isinstance(run_rows, list):
        raise ValueError("catalogue must contain cases and source-run descriptions")
    if type(catalogue.get("case_count")) is not int or catalogue["case_count"] != len(
        rows
    ):
        raise ValueError("catalogue case count is inconsistent")
    runs = {}
    for run in run_rows:
        identity = run["source_run"]
        if (
            identity in runs
            or not isinstance(run["problem"], str)
            or not run["problem"].strip()
        ):
            raise ValueError(
                "source runs must be unique and identify their FLASH problem"
            )
        runs[identity] = run
    split_counts = Counter(row["split"] for row in rows)
    if set(split_counts) != {"dev", "ranked"} or catalogue.get("split_counts") != dict(
        split_counts
    ):
        raise ValueError(
            "catalogue must contain consistent development and ranked splits"
        )
    run_counts = Counter(row["source_run"] for row in rows)
    if set(runs) != set(run_counts):
        raise ValueError("catalogue cases and source-run descriptions disagree")
    for identity, run in runs.items():
        if run["case_count"] != run_counts[identity]:
            raise ValueError("catalogue source-run case count is inconsistent")
    paths = [_archive_path(row["path"]) for row in rows]
    if len(set(paths)) != len(paths):
        raise ValueError("each catalogue case must have its own archive path")

    base_url = f"https://huggingface.co/datasets/{repository}/resolve/{revision}"
    catalogue_url = f"{base_url}/catalogue.json"
    contract = resolve_track("magnetic_diffusion_flash", "replay")
    releases, archive_cases = {}, []
    for split in ("dev", "ranked"):
        selected = [row for row in rows if row["split"] == split]
        group_runs = defaultdict(set)
        per_run = Counter()
        for row in selected:
            if row["role"] == "scored":
                group_runs[row["provenance_group"]].add(row["source_run"])
                per_run[row["provenance_group"], row["source_run"]] += 1
        cases = []
        for index, row in enumerate(selected):
            group, run_id = row["provenance_group"], row["source_run"]
            run = runs[run_id]
            if (run["split"], run["provenance_group"]) != (split, group):
                raise ValueError(
                    "case split or provenance disagrees with its source run"
                )
            weight = 0.0
            if row["role"] == "scored":
                weight = 1 / (
                    len(group_runs) * len(group_runs[group]) * per_run[group, run_id]
                )
            case = {
                "index": index,
                **{
                    key: row[key]
                    for key in (
                        "case_id",
                        "n",
                        "nnz",
                        "tolerance",
                        "role",
                        "provenance_group",
                        "source_run",
                    )
                },
                "weight": weight,
                "source": {
                    "kind": "huggingface",
                    "url": f"{base_url}/{quote(row['path'], safe='/')}",
                    **{
                        key: row[key]
                        for key in (
                            "archive_bytes",
                            "archive_sha256",
                            "matrix_sha256",
                            "b_sha256",
                            "x0_sha256",
                        )
                    },
                },
                "admission": {
                    "statement": (
                        "Captured scalar magnetic-diffusion system from FLASH "
                        f"({run['problem']}); replay evaluates this captured "
                        "solve independently."
                    ),
                    "citations": [catalogue_url, FLASH_DIFFUSION_URL],
                    "capture_kind": "scalar-magnetic-diffusion",
                    "relative_tolerance": row["tolerance"],
                    "absolute_tolerance": 0,
                    "step": row["step"],
                    "solve_index": row["solve_index"],
                },
                "qualification": None,
            }
            cases.append(case)
            archive_cases.append((row["path"], case))
        releases[split] = validate_release(
            seal_manifest(
                {
                    "schema_version": 2,
                    "kind": "linear-solver-bench-pilot-release",
                    "release_id": RELEASE_ID,
                    "family": contract.family,
                    "track": contract.track,
                    "split": split,
                    "contracts": contract.contracts,
                    "execution": dict(PILOT_EXECUTION),
                    "rhs": None,
                    "cases": cases,
                }
            )
        )
    validate_release_splits(list(releases.values()))
    if archive_dir is not None:
        root = pathlib.Path(archive_dir).expanduser().resolve()
        for relative, case in archive_cases:
            path = root / relative
            if (
                path.is_symlink()
                or not path.is_file()
                or not path.resolve().is_relative_to(root)
            ):
                raise ValueError(
                    f"case archive is missing or outside archive-dir: {relative}"
                )
            source = case["source"]
            if (
                path.stat().st_size != source["archive_bytes"]
                or sha256_file(path) != source["archive_sha256"]
            ):
                raise ValueError(f"case archive identity mismatch: {relative}")
            load_flash(path, case)
    return releases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalogue", type=pathlib.Path, help="public catalogue.json")
    parser.add_argument(
        "--revision", required=True, help="published 40-character dataset commit"
    )
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--archive-dir",
        type=pathlib.Path,
        help="optional dataset root containing the catalogue's relative ZIP paths",
    )
    args = parser.parse_args(argv)
    try:
        catalogue = json.loads(args.catalogue.read_text(encoding="utf-8"))
        if not isinstance(catalogue, dict):
            raise ValueError("catalogue must contain a JSON object")
        releases = build_releases(
            catalogue,
            revision=args.revision,
            repository=args.repository,
            archive_dir=args.archive_dir,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for split, release in releases.items():
            path = args.output_dir / f"flash-replay-{split}-pilot.json"
            path.write_text(
                json.dumps(release, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            print(path)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"build_flash_release: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
