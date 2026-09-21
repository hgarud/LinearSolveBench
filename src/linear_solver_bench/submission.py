"""Create and check the small pull-request submission bundle."""

from __future__ import annotations

import hashlib
import json
import pathlib
import re

from .families import FAMILY_SELECTORS


def _validate_details(name: object, families: object, submitter: object) -> None:
    if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,63}", name):
        raise ValueError("name must use 2-64 lowercase letters, digits, '-' or '_'")
    if (
        not isinstance(families, list)
        or not families
        or not all(isinstance(family, str) for family in families)
        or set(families) - set(FAMILY_SELECTORS)
    ):
        raise ValueError(f"select one or more of: {', '.join(FAMILY_SELECTORS)}")
    if not isinstance(submitter, str) or not submitter.strip():
        raise ValueError("submitter must be nonempty text")


def package(
    source: str | pathlib.Path,
    output: str | pathlib.Path,
    *,
    name: str,
    families: list[str],
    submitter: str,
) -> dict[str, object]:
    from .build import CANDIDATE_ABI, validate_source

    _validate_details(name, families, submitter)
    payload = validate_source(pathlib.Path(source)).encode("utf-8")
    destination = pathlib.Path(output)
    if destination.exists():
        raise ValueError(f"submission directory already exists: {destination}")
    destination.mkdir(parents=True)
    metadata = {
        "schema_version": 2,
        "name": name,
        "submitter": submitter.strip(),
        "families": sorted(set(families)),
        "source": "solver.c",
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "candidate_abi": CANDIDATE_ABI,
    }
    (destination / "solver.c").write_bytes(payload)
    (destination / "submission.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (destination / "README.md").write_text(
        f"# {name}\n\nDescribe the solver and how it was developed.\n",
        encoding="utf-8",
    )
    return metadata


def check(root: str | pathlib.Path) -> dict[str, object]:
    from .build import CANDIDATE_ABI, validate_source

    root = pathlib.Path(root)
    if not root.is_dir() or {path.name for path in root.iterdir()} != {
        "README.md",
        "solver.c",
        "submission.json",
    }:
        raise ValueError(
            "a submission must contain README.md, solver.c, submission.json"
        )
    metadata = json.loads((root / "submission.json").read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "name",
        "submitter",
        "families",
        "source",
        "source_sha256",
        "candidate_abi",
    }
    if not isinstance(metadata, dict) or set(metadata) != expected:
        raise ValueError("submission metadata fields are invalid")
    _validate_details(metadata["name"], metadata["families"], metadata["submitter"])
    if (
        metadata["schema_version"] != 2
        or metadata["source"] != "solver.c"
        or metadata["candidate_abi"] != CANDIDATE_ABI
    ):
        raise ValueError("submission metadata is invalid")
    source = root / "solver.c"
    validate_source(source)
    if hashlib.sha256(source.read_bytes()).hexdigest() != metadata["source_sha256"]:
        raise ValueError("solver.c differs from its recorded hash")
    if not (root / "README.md").read_text(encoding="utf-8").strip():
        raise ValueError("submission README is empty")
    return metadata
