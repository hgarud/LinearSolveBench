"""Create and check the small pull-request submission bundle."""

from __future__ import annotations

import hashlib
import json
import pathlib
import re


def _safe_name(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9_-]{1,63}", value
    ):
        raise ValueError("name must use 2-64 lowercase letters, digits, '-' or '_'")
    return value


def package(
    source: str | pathlib.Path,
    output: str | pathlib.Path,
    *,
    name: str,
    families: list[str],
    license_name: str,
    submitter: str,
) -> dict[str, object]:
    from .build import CANDIDATE_ABI, validate_source

    name = _safe_name(name)
    if (
        not families
        or not all(isinstance(family, str) for family in families)
        or set(families) - {"ns_mesh_pde", "flash"}
    ):
        raise ValueError("select ns_mesh_pde, flash, or both")
    if not isinstance(submitter, str) or not submitter.strip():
        raise ValueError("submitter must be nonempty text")
    if not isinstance(license_name, str) or not license_name.strip():
        raise ValueError("license must be nonempty text")
    validate_source(pathlib.Path(source))
    payload = pathlib.Path(source).read_bytes()
    destination = pathlib.Path(output)
    if destination.exists():
        raise ValueError(f"submission directory already exists: {destination}")
    destination.mkdir(parents=True)
    metadata = {
        "schema_version": 1,
        "name": name,
        "submitter": submitter.strip(),
        "license": license_name.strip(),
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
    check(destination)
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
        "license",
        "families",
        "source",
        "source_sha256",
        "candidate_abi",
    }
    if not isinstance(metadata, dict) or set(metadata) != expected:
        raise ValueError("submission metadata fields are invalid")
    _safe_name(metadata["name"])
    families = metadata["families"]
    if (
        metadata["schema_version"] != 1
        or metadata["source"] != "solver.c"
        or metadata["candidate_abi"] != CANDIDATE_ABI
        or not isinstance(metadata["submitter"], str)
        or not metadata["submitter"].strip()
        or not isinstance(metadata["license"], str)
        or not metadata["license"].strip()
        or not isinstance(families, list)
        or not families
        or not all(isinstance(family, str) for family in families)
        or set(families) - {"ns_mesh_pde", "flash"}
    ):
        raise ValueError("submission metadata is invalid")
    source = root / "solver.c"
    validate_source(source)
    if hashlib.sha256(source.read_bytes()).hexdigest() != metadata["source_sha256"]:
        raise ValueError("solver.c differs from its recorded hash")
    if not (root / "README.md").read_text(encoding="utf-8").strip():
        raise ValueError("submission README is empty")
    return metadata
