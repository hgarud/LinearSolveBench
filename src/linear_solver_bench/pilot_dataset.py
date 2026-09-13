"""Prepare public pilot cases and stream one trusted numerical system at a time."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import uuid
from collections.abc import Iterable, Iterator

from .archive import decode_system, encode_system
from .dataset import sha256_file
from .manifests import (
    digest,
    load_release,
    seal_manifest,
    strict_object,
    validate_qualification,
    validate_release,
)
from .models import EvaluationSystem
from .paths import cache_dir
from .sources.download import fetch_archive
from .sources.flash import load_flash
from .sources.suitesparse import load_suitesparse
from .workloads import (
    expected_flash_system_digest,
    manufacture_ns,
    matrix_digest,
    qualify_ns,
    system_digest,
    vector_digest,
)

PREPARED_KIND = "linear-solver-bench-pilot-prepared"
IDENTITY_FIELDS = ("release_id", "family", "track", "split", "contracts", "execution")


def validate_pilot_prepared_manifest(value: object, *, official: bool = False) -> dict:
    prepared = strict_object(
        value,
        {
            "schema_version",
            "kind",
            "source_release",
            "source_manifest_sha256",
            "case_count",
            "cases",
            "complete_split",
            "manifest_sha256",
            *IDENTITY_FIELDS,
        },
        "prepared pilot manifest",
    )
    if prepared["schema_version"] != 2 or prepared["kind"] != PREPARED_KIND:
        raise ValueError("unsupported prepared pilot schema")
    release = validate_release(prepared["source_release"], official=official)
    if prepared["source_manifest_sha256"] != release["manifest_sha256"]:
        raise ValueError("prepared release identity mismatch")
    for name in IDENTITY_FIELDS:
        if prepared[name] != release[name]:
            raise ValueError("prepared contract or release metadata mismatch")
    cases = prepared["cases"]
    if (
        not isinstance(cases, list)
        or not cases
        or type(prepared["case_count"]) is not int
        or prepared["case_count"] != len(cases)
    ):
        raise ValueError("prepared pilot case list is invalid")
    previous = -1
    paths = set()
    for index, case in enumerate(cases):
        strict_object(
            case,
            {
                "index",
                "case_id",
                "source_index",
                "relpath",
                "sha256",
                "n",
                "nnz",
                "tolerance",
                "system_sha256",
                "qualification",
            },
            "prepared case",
        )
        source_index = case["source_index"]
        if (
            type(case["index"]) is not int
            or case["index"] != index
            or type(source_index) is not int
            or not previous < source_index < len(release["cases"])
        ):
            raise ValueError("prepared case order or source index mismatch")
        previous = source_index
        source = release["cases"][source_index]
        if any(
            case[key] != source[key] for key in ("case_id", "n", "nnz", "tolerance")
        ):
            raise ValueError("prepared case disagrees with its release")
        name = case["relpath"]
        if (
            not isinstance(name, str)
            or pathlib.PurePosixPath(name).name != name
            or not name.endswith(".npz")
            or "\\" in name
            or name in paths
        ):
            raise ValueError("prepared case path is invalid")
        paths.add(name)
        digest(case["sha256"], "prepared archive digest")
        digest(case["system_sha256"], "prepared numerical digest")
        if release["family"] == "ns-mesh-pde":
            validate_qualification(case["qualification"], case["system_sha256"])
            if (
                source["qualification"] is not None
                and source["qualification"]["system_sha256"] != case["system_sha256"]
            ):
                raise ValueError(
                    "prepared NS system differs from the qualified release"
                )
        else:
            if case["qualification"] is not None:
                raise ValueError(
                    "replay timing qualification belongs in its reference artifact"
                )
            if case["system_sha256"] != expected_flash_system_digest(source):
                raise ValueError(
                    "prepared FLASH inputs differ from the captured release"
                )
    complete = len(cases) == len(release["cases"])
    if (
        type(prepared["complete_split"]) is not bool
        or prepared["complete_split"] != complete
    ):
        raise ValueError("prepared split completeness mismatch")
    if official and not complete:
        raise ValueError(
            "official evaluation requires every case in the declared split"
        )
    body = {key: item for key, item in prepared.items() if key != "manifest_sha256"}
    if seal_manifest(body)["manifest_sha256"] != prepared["manifest_sha256"]:
        raise ValueError("prepared manifest digest mismatch")
    return prepared


def load_pilot_prepared_manifest(root: pathlib.Path, *, official: bool = False) -> dict:
    try:
        value = json.loads(
            (pathlib.Path(root) / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read prepared pilot manifest") from exc
    return validate_pilot_prepared_manifest(value, official=official)


def prepare_release(
    manifest_path: pathlib.Path,
    output: pathlib.Path,
    *,
    rhs_key: bytes | None = None,
    cache: pathlib.Path | None = None,
    offline: bool = False,
    case_ids: Iterable[str] | None = None,
) -> dict:
    release = load_release(manifest_path)
    if release["family"] == "ns-mesh-pde":
        if rhs_key is None:
            public_key = release["rhs"]["public_key_hex"]
            if public_key is None:
                raise ValueError(
                    "ranked preparation requires the operator-held RHS key"
                )
            rhs_key = bytes.fromhex(public_key)
        if (
            not isinstance(rhs_key, bytes)
            or hashlib.sha256(rhs_key).hexdigest() != release["rhs"]["key_id"]
        ):
            raise ValueError("RHS key does not match the declared release")
    elif rhs_key is not None:
        raise ValueError("FLASH replay preserves captured RHS and accepts no RHS key")
    requested = set(case_ids or ())
    cases = [
        case
        for case in release["cases"]
        if not requested or case["case_id"] in requested
    ]
    if requested and requested != {case["case_id"] for case in cases}:
        raise ValueError("requested case ID is absent from the release")
    output = pathlib.Path(output).resolve()
    download_cache = pathlib.Path(cache or cache_dir() / "downloads").resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("prepared output directory must be absent or empty")
    if download_cache == output or output in download_cache.parents:
        raise ValueError("download cache must be outside prepared output")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    staging.mkdir()
    try:
        prepared_cases = []
        for index, case in enumerate(cases):
            archive = fetch_archive(case["source"], download_cache, offline=offline)
            if release["family"] == "ns-mesh-pde":
                matrix = load_suitesparse(archive, case)
                system = manufacture_ns(case, release, rhs_key, matrix)
                qualified = case["qualification"]
                if qualified is None:
                    qualified = qualify_ns(system)
                else:
                    # Qualification belongs to release preparation. Reuse its
                    # frozen evidence after binding it to these exact inputs;
                    # ordinary downloads must not repeat large factorizations or
                    # change prepared identity with platform-dependent metrics.
                    validate_qualification(qualified, system_digest(system))
            else:
                system = load_flash(archive, case)
                qualified = None
            payload = encode_system(system, schema_version=2)
            filename = f"{index:04d}.npz"
            (staging / filename).write_bytes(payload)
            prepared_cases.append(
                {
                    "index": index,
                    "case_id": case["case_id"],
                    "source_index": case["index"],
                    "relpath": filename,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "n": case["n"],
                    "nnz": case["nnz"],
                    "tolerance": case["tolerance"],
                    "system_sha256": system_digest(system),
                    "qualification": qualified,
                }
            )
            # Keep neither decoded systems nor matrix arrays across iterations.
            del payload, system
            if release["family"] == "ns-mesh-pde":
                del matrix
        body = {
            "schema_version": 2,
            "kind": PREPARED_KIND,
            **{key: release[key] for key in IDENTITY_FIELDS},
            "source_release": release,
            "source_manifest_sha256": release["manifest_sha256"],
            "case_count": len(prepared_cases),
            "cases": prepared_cases,
            "complete_split": len(prepared_cases) == len(release["cases"]),
        }
        manifest = validate_pilot_prepared_manifest(seal_manifest(body))
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if output.exists():
            output.rmdir()
        os.replace(staging, output)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def iter_pilot_prepared(
    root: pathlib.Path, *, official: bool = False
) -> Iterator[EvaluationSystem]:
    """Validate and yield each trusted archive without retaining the prior case."""
    root = pathlib.Path(root).resolve()
    manifest = load_pilot_prepared_manifest(root, official=official)
    for case in manifest["cases"]:
        source = manifest["source_release"]["cases"][case["source_index"]]["source"]
        path = root / case["relpath"]
        if (
            path.is_symlink()
            or not path.is_file()
            or sha256_file(path) != case["sha256"]
        ):
            raise ValueError("prepared archive is absent or has the wrong digest")
        system = decode_system(path.read_bytes())
        if (
            system.case_id != case["case_id"]
            or system.public.matrix.n != case["n"]
            or system.public.matrix.nnz != case["nnz"]
            or system.public.tolerance != case["tolerance"]
            or system_digest(system) != case["system_sha256"]
            or matrix_digest(system.public.matrix) != source["matrix_sha256"]
        ):
            raise ValueError("prepared numerical inputs disagree with the manifest")
        if (
            manifest["family"] == "ns-mesh-pde"
            and system.reference_kind != "manufactured"
        ):
            raise ValueError("NS coverage requires the declared manufactured target")
        if manifest["family"] == "magnetic_diffusion_flash":
            if system.reference_kind != "none":
                raise ValueError("pilot FLASH captures cannot add a reference array")
            if (
                vector_digest(system.public.b) != source["b_sha256"]
                or vector_digest(system.public.x0) != source["x0_sha256"]
            ):
                raise ValueError(
                    "prepared FLASH vectors differ from the captured release"
                )
        yield system
        del system
