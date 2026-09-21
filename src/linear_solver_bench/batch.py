"""Authenticated numeric-only bundles and a sandbox-local execution loop."""

from __future__ import annotations

import hashlib
import io
import json
import os
import pathlib
import sys
import tempfile
import zipfile
from dataclasses import dataclass

from .archives import read_zip_member, zip_members
from .build import _run_encoded_solver, encode_input
from .integrity import file_sha256
from .models import BenchmarkCase, RawExecution
from .protocol import output_size

RECORD_BYTES = 40_000


@dataclass(frozen=True)
class InputBundle:
    path: pathlib.Path
    entries: list[dict]
    sha256: str
    cache_hit: bool


def prepare_inputs(cases: list[BenchmarkCase], cache: pathlib.Path) -> InputBundle:
    """Cache one cohort archive, authenticated against freshly validated inputs.

    Only native input bytes enter the archive. Manufactured targets and other
    verification data stay in the caller's BenchmarkCase objects.
    """
    if not cases or any(not case.input_sha256 for case in cases):
        raise ValueError("input bundles require authenticated benchmark cases")
    key = hashlib.sha256(
        json.dumps(
            ["native-input-bundle-v1"]
            + [(case.case_id, case.input_sha256) for case in cases]
        ).encode()
    ).hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{key}.zip"
    cache_hit = path.is_file() and not path.is_symlink()
    entries = []
    temporary = None
    try:
        if cache_hit:
            archive = zipfile.ZipFile(path)
        else:
            with tempfile.NamedTemporaryFile(dir=cache, delete=False) as handle:
                temporary = pathlib.Path(handle.name)
            archive = zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED)
        with archive:
            expected_names = [f"inputs/{index:06d}.bin" for index in range(len(cases))]
            if cache_hit and archive.namelist() != expected_names:
                raise ValueError("cached input bundle has unexpected members")
            for case, name in zip(cases, expected_names, strict=True):
                payload = encode_input(case.input)
                digest = hashlib.sha256(payload).hexdigest()
                if cache_hit:
                    if archive.getinfo(name).file_size != len(payload) or (
                        hashlib.sha256(archive.read(name)).hexdigest() != digest
                    ):
                        raise ValueError("cached input bundle differs from its cases")
                else:
                    member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    member.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(member, payload, compresslevel=1)
                entries.append(
                    {
                        "name": name,
                        "case_id": case.case_id,
                        "n": case.input.matrix.n,
                        "size": len(payload),
                        "sha256": digest,
                    }
                )
        if temporary is not None:
            os.replace(temporary, path)
        return InputBundle(path, entries, file_sha256(path), cache_hit)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"cannot prepare input bundle: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def output_name(index: int, role: str) -> str:
    return f"outputs/{index:06d}-{role}.bin"


def result_limit(entries: list[dict], roles: list[str]) -> int:
    """Bound the archive including outputs, JSON records, and ZIP overhead."""
    return 4096 + sum(
        len(roles) * (output_size(entry["n"]) + 1 + RECORD_BYTES + 512)
        for entry in entries
    )


def read_results(
    payload: bytes, entries: list[dict], roles: list[str]
) -> list[dict[str, RawExecution]]:
    """Decode untrusted transport without extracting any remote paths."""
    if len(payload) > result_limit(entries, roles):
        raise ValueError("batch results exceed the transport limit")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = zip_members(archive)
        records = json.loads(
            read_zip_member(
                archive, "results.json", len(entries) * len(roles) * RECORD_BYTES
            )
        )
        if not isinstance(records, list) or len(records) != len(entries):
            raise ValueError("batch results do not cover the requested cases")
        expected_names = {"results.json"}
        results = []
        for index, (entry, record) in enumerate(zip(entries, records, strict=True)):
            if not isinstance(record, dict) or set(record) != set(roles):
                raise ValueError("batch result solver roles differ from the request")
            row = {}
            for role in roles:
                value = record[role]
                if not isinstance(value, dict) or set(value) != {
                    "returncode",
                    "wall_seconds",
                    "diagnostics",
                    "output",
                }:
                    raise ValueError("batch execution record has unexpected fields")
                output = None
                if value["output"] is not None:
                    name = output_name(index, role)
                    if value["output"] != name:
                        raise ValueError("batch output belongs to another execution")
                    output = read_zip_member(archive, name, output_size(entry["n"]) + 1)
                    expected_names.add(name)
                row[role] = RawExecution(
                    value["returncode"],
                    value["wall_seconds"],
                    value["diagnostics"],
                    output,
                )
            results.append(row)
        if names != expected_names:
            raise ValueError("unexpected batch result members")
    return results


def run_batch(
    archive_path: pathlib.Path, plan: dict, destination: pathlib.Path
) -> None:
    """Run every solver in a fresh process; return raw evidence, never scores."""
    if file_sha256(archive_path) != plan["archive_sha256"]:
        raise ValueError("input bundle digest mismatch")
    records = []
    with (
        zipfile.ZipFile(archive_path) as inputs,
        zipfile.ZipFile(
            destination, "w", zipfile.ZIP_DEFLATED, compresslevel=1
        ) as outputs,
    ):
        if inputs.namelist() != [entry["name"] for entry in plan["entries"]]:
            raise ValueError("input bundle has unexpected members")
        for index, entry in enumerate(plan["entries"]):
            member = inputs.getinfo(entry["name"])
            if member.file_size != entry["size"]:
                raise ValueError("input bundle member size mismatch")
            payload = inputs.read(member)
            if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                raise ValueError("input bundle member digest mismatch")
            print(
                f"Running {entry['case_id']} ({index + 1}/{len(plan['entries'])})",
                flush=True,
            )
            row = {}
            for role, executable in plan["executables"].items():
                # The helper writes pristine input into a new temporary directory
                # for each process, including each reference/candidate pair.
                raw = _run_encoded_solver(pathlib.Path(executable), payload, entry["n"])
                if role == "reference" and raw.returncode == 124:
                    raise RuntimeError(
                        "fixed reference timed out; evaluation incomplete"
                    )
                name = None
                if raw.output is not None:
                    name = output_name(index, role)
                    outputs.writestr(name, raw.output)
                row[role] = {
                    "returncode": raw.returncode,
                    "wall_seconds": raw.wall_seconds,
                    "diagnostics": raw.diagnostics,
                    "output": name,
                }
            records.append(row)
        outputs.writestr("results.json", json.dumps(records, ensure_ascii=False))


if __name__ == "__main__":
    run_batch(
        pathlib.Path(sys.argv[1]),
        json.loads(pathlib.Path(sys.argv[2]).read_text()),
        pathlib.Path(sys.argv[3]),
    )
