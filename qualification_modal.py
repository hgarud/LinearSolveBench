"""Run the uniform 1-norm condition campaign on locked SuiteSparse inputs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import subprocess
import urllib.request
import uuid
from collections.abc import Mapping

import modal

ROOT = pathlib.Path(__file__).resolve().parent
APP_NAME = "linear-solver-bench-condition-v2"
BASE_IMAGE = (
    "debian:trixie-slim@sha256:"
    "abc9cb88a5587630d7f915f47b23b0668fe250fbfc6457aa4d52b534c1bbf73f"
)
APT_PACKAGES = (
    "build-essential=12.12",
    "libsuitesparse-dev=1:7.10.1+dfsg-1",
    "pkg-config=1.8.1-4",
)
CACHE_APT_PACKAGES = ("ca-certificates=20250419",)
PYTHON_PACKAGES = (
    "numpy==2.5.1",
    "scipy==1.18.0",
    "scikit-sparse==0.5.0",
)
REMOTE_ROOT = "/opt/LinearSolverBench"
REMOTE_SOURCE = "/opt/LinearSolverBench/src"
REMOTE_CACHE = "/condition-cache"
CACHE_VOLUME_NAME = "linear-solver-bench-qualification-cache-v1"
SMALL_FILL_MAX = 1_000_000
MEDIUM_FILL_MAX = 10_000_000
MAX_ARCHIVE_BYTES = 8 * 1024**3

app = modal.App(APP_NAME)
archive_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.from_registry(BASE_IMAGE, add_python="3.12")
    .apt_install(*APT_PACKAGES)
    .env(
        {
            "CC": "gcc",
            "CXX": "g++",
            "PYTHONPATH": f"{REMOTE_ROOT}:{REMOTE_SOURCE}",
        }
    )
    .pip_install(*PYTHON_PACKAGES)
    .add_local_dir(ROOT / "src", REMOTE_SOURCE, copy=True)
    .add_local_file(
        ROOT / "qualification_modal.py",
        f"{REMOTE_ROOT}/qualification_modal.py",
        copy=True,
    )
)
cache_image = (
    modal.Image.from_registry(BASE_IMAGE, add_python="3.12")
    .apt_install(*CACHE_APT_PACKAGES)
    .env({"PYTHONPATH": REMOTE_ROOT})
    .add_local_file(
        ROOT / "qualification_modal.py",
        f"{REMOTE_ROOT}/qualification_modal.py",
        copy=True,
    )
)


def _remote_archive(case: Mapping[str, object]) -> pathlib.Path:
    return (
        pathlib.Path(REMOTE_CACHE)
        / str(case["archive_sha256"])
        / str(case["archive_filename"])
    )


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@app.function(
    image=cache_image,
    volumes={REMOTE_CACHE: archive_volume},
    cpu=0.25,
    memory=512,
    timeout=1800,
    max_containers=16,
    serialized=True,
)
def cache_archive(job: Mapping[str, object]) -> int:
    """Populate one hash-addressed archive without changing the worker image."""
    case = job["case"]
    if not isinstance(case, Mapping):
        raise TypeError("condition cache job case is invalid")
    expected = str(case["archive_sha256"])
    target = _remote_archive(case)
    if target.is_file() and _sha256(target) == expected:
        return int(case["index"])
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    request = urllib.request.Request(
        str(case["source_url"]),
        headers={"User-Agent": "linear-solver-bench-qualification/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) > MAX_ARCHIVE_BYTES:
                raise ValueError("qualification archive exceeds the size limit")
            digest = hashlib.sha256()
            total = 0
            with temporary.open("wb") as handle:
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > MAX_ARCHIVE_BYTES:
                        raise ValueError("qualification archive exceeds the size limit")
                    digest.update(block)
                    handle.write(block)
        if digest.hexdigest() != expected:
            raise ValueError("qualification archive digest mismatch")
        os.replace(temporary, target)
        archive_volume.commit()
    finally:
        temporary.unlink(missing_ok=True)
    return int(case["index"])


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _command_output(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return result.stdout.strip() or result.stderr.strip() or "unavailable"


def _toolchain() -> dict[str, object]:
    import numpy
    import scipy

    import linear_solver_bench.qualification as qualification

    source = pathlib.Path(qualification.__file__)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "scikit_sparse": _package_version("scikit-sparse"),
        "suite_sparse_debian": _command_output(
            ["dpkg-query", "--show", "--showformat=${Version}", "libsuitesparse-dev"]
        ),
        "compiler": _command_output(["gcc", "--version"]).splitlines()[0],
        "image_contract": {
            "base": BASE_IMAGE,
            "apt_packages": list(APT_PACKAGES),
            "python_packages": list(PYTHON_PACKAGES),
        },
        "source_sha256": {
            "qualification.py": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
    }


def _estimate(job: Mapping[str, object]) -> dict[str, object]:
    from linear_solver_bench.dataset import (
        load_matrix_market_archive,
        matrix_sha256,
        sha256_file,
    )
    from linear_solver_bench.models import CsrMatrix
    from linear_solver_bench.qualification import (
        indeterminate_condition,
        qualify_matrix,
    )

    case = job["case"]
    if not isinstance(case, Mapping):
        raise TypeError("condition job case is invalid")
    archive_volume.reload()
    archive = _remote_archive(case)
    if sha256_file(archive) != case["archive_sha256"]:
        raise ValueError("condition source archive digest mismatch")
    matrix = load_matrix_market_archive(archive, case)
    canonical_sha256 = matrix_sha256(CsrMatrix.from_scipy(matrix))
    try:
        condition, inverse = qualify_matrix(matrix, case_id=str(case["case_id"]))
    except Exception as exc:  # noqa: BLE001 - preserve per-case failure evidence
        condition = indeterminate_condition(f"{type(exc).__name__}: {exc}")
        inverse = None
    return {
        **{
            key: case[key]
            for key in (
                "index",
                "matrix_id",
                "case_id",
                "group",
                "name",
                "n",
                "nnz",
                "archive_sha256",
            )
        },
        "matrix_sha256": canonical_sha256,
        "inverse_norm_evidence": inverse.as_record() if inverse else None,
        "condition": condition,
    }


@app.function(
    image=image,
    volumes={REMOTE_CACHE: archive_volume},
    cpu=4.0,
    memory=8192,
    timeout=1800,
    max_containers=8,
    block_network=True,
    serialized=True,
)
def estimate_small(job: Mapping[str, object]) -> dict[str, object]:
    return {"result": _estimate(job), "toolchain": _toolchain()}


@app.function(
    image=image,
    volumes={REMOTE_CACHE: archive_volume},
    cpu=8.0,
    memory=32768,
    timeout=3600,
    max_containers=8,
    block_network=True,
    serialized=True,
)
def estimate_medium(job: Mapping[str, object]) -> dict[str, object]:
    return {"result": _estimate(job), "toolchain": _toolchain()}


@app.function(
    image=image,
    volumes={REMOTE_CACHE: archive_volume},
    cpu=16.0,
    memory=65536,
    timeout=7200,
    max_containers=8,
    block_network=True,
    serialized=True,
)
def estimate_large(job: Mapping[str, object]) -> dict[str, object]:
    return {"result": _estimate(job), "toolchain": _toolchain()}


TIER_FUNCTIONS = {
    "small": estimate_small,
    "medium": estimate_medium,
    "large": estimate_large,
}


def _load_inputs(path: pathlib.Path) -> dict[str, object]:
    from linear_solver_bench.dataset import identity_sha256

    value = json.loads(path.read_text(encoding="utf-8"))
    digest = value.get("manifest_sha256")
    body = {key: item for key, item in value.items() if key != "manifest_sha256"}
    if (
        value.get("schema_version") != 1
        or value.get("kind") != "linear-solver-bench-condition-inputs-v1"
        or digest != identity_sha256(body)
        or value.get("case_count") != len(value.get("cases", ()))
    ):
        raise ValueError("qualification input lock is invalid")
    return value


def _tier(record: Mapping[str, object]) -> str:
    fill = int(record["amd_vnz"]) + int(record["amd_rnz"])
    if fill <= SMALL_FILL_MAX:
        return "small"
    if fill <= MEDIUM_FILL_MAX:
        return "medium"
    return "large"


def _write(path: pathlib.Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@app.local_entrypoint()
def main(
    output: str = "data/qualification-1norm-v2.json",
    inputs: str = "data/qualification-inputs-v1.json",
    case_limit: int = 0,
) -> None:
    from linear_solver_bench.qualification import qualification_document

    if case_limit < 0:
        raise ValueError("case_limit cannot be negative")
    input_path = pathlib.Path(inputs).expanduser().resolve()
    output_path = pathlib.Path(output).expanduser().resolve()
    locked = _load_inputs(input_path)
    catalogue = json.loads((ROOT / "data/catalogue.json").read_text())
    records = catalogue["records"]
    if len(records) != locked["case_count"]:
        raise ValueError("qualification inputs disagree with the catalogue")
    completed = []
    expected_toolchain = None
    if output_path.is_file():
        checkpoint = json.loads(output_path.read_text())
        completed = list(checkpoint["results"])
        expected_toolchain = checkpoint["toolchain"]
    completed_indices = {int(result["index"]) for result in completed}
    jobs = {tier: [] for tier in TIER_FUNCTIONS}
    for case, record in zip(locked["cases"], records, strict=True):
        if case["index"] in completed_indices:
            continue
        jobs[_tier(record)].append({"case": case})
    if case_limit:
        remaining = case_limit
        for tier in jobs:
            jobs[tier] = jobs[tier][:remaining]
            remaining -= len(jobs[tier])
    print(
        json.dumps(
            {
                "pending": {tier: len(items) for tier, items in jobs.items()},
                "completed": len(completed),
                "output": str(output_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    cache_jobs = [job for tier in jobs.values() for job in tier]
    for cached in cache_archive.map(
        cache_jobs, order_outputs=False, return_exceptions=False
    ):
        print(f"cached qualification input {cached}", flush=True)
    for tier, pending in jobs.items():
        if not pending:
            continue
        for packet in TIER_FUNCTIONS[tier].map(
            pending, order_outputs=False, return_exceptions=False
        ):
            toolchain = packet["toolchain"]
            if expected_toolchain is None:
                expected_toolchain = toolchain
            elif toolchain != expected_toolchain:
                raise RuntimeError("qualification worker toolchains differ")
            result = packet["result"]
            index = int(result["index"])
            if index in completed_indices:
                raise RuntimeError("qualification worker returned a duplicate case")
            completed.append(result)
            completed_indices.add(index)
            _write(
                output_path,
                qualification_document(
                    catalogue_sha256=str(locked["catalogue_sha256"]),
                    input_manifest_sha256=str(locked["manifest_sha256"]),
                    expected_case_count=int(locked["case_count"]),
                    results=completed,
                    toolchain=expected_toolchain,
                ),
            )
            print(f"qualified {len(completed)}/{locked['case_count']}", flush=True)
    if expected_toolchain is None:
        raise RuntimeError("qualification campaign produced no toolchain evidence")
    document = qualification_document(
        catalogue_sha256=str(locked["catalogue_sha256"]),
        input_manifest_sha256=str(locked["manifest_sha256"]),
        expected_case_count=int(locked["case_count"]),
        results=completed,
        toolchain=expected_toolchain,
    )
    _write(output_path, document)
    print(
        json.dumps(
            {
                "complete": document["complete"],
                "condition_results_sha256": document["condition_results_sha256"],
                "status_counts": document["status_counts"],
                "method_counts": document["method_counts"],
                "output": str(output_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
