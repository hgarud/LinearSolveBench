"""Run benchmark solvers in one Modal CPU sandbox.

This module is only a transport.  It builds programs and returns raw process
output; correctness checks and family-specific scoring belong to ``benchmark``.
"""

from __future__ import annotations

import json
import pathlib
import time
import zipfile
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import modal

from .assets import asset_root
from .batch import InputBundle, read_results, result_limit
from .build import RUN_TIMEOUT_SECONDS
from .models import RawExecution
from .protocol import DIAGNOSTIC_LIMIT

APP_NAME = "linear-solver-bench"
VENUE_ID = "modal-cpu-v1"
BASE_IMAGE = (
    "debian:trixie-slim@sha256:"
    "abc9cb88a5587630d7f915f47b23b0668fe250fbfc6457aa4d52b534c1bbf73f"
)
REMOTE_PROJECT = "/opt/linear-solver-bench"
REMOTE_RUNTIME = "/opt/lsb-runtime"
SANDBOX_LIFETIME_SECONDS = 24 * 60 * 60
BUILD_TIMEOUT_SECONDS = 300


def create_image():
    """Describe the immutable image used by benchmark runs."""
    project = asset_root("native")
    package = pathlib.Path(__file__).resolve().parent
    return (
        modal.Image.from_registry(BASE_IMAGE, add_python="3.12")
        .apt_install("build-essential", "cmake", "git", "ninja-build")
        .pip_install("numpy==2.5.1", "scipy==1.18.0")
        .add_local_dir(
            package,
            f"{REMOTE_PROJECT}/src/linear_solver_bench",
            copy=True,
            ignore=["**/__pycache__/**", "**/*.pyc"],
        )
        .add_local_dir(project / "native", f"{REMOTE_PROJECT}/native", copy=True)
        .env(
            {
                "LINEAR_SOLVER_BENCH_ROOT": REMOTE_PROJECT,
                "PYTHONPATH": f"{REMOTE_PROJECT}/src",
            }
        )
        .run_commands(
            'python -c "from pathlib import Path; '
            "from linear_solver_bench.build import build_runtime; "
            f"build_runtime(Path('{REMOTE_RUNTIME}'), jobs=2)\""
        )
    )


def _stream_tail(stream, limit: int = DIAGNOSTIC_LIMIT) -> str:
    tail = ""
    for chunk in stream:
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8", errors="replace")
        tail = (tail + chunk)[-limit:]
    return tail


def _drain(process, *, sandbox, read_stdout=_stream_tail):
    """Drain stdout and stderr together so neither pipe can block the other."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        stdout = pool.submit(read_stdout, process.stdout)
        stderr = pool.submit(_stream_tail, process.stderr)
        try:
            return stdout.result(), stderr.result()
        except BaseException:
            sandbox.terminate()
            raise


def _read_output(sandbox, path: str, maximum_bytes: int) -> bytes | None:
    process = sandbox.exec("head", "-c", str(maximum_bytes), path, text=False)

    def collect(stream) -> bytes:
        chunks: list[bytes] = []
        size = 0
        for chunk in stream:
            size += len(chunk)
            if size > maximum_bytes:
                raise RuntimeError("output transport exceeded its byte limit")
            chunks.append(chunk)
        return b"".join(chunks)

    payload, _ = _drain(process, sandbox=sandbox, read_stdout=collect)
    returncode = process.wait()
    if returncode in {124, -1}:
        raise RuntimeError(
            "Modal interrupted output transport; evaluation is incomplete"
        )
    return payload if returncode == 0 else None


class ModalSession:
    """One evaluation session inside one isolated sandbox."""

    def __init__(
        self,
        sandbox,
        candidate_source: pathlib.Path,
        reference_source: pathlib.Path | None,
        execution: Mapping,
    ):
        self.sandbox = sandbox
        self.sources = {
            role: pathlib.Path(source)
            for role, source in {
                "reference": reference_source,
                "candidate": candidate_source,
            }.items()
            if source is not None
        }
        self.execution = dict(execution)
        self._build: dict | None = None
        self._executables: dict[str, str] = {}

    def build(self) -> dict:
        """Compile each solver once and return the identities used by reports."""
        if self._build is not None:
            return dict(self._build)
        if self.sandbox.exec("mkdir", "-p", "/work").wait() != 0:
            raise RuntimeError("cannot initialize sandbox working directory")
        for role, source in self.sources.items():
            self.sandbox.filesystem.copy_from_local(source, f"/work/{role}.c")
        reference_args = (
            ("--reference", "/work/reference.c") if "reference" in self.sources else ()
        )
        process = self.sandbox.exec(
            "python",
            "-m",
            "linear_solver_bench.build",
            REMOTE_RUNTIME,
            "/work/candidate.c",
            "/work",
            *reference_args,
            text=False,
            timeout=BUILD_TIMEOUT_SECONDS,
        )
        stdout, stderr = _drain(process, sandbox=self.sandbox)
        if process.wait() != 0:
            detail = (stdout + stderr)[-DIAGNOSTIC_LIMIT:] or "no diagnostics"
            raise RuntimeError(f"sandbox solver build failed: {detail}")
        metadata = json.loads(self.sandbox.filesystem.read_text("/work/build.json"))
        self._executables = metadata["executables"]
        self._build = {
            "venue": {
                "id": VENUE_ID,
                "image_id": metadata["image_id"],
                "base_image": BASE_IMAGE,
                "cpus": self.execution["cpus"],
                "memory_bytes": self.execution["memory_bytes"],
                "limits_enforced": True,
            },
            **metadata["identities"],
        }
        return dict(self._build)

    def run_cases(
        self, bundle: InputBundle, progress
    ) -> tuple[list[dict[str, RawExecution]], dict]:
        """Upload one numeric bundle, execute remotely, collect one result bundle."""
        if not self._build:
            raise RuntimeError("build the session before running cases")
        roles = list(self._executables)
        plan = {
            "archive_sha256": bundle.sha256,
            "entries": bundle.entries,
            "executables": self._executables,
        }
        size = bundle.path.stat().st_size
        progress(
            f"Uploading {len(bundle.entries)} inputs "
            f"in one {size / 2**20:.1f} MiB bundle"
        )
        started = time.monotonic()
        self.sandbox.filesystem.copy_from_local(bundle.path, "/work/inputs.zip")
        self.sandbox.filesystem.write_text(json.dumps(plan), "/work/plan.json")
        statistics = {
            "input_uploads": 1,
            "input_bundle_bytes": size,
            "input_bundle_cache_hit": bundle.cache_hit,
            "input_upload_seconds": time.monotonic() - started,
        }
        progress("Running the staged inputs in the Modal sandbox")
        started = time.monotonic()
        process = self.sandbox.exec(
            "python",
            "-m",
            "linear_solver_bench.batch",
            "/work/inputs.zip",
            "/work/plan.json",
            "/work/results.zip",
            text=False,
            timeout=min(
                SANDBOX_LIFETIME_SECONDS - BUILD_TIMEOUT_SECONDS,
                int(RUN_TIMEOUT_SECONDS * len(bundle.entries) * len(roles)) + 300,
            ),
        )

        def read_progress(stream):
            pending = ""
            for chunk in stream:
                pending += (
                    chunk.decode("utf-8", errors="replace")
                    if isinstance(chunk, bytes)
                    else chunk
                )
                while "\n" in pending:
                    line, pending = pending.split("\n", 1)
                    if len(line) > 2048:
                        raise RuntimeError("oversized batch progress message")
                    progress(line)
                if len(pending) > 2048:
                    raise RuntimeError("oversized batch progress message")
            if pending:
                progress(pending)

        _, diagnostics = _drain(
            process, sandbox=self.sandbox, read_stdout=read_progress
        )
        if process.wait() != 0:
            raise RuntimeError(
                "Modal batch did not complete: "
                + (diagnostics or "interrupted execution")
            )
        statistics["remote_execution_seconds"] = time.monotonic() - started
        progress("Collecting one result bundle for independent verification")
        started = time.monotonic()
        maximum_bytes = result_limit(bundle.entries, roles)
        payload = _read_output(self.sandbox, "/work/results.zip", maximum_bytes + 1)
        if payload is None:
            raise RuntimeError("Modal batch did not produce its result bundle")
        statistics["output_download_seconds"] = time.monotonic() - started
        statistics["output_bundle_bytes"] = len(payload)
        try:
            results = read_results(payload, bundle.entries, roles)
        except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile) as exc:
            raise RuntimeError(f"invalid Modal batch results: {exc}") from exc
        return results, statistics


@contextmanager
def open_session(
    candidate_source: pathlib.Path,
    reference_source: pathlib.Path | None,
    execution: Mapping,
):
    """Create one network-disabled sandbox for an evaluation."""
    cpus = execution.get("cpus")
    memory_bytes = execution.get("memory_bytes")
    if isinstance(cpus, bool) or not isinstance(cpus, (int, float)) or cpus <= 0:
        raise ValueError("cpus must be a positive number")
    if isinstance(memory_bytes, bool) or not isinstance(memory_bytes, int):
        raise ValueError("memory_bytes must be a positive whole number of MiB")
    memory_mib, remainder = divmod(memory_bytes, 1024**2)
    if remainder or memory_mib < 1:
        raise ValueError("memory_bytes must be a positive whole number of MiB")

    try:
        sandbox = modal.Sandbox.create(
            app=modal.App.lookup(APP_NAME, create_if_missing=True),
            image=create_image(),
            cpu=(float(cpus), float(cpus)),
            memory=(memory_mib, memory_mib),
            timeout=SANDBOX_LIFETIME_SECONDS,
            block_network=True,
        )
    except Exception as exc:
        raise RuntimeError(f"cannot start Modal sandbox: {exc}") from exc
    try:
        yield ModalSession(sandbox, candidate_source, reference_source, execution)
    finally:
        sandbox.terminate()
