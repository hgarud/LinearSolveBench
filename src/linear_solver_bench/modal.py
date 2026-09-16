"""Run a candidate and the fixed reference in one Modal CPU sandbox.

This module is only a transport.  It builds programs and returns raw process
output; correctness checks and speedup calculations belong to ``benchmark``.
Importing it does not import Modal or require Modal credentials.
"""

from __future__ import annotations

import importlib
import json
import pathlib
import time
import zlib
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from .assets import asset_root
from .build import OUTPUT_HEADER, REFERENCE_ID, RUN_TIMEOUT_SECONDS, encode_input
from .models import CaseInput, RawExecution

APP_NAME = "linear-solver-bench"
VENUE_ID = "modal-cpu-v1"
BASE_IMAGE = (
    "debian:trixie-slim@sha256:"
    "abc9cb88a5587630d7f915f47b23b0668fe250fbfc6457aa4d52b534c1bbf73f"
)
REMOTE_PROJECT = "/opt/linear-solver-bench"
REMOTE_RUNTIME = "/opt/lsb-runtime"
SANDBOX_LIFETIME_SECONDS = 24 * 60 * 60
DIAGNOSTIC_LIMIT = 8_000
BUILD_TIMEOUT_SECONDS = 300


def require_modal():
    """Load the optional SDK only when a Modal run is requested."""
    try:
        return importlib.import_module("modal")
    except ModuleNotFoundError as exc:
        if exc.name != "modal":
            raise
        raise RuntimeError(
            "Modal evaluation requires: pip install 'linear-solver-bench[modal]'"
        ) from exc


def create_image():
    """Describe the immutable image shared by reference and candidate runs."""
    modal = require_modal()
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


_BUILD_EXEC = r"""
import json
import os
import pathlib
import sys

from linear_solver_bench.build import build_reference, build_solver, load_runtime

runtime = load_runtime(pathlib.Path(sys.argv[1]))
candidate = build_solver(pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[4]), runtime)
reference = build_reference(
    pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[5]), runtime
)
result = {
    "candidate_source_sha256": candidate.source_sha256,
    "candidate_executable_sha256": candidate.executable_sha256,
    "reference_source_sha256": reference.source_sha256,
    "reference_executable_sha256": reference.executable_sha256,
    "runtime_manifest_sha256": runtime.manifest_sha256,
    "image_id": os.environ.get("MODAL_IMAGE_ID"),
}
pathlib.Path(sys.argv[6]).write_text(json.dumps(result), encoding="utf-8")
"""

_INPUT_ERROR = "linear-solver-bench input preparation failed:"
_INPUT_EXEC = r"""
import os
import sys
import zlib

try:
    remaining = int(sys.argv[1])
    decoder = zlib.decompressobj()
    with open(sys.argv[2], "rb") as source, open(sys.argv[4], "wb") as target:
        while chunk := source.read(1024 * 1024):
            while chunk:
                decoded = decoder.decompress(chunk, min(1024 * 1024, remaining + 1))
                if len(decoded) > remaining or decoder.unused_data:
                    raise ValueError("compressed input exceeds its declared size")
                target.write(decoded)
                remaining -= len(decoded)
                chunk = decoder.unconsumed_tail
        if remaining or not decoder.eof:
            raise ValueError("compressed input is incomplete")
    os.unlink(sys.argv[2])
    os.execv(sys.argv[3], sys.argv[3:])
except (OSError, ValueError, zlib.error) as error:
    print("linear-solver-bench input preparation failed:", error, file=sys.stderr)
    sys.exit(125)
"""


def _stream_tail(stream, limit: int = DIAGNOSTIC_LIMIT) -> str:
    tail = ""
    for chunk in stream:
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8", errors="replace")
        tail = (tail + chunk)[-limit:]
    return tail


def _diagnostics(process, *, sandbox) -> tuple[str, str]:
    """Drain stdout and stderr together so neither pipe can block the other."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        stdout = pool.submit(_stream_tail, process.stdout)
        stderr = pool.submit(_stream_tail, process.stderr)
        try:
            return stdout.result(), stderr.result()
        except BaseException:
            sandbox.terminate()
            raise


def _read_output(sandbox, path: str, maximum_bytes: int) -> bytes | None:
    process = sandbox.exec("head", "-c", str(maximum_bytes), path, text=False)

    def collect() -> bytes:
        chunks: list[bytes] = []
        size = 0
        for chunk in process.stdout:
            size += len(chunk)
            if size > maximum_bytes:
                raise RuntimeError("output transport exceeded its byte limit")
            chunks.append(chunk)
        return b"".join(chunks)

    with ThreadPoolExecutor(max_workers=2) as pool:
        stdout = pool.submit(collect)
        stderr = pool.submit(_stream_tail, process.stderr)
        try:
            payload = stdout.result()
            stderr.result()
        except BaseException:
            sandbox.terminate()
            raise
    returncode = process.wait()
    if returncode in {124, -1}:
        raise RuntimeError(
            "Modal interrupted output transport; evaluation is incomplete"
        )
    return payload if returncode == 0 else None


class ModalSession:
    """One paired evaluation session inside one isolated sandbox."""

    def __init__(
        self,
        sandbox,
        candidate_source: pathlib.Path,
        reference_source: pathlib.Path,
        execution: Mapping,
    ):
        self.sandbox = sandbox
        self.candidate_source = pathlib.Path(candidate_source)
        self.reference_source = pathlib.Path(reference_source)
        self.execution = dict(execution)
        self._build: dict | None = None
        self._executables: dict[str, str] = {}
        self._case_number = 0

    def build(self) -> dict:
        """Compile each solver once and return the identities used by reports."""
        if self._build is not None:
            return dict(self._build)
        if self.sandbox.exec("mkdir", "-p", "/work").wait() != 0:
            raise RuntimeError("cannot initialize sandbox working directory")
        self.sandbox.filesystem.copy_from_local(
            self.candidate_source, "/work/candidate.c"
        )
        self.sandbox.filesystem.copy_from_local(
            self.reference_source, "/work/reference.c"
        )
        process = self.sandbox.exec(
            "python",
            "-c",
            _BUILD_EXEC,
            REMOTE_RUNTIME,
            "/work/candidate.c",
            "/work/reference.c",
            "/work/candidate",
            "/work/reference",
            "/work/build.json",
            text=False,
            timeout=BUILD_TIMEOUT_SECONDS,
        )
        stdout, stderr = _diagnostics(process, sandbox=self.sandbox)
        if process.wait() != 0:
            detail = (stdout + stderr)[-DIAGNOSTIC_LIMIT:] or "no diagnostics"
            raise RuntimeError(f"sandbox solver build failed: {detail}")
        metadata = json.loads(self.sandbox.filesystem.read_text("/work/build.json"))
        self._executables = {
            "candidate": "/work/candidate",
            "reference": "/work/reference",
        }
        self._build = {
            "venue": {
                "id": VENUE_ID,
                "image_id": metadata["image_id"],
                "base_image": BASE_IMAGE,
                "cpus": self.execution["cpus"],
                "memory_bytes": self.execution["memory_bytes"],
                "limits_enforced": True,
            },
            "runtime_sha256": metadata["runtime_manifest_sha256"],
            "reference": {
                "id": REFERENCE_ID,
                "source_sha256": metadata["reference_source_sha256"],
                "executable_sha256": metadata["reference_executable_sha256"],
            },
            "candidate": {
                "source_sha256": metadata["candidate_source_sha256"],
                "executable_sha256": metadata["candidate_executable_sha256"],
            },
        }
        return dict(self._build)

    def run_reference(self, case: CaseInput) -> RawExecution:
        """Measure a fresh reference process for one public input."""
        return self._run("reference", case)

    def run_candidate(self, case: CaseInput) -> RawExecution:
        """Measure a fresh candidate process for one public input."""
        return self._run("candidate", case)

    def _run(self, solver: str, case: CaseInput) -> RawExecution:
        if not isinstance(case, CaseInput):
            raise TypeError("Modal transport accepts CaseInput only")
        if solver not in self._executables:
            raise RuntimeError("build the session before running cases")
        case_root = f"/work/cases/{self._case_number:06d}"
        self._case_number += 1
        compressed_path = f"{case_root}/input.zlib"
        input_path = f"{case_root}/input.bin"
        output_path = f"{case_root}/output.bin"
        payload = encode_input(case)
        self.sandbox.filesystem.write_bytes(
            zlib.compress(payload, level=1), compressed_path
        )
        try:
            started = time.monotonic()
            try:
                process = self.sandbox.exec(
                    "python",
                    "-c",
                    _INPUT_EXEC,
                    str(len(payload)),
                    compressed_path,
                    self._executables[solver],
                    input_path,
                    output_path,
                    text=False,
                    env={
                        "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                    },
                    timeout=int(RUN_TIMEOUT_SECONDS),
                )
            except Exception as exc:
                exceptions = getattr(require_modal(), "exception", None)
                timeout = getattr(exceptions, "ExecTimeoutError", ())
                if timeout and isinstance(exc, timeout):
                    raise RuntimeError(
                        "Modal interrupted the run; evaluation is incomplete"
                    ) from exc
                raise
            stdout, stderr = _diagnostics(process, sandbox=self.sandbox)
            returncode = process.wait()
            wall_seconds = time.monotonic() - started
            diagnostics = (stdout + stderr)[-DIAGNOSTIC_LIMIT:]
            if returncode == 125 and _INPUT_ERROR in diagnostics:
                raise RuntimeError(diagnostics)
            if returncode in {124, -1}:
                raise RuntimeError(
                    "Modal interrupted the run; evaluation is incomplete"
                )
            output = (
                _read_output(
                    self.sandbox,
                    output_path,
                    OUTPUT_HEADER.size + case.matrix.n * 8 + 1,
                )
                if returncode == 0
                else None
            )
            return RawExecution(returncode, wall_seconds, diagnostics, output)
        finally:
            self.sandbox.filesystem.remove(case_root, recursive=True)


@contextmanager
def open_session(
    candidate_source: pathlib.Path,
    reference_source: pathlib.Path,
    execution: Mapping,
):
    """Create one network-disabled sandbox for a paired evaluation."""
    cpus = execution.get("cpus")
    memory_bytes = execution.get("memory_bytes")
    if isinstance(cpus, bool) or not isinstance(cpus, (int, float)) or cpus <= 0:
        raise ValueError("cpus must be a positive number")
    if isinstance(memory_bytes, bool) or not isinstance(memory_bytes, int):
        raise ValueError("memory_bytes must be a positive whole number of MiB")
    memory_mib, remainder = divmod(memory_bytes, 1024**2)
    if remainder or memory_mib < 1:
        raise ValueError("memory_bytes must be a positive whole number of MiB")

    modal = require_modal()
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
