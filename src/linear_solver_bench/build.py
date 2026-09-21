"""Build the pinned HYPRE runtime and run one solver process.

This module is deliberately the whole native boundary.  A runtime is built
once, a solver is compiled once, and every case is executed in a fresh process.
Correctness checking lives in :mod:`linear_solver_bench.verify`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .assets import asset_root
from .integrity import file_sha256, json_sha256
from .models import CaseInput, RawExecution
from .protocol import (
    DIAGNOSTIC_LIMIT,
    INPUT_HEADER,
    INPUT_MAGIC,
    PROTOCOL_VERSION,
    output_size,
)

HYPRE_REPOSITORY = "https://github.com/hypre-space/hypre.git"
HYPRE_COMMIT = "e0ce7986941ce27111587ee7ea635ef7ca2eca94"
HYPRE_TREE = "c985eacc03d590011ab7652fe6fc4e9b25f1064d"
BUILD_KIND = "sequential-cpu-float64-int32-static"
CANDIDATE_ABI = "hypre-native-krylov-solver-v5"

MAX_SOURCE_BYTES = 65_536
FACTORY_EXPORT = "solver_create"
RUN_TIMEOUT_SECONDS = 600.0

REPOSITORY_ROOT = asset_root("native")
NATIVE_DIR = REPOSITORY_ROOT / "native"
REFERENCE_SOURCE = REPOSITORY_ROOT / "reference" / "solver.c"
REFERENCE_ID = "hypre-gmres-ilut-v1"
REFERENCE_SOURCE_SHA256 = (
    "b9314e7f2346b84df153e7dbc1b48fcca88fe7cf6770715da092b211112c8785"
)

_REFERENCE_SYMBOLS = {
    "HYPRE_ParCSRGMRESCreate",
    "HYPRE_ParCSRGMRESDestroy",
    "HYPRE_ParCSRGMRESSetup",
    "HYPRE_ParCSRGMRESSolve",
    "HYPRE_ParCSRGMRESSetKDim",
    "HYPRE_ParCSRGMRESSetMaxIter",
    "HYPRE_ParCSRGMRESSetTol",
    "HYPRE_ParCSRGMRESSetAbsoluteTol",
    "HYPRE_ParCSRGMRESSetPrecond",
}

_INCLUDE = re.compile(
    r'^[ \t]*#[ \t]*include[ \t]*([<"])([^>"]+)[>"][ \t]*(?://.*)?$',
    re.MULTILINE,
)
_INCLUDE_DIRECTIVE = re.compile(r"^[ \t]*#[ \t]*include\b.*$", re.MULTILINE)


class BuildError(RuntimeError):
    """A candidate failed a named build or audit stage."""

    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(f"{stage}: {message}")


@dataclass(frozen=True)
class Runtime:
    root: Path
    manifest_sha256: str


@dataclass(frozen=True)
class SolverBuild:
    executable: Path
    source_sha256: str
    executable_sha256: str


def _regular_file(path: Path, label: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ValueError(f"{label} is unavailable: {path}") from exc
    if not stat.S_ISREG(mode):
        raise ValueError(f"{label} must be a regular, non-symlink file")
    return path


def validate_source(source: Path) -> str:
    """Return the candidate text after enforcing the public source contract."""

    source = _regular_file(Path(source), "solver source")
    if source.stat().st_size > MAX_SOURCE_BYTES:
        raise BuildError("source audit", "source exceeds 65,536 bytes")
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise BuildError("source audit", "source must be UTF-8") from exc
    if "\0" in text or "\r" in text:
        raise BuildError("source audit", "source must use LF text without NUL bytes")

    includes = tuple(match.group(2) for match in _INCLUDE.finditer(text))
    directives = tuple(_INCLUDE_DIRECTIVE.finditer(text))
    if includes != ("HYPRE_parcsr_ls.h",) or len(directives) != 1:
        raise BuildError(
            "source audit", "source must include exactly HYPRE_parcsr_ls.h"
        )
    return text


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"required build tool is unavailable: {name}")
    return path


def _run(
    command: tuple[str, ...],
    *,
    timeout: float = 900.0,
    env: dict[str, str] | None = None,
    error_type: type[Exception] = RuntimeError,
    stage: str = "command",
) -> str:
    """Run a tool while retaining only the tail of its combined diagnostics."""

    with tempfile.TemporaryFile() as output:
        try:
            completed = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            message = f"timed out after {timeout:g} seconds"
            if error_type is BuildError:
                raise BuildError(stage, message) from exc
            raise error_type(f"{stage} {message}") from exc
        output.seek(0, os.SEEK_END)
        size = output.tell()
        output.seek(max(0, size - DIAGNOSTIC_LIMIT))
        detail = output.read().decode("utf-8", errors="replace").strip()
    if completed.returncode != 0:
        message = f"exited {completed.returncode}: {detail or 'no diagnostics'}"
        if error_type is BuildError:
            raise BuildError(stage, message)
        raise error_type(f"{stage} {message}")
    return detail


def _version(tool: str) -> str:
    return _run((tool, "--version"), timeout=30).splitlines()[0]


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256(b"linear-solve-bench-tree-v1\0")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"file tree is empty: {root}")
    for path in files:
        _regular_file(path, "runtime file")
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def _manifest(root: Path, tools: dict[str, str]) -> dict[str, object]:
    files = {
        "policy_header": "include/nsl_hypre_solver.h",
        "hypre_library": "lib/libHYPRE.a",
        "driver_object": "trusted/driver.o",
        "allowed_symbols": "trusted/allowed-symbols.txt",
    }
    return {
        "schema_version": 1,
        "build_kind": BUILD_KIND,
        "candidate_abi": CANDIDATE_ABI,
        "factory_export": FACTORY_EXPORT,
        "hypre": {
            "repository": HYPRE_REPOSITORY,
            "commit": HYPRE_COMMIT,
            "tree": HYPRE_TREE,
        },
        "include_tree_sha256": _tree_sha256(root / "include"),
        "files": {
            name: {"path": relative, "sha256": file_sha256(root / relative)}
            for name, relative in files.items()
        },
        "tool_versions": {name: _version(path) for name, path in tools.items()},
    }


def build_runtime(root: Path, jobs: int = 2) -> Runtime:
    """Build the one immutable, sequential HYPRE runtime used by the benchmark."""

    root = Path(root).expanduser().resolve()
    if root.exists():
        raise ValueError(f"runtime output already exists: {root}")
    if type(jobs) is not int or jobs <= 0:
        raise ValueError("jobs must be a positive integer")

    tools = {
        name: _tool(executable)
        for name, executable in {
            "cc": "cc",
            "cxx": "c++",
            "cmake": "cmake",
            "git": "git",
            "ninja": "ninja",
            "nm": "nm",
        }.items()
    }
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = root.parent / f".{root.name}.{uuid.uuid4().hex}.tmp"

    try:
        with tempfile.TemporaryDirectory(prefix="lsb-runtime-") as temporary:
            work = Path(temporary)
            source = work / "hypre"
            build = work / "hypre-build"
            install = work / "hypre-install"

            _run(
                (
                    tools["git"],
                    "clone",
                    "--no-checkout",
                    HYPRE_REPOSITORY,
                    str(source),
                ),
                stage="cloning HYPRE",
            )
            _run(
                (tools["git"], "-C", str(source), "checkout", "--detach", HYPRE_COMMIT),
                stage="checking out HYPRE",
            )
            commit = _run(
                (tools["git"], "-C", str(source), "rev-parse", "HEAD"),
                stage="reading HYPRE commit",
            )
            tree = _run(
                (tools["git"], "-C", str(source), "rev-parse", "HEAD^{tree}"),
                stage="reading HYPRE tree",
            )
            if commit != HYPRE_COMMIT or tree != HYPRE_TREE:
                raise RuntimeError(
                    "downloaded HYPRE source identity does not match the pin"
                )

            source_date = _run(
                (tools["git"], "-C", str(source), "show", "-s", "--format=%ct", "HEAD"),
                stage="reading HYPRE source date",
            )
            build_env = {
                **os.environ,
                "SOURCE_DATE_EPOCH": source_date,
                "ZERO_AR_DATE": "1",
            }
            prefix_map = f"-ffile-prefix-map={work}=/linear-solve-bench/runtime"
            _run(
                (
                    tools["cmake"],
                    "-S",
                    str(source / "src"),
                    "-B",
                    str(build),
                    "-G",
                    "Ninja",
                    f"-DCMAKE_INSTALL_PREFIX={install}",
                    "-DCMAKE_BUILD_TYPE=Release",
                    f"-DCMAKE_C_COMPILER={tools['cc']}",
                    f"-DCMAKE_CXX_COMPILER={tools['cxx']}",
                    f"-DCMAKE_C_FLAGS={prefix_map}",
                    f"-DCMAKE_CXX_FLAGS={prefix_map}",
                    "-DBUILD_SHARED_LIBS=OFF",
                    "-DHYPRE_ENABLE_MPI=OFF",
                    "-DHYPRE_ENABLE_OPENMP=OFF",
                    "-DHYPRE_ENABLE_CUDA=OFF",
                    "-DHYPRE_ENABLE_HIP=OFF",
                    "-DHYPRE_ENABLE_SYCL=OFF",
                    "-DHYPRE_ENABLE_SINGLE=OFF",
                    "-DHYPRE_ENABLE_LONG_DOUBLE=OFF",
                    "-DHYPRE_ENABLE_BIGINT=OFF",
                    "-DHYPRE_ENABLE_MIXEDINT=OFF",
                    "-DHYPRE_BUILD_EXAMPLES=OFF",
                    "-DHYPRE_BUILD_TESTS=OFF",
                ),
                env=build_env,
                stage="configuring HYPRE",
            )
            _run(
                (
                    tools["cmake"],
                    "--build",
                    str(build),
                    "--target",
                    "install",
                    "--parallel",
                    str(jobs),
                ),
                env=build_env,
                stage="building HYPRE",
            )
            libraries = tuple(install.rglob("libHYPRE.a"))
            if len(libraries) != 1:
                raise RuntimeError(
                    "HYPRE build did not produce exactly one static library"
                )

            (staging / "include").mkdir(parents=True)
            (staging / "lib").mkdir()
            (staging / "trusted").mkdir()
            for header in (install / "include").iterdir():
                if header.is_file():
                    shutil.copy2(header, staging / "include" / header.name)
            shutil.copy2(
                NATIVE_DIR / "include" / "nsl_hypre_solver.h",
                staging / "include" / "nsl_hypre_solver.h",
            )
            shutil.copy2(libraries[0], staging / "lib" / "libHYPRE.a")
            shutil.copy2(
                NATIVE_DIR / "allowed-symbols.txt",
                staging / "trusted" / "allowed-symbols.txt",
            )

            driver_build = work / "driver-build"
            _run(
                (
                    tools["cmake"],
                    "-S",
                    str(NATIVE_DIR),
                    "-B",
                    str(driver_build),
                    "-G",
                    "Ninja",
                    f"-DCMAKE_C_COMPILER={tools['cc']}",
                    f"-DCMAKE_CXX_COMPILER={tools['cxx']}",
                    f"-DCMAKE_INSTALL_PREFIX={staging}",
                    f"-DHYPRE_INCLUDE_DIR={staging / 'include'}",
                    f"-DLSB_RUNTIME_ROOT={staging}",
                ),
                env=build_env,
                stage="configuring the native driver",
            )
            _run(
                (tools["cmake"], "--build", str(driver_build), "--target", "install"),
                env=build_env,
                stage="building the native driver",
            )

        manifest = _manifest(staging, tools)
        manifest["manifest_sha256"] = json_sha256(manifest)
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, root)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return load_runtime(root)


def load_runtime(root: Path) -> Runtime:
    """Load a runtime after checking its pinned identity and trusted files."""

    root = Path(root).expanduser().resolve()
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"runtime manifest is unreadable: {root}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("runtime manifest schema mismatch")

    digest = manifest.get("manifest_sha256")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if not isinstance(digest, str) or digest != json_sha256(body):
        raise ValueError("runtime manifest digest mismatch")
    if (
        manifest.get("build_kind") != BUILD_KIND
        or manifest.get("candidate_abi") != CANDIDATE_ABI
        or manifest.get("factory_export") != FACTORY_EXPORT
        or manifest.get("hypre")
        != {
            "repository": HYPRE_REPOSITORY,
            "commit": HYPRE_COMMIT,
            "tree": HYPRE_TREE,
        }
        or manifest.get("include_tree_sha256") != _tree_sha256(root / "include")
    ):
        raise ValueError("runtime scientific identity mismatch")

    expected = {
        "policy_header": "include/nsl_hypre_solver.h",
        "hypre_library": "lib/libHYPRE.a",
        "driver_object": "trusted/driver.o",
        "allowed_symbols": "trusted/allowed-symbols.txt",
    }
    records = manifest.get("files")
    if not isinstance(records, dict) or set(records) != set(expected):
        raise ValueError("runtime manifest file list mismatch")
    for name, relative in expected.items():
        record = records[name]
        path = root / relative
        if (
            not isinstance(record, dict)
            or record.get("path") != relative
            or not path.is_file()
            or path.is_symlink()
            or record.get("sha256") != file_sha256(path)
        ):
            raise ValueError(f"runtime file identity mismatch: {name}")
    return Runtime(root, digest)


def _symbols(output: str, *, undefined: bool) -> tuple[str, ...]:
    symbols: set[str] = set()
    for line in output.splitlines():
        fields = line.split()
        if not fields:
            continue
        if undefined:
            if len(fields) >= 2 and fields[-2].upper() == "U":
                symbols.add(fields[-1])
            elif len(fields) == 1:
                symbols.add(fields[0])
        elif len(fields) >= 2 and fields[-2].upper() not in {"U", "?"}:
            symbols.add(fields[-1])
    return tuple(sorted(symbols))


def _normalize(symbol: str, allowed: set[str]) -> str:
    if symbol.startswith("_") and symbol[1:] in allowed:
        return symbol[1:]
    return symbol


def _build_solver(
    source: Path,
    output: Path,
    runtime: Runtime,
    additional_symbols: set[str] | None = None,
) -> SolverBuild:

    text = validate_source(source)
    if not isinstance(runtime, Runtime):
        raise TypeError("runtime must be a Runtime")
    loaded = load_runtime(runtime.root)
    if loaded.manifest_sha256 != runtime.manifest_sha256:
        raise ValueError("runtime identity changed after it was loaded")

    output = Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError(f"solver output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    cc, cxx, nm = _tool("cc"), _tool("c++"), _tool("nm")
    root = loaded.root

    with tempfile.TemporaryDirectory(
        prefix="lsb-solver-", dir=output.parent
    ) as temporary:
        work = Path(temporary)
        staged = work / "solver.c"
        staged.write_text(text, encoding="utf-8")
        candidate_object = work / "solver.o"
        executable = work / "solver"

        _run(
            (
                cc,
                "-std=c17",
                "-O3",
                "-DNDEBUG",
                "-fno-omit-frame-pointer",
                "-fvisibility=hidden",
                f"-ffile-prefix-map={work}=/linear-solve-bench/candidate",
                "-include",
                str(root / "include" / "nsl_hypre_solver.h"),
                "-I",
                str(root / "include"),
                "-c",
                str(staged),
                "-o",
                str(candidate_object),
            ),
            timeout=60,
            error_type=BuildError,
            stage="candidate compilation",
        )

        allowed = set(
            (root / "trusted" / "allowed-symbols.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        allowed.update(
            {
                "_GLOBAL_OFFSET_TABLE_",
                "__stack_chk_fail",
                "memcpy",
                "memmove",
                "memset",
                "sqrt",
            }
        )
        allowed.update(additional_symbols or ())
        undefined = {
            _normalize(symbol, allowed)
            for symbol in _symbols(
                _run(
                    (nm, "-u", str(candidate_object)),
                    timeout=30,
                    error_type=BuildError,
                    stage="candidate symbol audit",
                ),
                undefined=True,
            )
        }
        forbidden = sorted(undefined - allowed)
        if forbidden:
            raise BuildError(
                "candidate build", "forbidden symbols: " + ", ".join(forbidden)
            )

        exports = {
            _normalize(symbol, {FACTORY_EXPORT})
            for symbol in _symbols(
                _run(
                    (nm, "-g", str(candidate_object)),
                    timeout=30,
                    error_type=BuildError,
                    stage="candidate export audit",
                ),
                undefined=False,
            )
        }
        if exports != {FACTORY_EXPORT}:
            raise BuildError(
                "candidate build",
                f"global exports must be exactly {FACTORY_EXPORT}; "
                f"observed {sorted(exports)}",
            )

        _run(
            (
                cxx,
                str(root / "trusted" / "driver.o"),
                str(candidate_object),
                str(root / "lib" / "libHYPRE.a"),
                "-lm",
                "-o",
                str(executable),
            ),
            timeout=60,
            error_type=BuildError,
            stage="candidate link",
        )
        executable.chmod(0o755)
        executable_sha256 = file_sha256(executable)
        os.replace(executable, output)

    return SolverBuild(output, file_sha256(Path(source)), executable_sha256)


def build_solver(source: Path, output: Path, runtime: Runtime) -> SolverBuild:
    """Compile, audit, and link one participant translation unit."""

    return _build_solver(source, output, runtime)


def build_reference(source: Path, output: Path, runtime: Runtime) -> SolverBuild:
    """Build the immutable reference, with its small stock-GMRES allowance."""

    source = Path(source)
    if file_sha256(source) != REFERENCE_SOURCE_SHA256:
        raise BuildError("reference audit", "source does not match the fixed reference")
    return _build_solver(source, output, runtime, _REFERENCE_SYMBOLS)


def build_solvers(
    candidate: Path, reference: Path | None, workspace: Path, runtime: Runtime
) -> dict[str, SolverBuild]:
    """Build the solvers used by either execution venue, in reference-first order."""
    builds = {}
    if reference is not None:
        builds["reference"] = build_reference(
            reference, workspace / "reference", runtime
        )
    builds["candidate"] = build_solver(candidate, workspace / "candidate", runtime)
    return builds


def build_identities(builds: dict[str, SolverBuild], runtime: Runtime) -> dict:
    identities = {"runtime_sha256": runtime.manifest_sha256}
    for role, build in builds.items():
        identities[role] = {
            "source_sha256": build.source_sha256,
            "executable_sha256": build.executable_sha256,
        }
    if "reference" in builds:
        identities["reference"]["id"] = REFERENCE_ID
    return identities


def encode_input(value: CaseInput) -> bytes:
    """Encode one validated case for the trusted native driver."""

    matrix = value.matrix
    return b"".join(
        (
            INPUT_HEADER.pack(
                INPUT_MAGIC,
                PROTOCOL_VERSION,
                0,
                matrix.n,
                len(matrix.values),
                value.tolerance,
            ),
            np.asarray(matrix.row_offsets, dtype="<u8").tobytes(),
            np.asarray(matrix.column_indices, dtype="<u4").tobytes(),
            np.asarray(matrix.values, dtype="<f8").tobytes(),
            np.asarray(value.b, dtype="<f8").tobytes(),
            np.asarray(value.x0, dtype="<f8").tobytes(),
        )
    )


def run_solver(
    executable: Path,
    value: CaseInput,
    timeout_seconds: float = RUN_TIMEOUT_SECONDS,
) -> RawExecution:
    """Run one case in a fresh process and retain bounded raw evidence."""

    if not isinstance(value, CaseInput):
        raise TypeError("value must be a CaseInput")
    return _run_encoded_solver(
        executable, encode_input(value), value.matrix.n, timeout_seconds
    )


def _run_encoded_solver(
    executable: Path,
    payload: bytes,
    n: int,
    timeout_seconds: float = RUN_TIMEOUT_SECONDS,
) -> RawExecution:
    """Run authenticated wire input in a fresh process with bounded output."""
    timeout_seconds = float(timeout_seconds)
    if not np.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive and finite")
    executable = _regular_file(Path(executable), "solver executable")
    with tempfile.TemporaryDirectory(prefix="lsb-case-") as temporary:
        root = Path(temporary)
        input_path = root / "input.bin"
        output_path = root / "output.bin"
        input_path.write_bytes(payload)

        started = time.monotonic()
        process = subprocess.Popen(
            (str(executable), str(input_path), str(output_path)),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        )
        diagnostics = b""

        def read_diagnostics() -> None:
            nonlocal diagnostics
            assert process.stdout is not None
            with process.stdout:
                while chunk := process.stdout.read(64 * 1024):
                    diagnostics = (diagnostics + chunk)[-DIAGNOSTIC_LIMIT:]

        reader = threading.Thread(target=read_diagnostics, daemon=True)
        reader.start()
        timed_out = False
        try:
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            returncode = process.wait()
            reader.join()
        wall_seconds = time.monotonic() - started

        if timed_out:
            returncode = 124
            timeout_message = f"\nsolver timed out after {timeout_seconds:g} seconds"
            diagnostics = (diagnostics + timeout_message.encode())[-DIAGNOSTIC_LIMIT:]

        payload = None
        if returncode == 0:
            try:
                with output_path.open("rb") as result:
                    # The extra byte lets the verifier detect oversized output.
                    payload = result.read(output_size(n) + 1)
            except FileNotFoundError:
                pass
        return RawExecution(
            returncode=returncode,
            wall_seconds=wall_seconds,
            diagnostics=diagnostics.decode("utf-8", errors="replace"),
            output=payload,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build solvers inside the sandbox")
    parser.add_argument("runtime", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    runtime = load_runtime(args.runtime)
    builds = build_solvers(args.candidate, args.reference, args.workspace, runtime)
    (args.workspace / "build.json").write_text(
        json.dumps(
            {
                "identities": build_identities(builds, runtime),
                "executables": {
                    role: str(build.executable) for role, build in builds.items()
                },
                "image_id": os.environ.get("MODAL_IMAGE_ID"),
            }
        ),
        encoding="utf-8",
    )
