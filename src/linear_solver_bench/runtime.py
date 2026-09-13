"""Build and validate the pinned, candidate-independent HYPRE runtime."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shlex
import shutil
import tempfile
import uuid
from dataclasses import dataclass

from .capabilities import KRYLOV_PRIMITIVE_ABI, KRYLOV_PRIMITIVE_SURFACE_SHA256
from .compiler import FACTORY_EXPORT, sha256_file
from .paths import native_dir
from .process import bounded_process

HYPRE_REPOSITORY = "https://github.com/hypre-space/hypre.git"
HYPRE_COMMIT = "e0ce7986941ce27111587ee7ea635ef7ca2eca94"
HYPRE_TREE = "c985eacc03d590011ab7652fe6fc4e9b25f1064d"
RUNTIME_SCHEMA = 1
BUILD_KIND = "sequential-cpu-float64-int32-static"


@dataclass(frozen=True)
class Runtime:
    root: pathlib.Path
    manifest_sha256: str
    include_dir: pathlib.Path
    policy_header: pathlib.Path
    hypre_library: pathlib.Path
    driver_object: pathlib.Path
    cc: str
    cxx: str
    nm: str


def _run(
    command: tuple[str, ...],
    *,
    timeout_s: float = 900.0,
    env: dict[str, str] | None = None,
) -> str:
    result = bounded_process(command, timeout_s=timeout_s, env=env)
    if result.returncode != 0:
        output = result.output.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"runtime command failed ({result.returncode}): {' '.join(command)}\n"
            f"{output[-8000:]}"
        )
    return result.output.decode("utf-8", errors="replace").strip()


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"required build tool is unavailable: {name}")
    return path


def _tree_sha256(root: pathlib.Path) -> str:
    digest = hashlib.sha256(b"linear-solver-bench-file-tree-v1\0")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"runtime file tree is empty: {root}")
    for path in files:
        if path.is_symlink():
            raise ValueError(f"runtime file tree contains a symlink: {path}")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _manifest_body(root: pathlib.Path, tools: dict[str, str]) -> dict[str, object]:
    files = {
        "policy_header": "include/nsl_hypre_solver.h",
        "hypre_library": "lib/libHYPRE.a",
        "driver_object": "trusted/driver.o",
    }
    return {
        "schema_version": RUNTIME_SCHEMA,
        "build_kind": BUILD_KIND,
        "candidate_factory_export": FACTORY_EXPORT,
        "hypre_repository": HYPRE_REPOSITORY,
        "hypre_commit": HYPRE_COMMIT,
        "hypre_tree": HYPRE_TREE,
        "krylov_primitive_abi": KRYLOV_PRIMITIVE_ABI,
        "krylov_primitive_surface_sha256": KRYLOV_PRIMITIVE_SURFACE_SHA256,
        "include_tree_sha256": _tree_sha256(root / "include"),
        "files": {
            name: {"path": relpath, "sha256": sha256_file(root / relpath)}
            for name, relpath in files.items()
        },
        "tools": tools,
        "tool_versions": {
            name: _run((path, "--version"), timeout_s=30).splitlines()[0]
            for name, path in tools.items()
        },
    }


def build_runtime(
    output: pathlib.Path,
    *,
    hypre_source: str = HYPRE_REPOSITORY,
    jobs: int = 2,
) -> Runtime:
    output = pathlib.Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError("runtime output already exists")
    if type(jobs) is not int or jobs <= 0:
        raise ValueError("jobs must be a positive integer")
    tools = {
        "cc": _tool("cc"),
        "cxx": _tool("c++"),
        "cmake": _tool("cmake"),
        "git": _tool("git"),
        "ninja": _tool("ninja"),
        "nm": _tool("nm"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    try:
        with tempfile.TemporaryDirectory(prefix="lsb-runtime-") as temporary:
            work = pathlib.Path(temporary)
            source = work / "hypre"
            build = work / "build"
            install = work / "install"
            _run((tools["git"], "clone", "--no-checkout", hypre_source, str(source)))
            _run(
                (tools["git"], "-C", str(source), "checkout", "--detach", HYPRE_COMMIT)
            )
            commit = _run((tools["git"], "-C", str(source), "rev-parse", "HEAD"))
            tree = _run((tools["git"], "-C", str(source), "rev-parse", "HEAD^{tree}"))
            if commit != HYPRE_COMMIT or tree != HYPRE_TREE:
                raise RuntimeError("HYPRE source identity mismatch")
            # HYPRE error paths use __FILE__. Without normalization, random
            # temporary directories alter the static library and invalidate
            # an otherwise identical runtime's frozen reference calibration.
            prefix_map = shlex.quote(
                f"-ffile-prefix-map={work}=/linear-solver-bench/runtime"
            )
            build_env = {
                **os.environ,
                "SOURCE_DATE_EPOCH": _run(
                    (
                        tools["git"],
                        "-C",
                        str(source),
                        "show",
                        "-s",
                        "--format=%ct",
                        "HEAD",
                    )
                ),
                "ZERO_AR_DATE": "1",
            }
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
            )
            libraries = tuple(install.rglob("libHYPRE.a"))
            if len(libraries) != 1:
                raise RuntimeError(
                    "pinned HYPRE build did not produce one static library"
                )

            (staging / "include").mkdir(parents=True)
            (staging / "lib").mkdir()
            (staging / "trusted").mkdir()
            for header in (install / "include").iterdir():
                if header.is_file():
                    shutil.copy2(header, staging / "include" / header.name)
            shutil.copy2(
                native_dir() / "include" / "nsl_hypre_solver.h",
                staging / "include" / "nsl_hypre_solver.h",
            )
            shutil.copy2(libraries[0], staging / "lib" / "libHYPRE.a")
            _run(
                (
                    tools["cxx"],
                    "-std=c++17",
                    "-O3",
                    "-DNDEBUG",
                    "-fno-omit-frame-pointer",
                    f"-ffile-prefix-map={native_dir()}=/linear-solver-bench/native",
                    f"-ffile-prefix-map={staging}=/linear-solver-bench/runtime",
                    "-I",
                    str(staging / "include"),
                    "-c",
                    str(native_dir() / "src" / "driver.cpp"),
                    "-o",
                    str(staging / "trusted" / "driver.o"),
                ),
                env=build_env,
            )
        body = _manifest_body(staging, tools)
        body_json = json.dumps(
            body, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        manifest = {
            **body,
            "manifest_sha256": hashlib.sha256(body_json.encode("ascii")).hexdigest(),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return load_runtime(output)


def load_runtime(root: pathlib.Path) -> Runtime:
    root = pathlib.Path(root).expanduser().resolve()
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("runtime manifest is unreadable") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("runtime manifest schema mismatch")
    digest = manifest.get("manifest_sha256")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    observed = hashlib.sha256(encoded.encode("ascii")).hexdigest()
    if digest != observed:
        raise ValueError("runtime manifest digest mismatch")
    if manifest.get("candidate_factory_export") != FACTORY_EXPORT:
        raise ValueError(
            f"runtime must be rebuilt for the {FACTORY_EXPORT} candidate entry point"
        )
    if (
        manifest.get("hypre_commit") != HYPRE_COMMIT
        or manifest.get("hypre_tree") != HYPRE_TREE
        or manifest.get("build_kind") != BUILD_KIND
        or manifest.get("krylov_primitive_abi") != KRYLOV_PRIMITIVE_ABI
        or manifest.get("krylov_primitive_surface_sha256")
        != KRYLOV_PRIMITIVE_SURFACE_SHA256
        or manifest.get("include_tree_sha256") != _tree_sha256(root / "include")
    ):
        raise ValueError("runtime scientific identity mismatch")
    expected_files = {
        "policy_header": "include/nsl_hypre_solver.h",
        "hypre_library": "lib/libHYPRE.a",
        "driver_object": "trusted/driver.o",
    }
    paths = {}
    for name, record in manifest.get("files", {}).items():
        if name not in expected_files or record.get("path") != expected_files[name]:
            raise ValueError(f"runtime file path mismatch: {name}")
        path = root / expected_files[name]
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"runtime file identity mismatch: {name}")
        paths[name] = path
    tools = manifest.get("tools", {})
    if set(paths) != {"policy_header", "hypre_library", "driver_object"} or not all(
        isinstance(tools.get(name), str) for name in ("cc", "cxx", "nm")
    ):
        raise ValueError("runtime manifest is incomplete")
    return Runtime(
        root=root,
        manifest_sha256=str(digest),
        include_dir=root / "include",
        policy_header=paths["policy_header"],
        hypre_library=paths["hypre_library"],
        driver_object=paths["driver_object"],
        cc=tools["cc"],
        cxx=tools["cxx"],
        nm=tools["nm"],
    )
