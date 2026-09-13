"""Compile and link one audited C translation unit."""

from __future__ import annotations

import hashlib
import pathlib
import re
import shutil
import stat
from dataclasses import dataclass

from .capabilities import KRYLOV_PRIMITIVE_SYMBOLS
from .process import ProcessTimeout, bounded_process

MAX_SOURCE_BYTES = 65_536
FACTORY_EXPORT = "solver_create"
_INCLUDE_RE = re.compile(
    r"^[ \t]*#[ \t]*include[ \t]*([<\"])([^>\"]+)[>\"][ \t]*(?://.*)?$",
    re.MULTILINE,
)
_INCLUDE_DIRECTIVE_RE = re.compile(r"^[ \t]*#[ \t]*include\b.*$", re.MULTILINE)


class CandidateBuildError(RuntimeError):
    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(f"CANDIDATE_{stage.upper()}_FAILED: {message}")


@dataclass(frozen=True)
class CandidateBuild:
    executable: pathlib.Path
    source_sha256: str
    object_sha256: str
    executable_sha256: str
    undefined_symbols: tuple[str, ...]
    exported_symbols: tuple[str, ...]
    compile_wall_s: float


def _regular(path: pathlib.Path, label: str) -> pathlib.Path:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ValueError(f"{label} is unavailable: {path}") from exc
    if not stat.S_ISREG(mode) or path.is_symlink():
        raise ValueError(f"{label} must be a regular, non-symlink file")
    return path


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source(path: pathlib.Path) -> str:
    source = _regular(pathlib.Path(path), "candidate source")
    if source.stat().st_size > MAX_SOURCE_BYTES:
        raise CandidateBuildError("static", "candidate source exceeds 65,536 bytes")
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CandidateBuildError("static", "candidate source is not UTF-8") from exc
    if "\x00" in text or "\r" in text:
        raise CandidateBuildError("static", "candidate source must use LF text")
    includes = tuple(match.group(2) for match in _INCLUDE_RE.finditer(text))
    directives = tuple(_INCLUDE_DIRECTIVE_RE.finditer(text))
    if includes != ("HYPRE_parcsr_ls.h",) or len(directives) != 1:
        raise CandidateBuildError(
            "static",
            "candidate must include exactly HYPRE_parcsr_ls.h",
        )
    return text


def _symbols(output: bytes, *, undefined: bool) -> tuple[str, ...]:
    symbols = set()
    for line in output.decode("utf-8", errors="replace").splitlines():
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
    return symbol[1:] if symbol.startswith("_") and symbol[1:] in allowed else symbol


def _checked(command: tuple[str, ...], timeout_s: float, stage: str):
    try:
        result = bounded_process(command, timeout_s=timeout_s)
    except ProcessTimeout as exc:
        raise CandidateBuildError(stage, f"{command[0]} timed out") from exc
    if result.returncode != 0:
        detail = result.output.decode("utf-8", errors="replace").strip()
        raise CandidateBuildError(
            stage,
            f"command exited {result.returncode}: {detail[-8000:] or 'no diagnostics'}",
        )
    return result


def build_candidate(
    source: pathlib.Path,
    workspace: pathlib.Path,
    *,
    runtime: object,
    timeout_s: float = 60.0,
) -> CandidateBuild:
    validate_source(source)
    workspace = pathlib.Path(workspace)
    if workspace.exists():
        raise ValueError("candidate build workspace already exists")
    source_dir = workspace / "source"
    build_dir = workspace / "build"
    source_dir.mkdir(parents=True)
    build_dir.mkdir()
    staged = source_dir / "policy.c"
    shutil.copyfile(source, staged)
    candidate_object = build_dir / "policy.o"
    executable = build_dir / "lsb_driver"
    compile_command = (
        runtime.cc,
        "-std=c17",
        "-O3",
        "-DNDEBUG",
        "-fno-omit-frame-pointer",
        "-fvisibility=hidden",
        "-include",
        str(runtime.policy_header),
        "-I",
        str(runtime.include_dir),
        "-c",
        str(staged),
        "-o",
        str(candidate_object),
    )
    compiled = _checked(compile_command, timeout_s, "compile")

    undefined_result = _checked(
        (runtime.nm, "-u", str(candidate_object)), timeout_s, "abi"
    )
    allowed = {
        "__stack_chk_fail",
        "memcpy",
        "memmove",
        "memset",
        "sqrt",
        *KRYLOV_PRIMITIVE_SYMBOLS,
    }
    undefined = tuple(
        sorted(
            _normalize(symbol, allowed)
            for symbol in _symbols(undefined_result.output, undefined=True)
        )
    )
    forbidden = sorted(set(undefined) - allowed)
    if forbidden:
        raise CandidateBuildError("abi", "forbidden symbols: " + ", ".join(forbidden))

    export_result = _checked(
        (runtime.nm, "-g", str(candidate_object)), timeout_s, "abi"
    )
    exports = tuple(
        sorted(
            _normalize(symbol, {FACTORY_EXPORT})
            for symbol in _symbols(export_result.output, undefined=False)
        )
    )
    if exports != (FACTORY_EXPORT,):
        raise CandidateBuildError(
            "abi",
            f"global exports must be exactly {FACTORY_EXPORT}; observed {exports!r}",
        )

    linked = _checked(
        (
            runtime.cxx,
            str(runtime.driver_object),
            str(candidate_object),
            str(runtime.hypre_library),
            "-lm",
            "-o",
            str(executable),
        ),
        timeout_s,
        "link",
    )
    executable.chmod(0o755)
    return CandidateBuild(
        executable=executable,
        source_sha256=sha256_file(staged),
        object_sha256=sha256_file(candidate_object),
        executable_sha256=sha256_file(executable),
        undefined_symbols=undefined,
        exported_symbols=exports,
        compile_wall_s=(
            compiled.wall_s
            + undefined_result.wall_s
            + export_result.wall_s
            + linked.wall_s
        ),
    )
