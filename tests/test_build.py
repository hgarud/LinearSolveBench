from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from linear_solver_bench import build


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _mock_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> build.Runtime:
    cc, cxx, ar = shutil.which("cc"), shutil.which("c++"), shutil.which("ar")
    if not all((cc, cxx, ar, shutil.which("nm"))):
        pytest.skip("a C/C++ toolchain is required")

    root = tmp_path / "runtime"
    include = root / "include"
    trusted = root / "trusted"
    library = root / "lib" / "libHYPRE.a"
    include.mkdir(parents=True)
    trusted.mkdir()
    library.parent.mkdir()
    _write(
        include / "HYPRE_parcsr_ls.h",
        """
#ifndef MOCK_HYPRE
#define MOCK_HYPRE
typedef int HYPRE_Int;
typedef double HYPRE_Real;
typedef void *HYPRE_Solver;
#endif
""".lstrip(),
    )
    _write(
        include / "nsl_hypre_solver.h",
        '#include "HYPRE_parcsr_ls.h"\n'
        "HYPRE_Int solver_create(HYPRE_Solver *, HYPRE_Real, HYPRE_Real);\n",
    )
    driver = _write(
        tmp_path / "driver.cpp",
        """
extern "C" int solver_create(void **, double, double);
int main() {
    void *solver = nullptr;
    return solver_create(&solver, 1.0e-4, 0.0);
}
""".lstrip(),
    )
    subprocess.run(
        (cxx, "-c", str(driver), "-o", str(trusted / "driver.o")), check=True
    )
    unused = _write(tmp_path / "unused.c", "int unused_symbol = 0;\n")
    unused_object = tmp_path / "unused.o"
    subprocess.run((cc, "-c", str(unused), "-o", str(unused_object)), check=True)
    subprocess.run((ar, "rcs", str(library), str(unused_object)), check=True)
    _write(trusted / "allowed-symbols.txt", "")
    runtime = build.Runtime(root, "a" * 64)
    monkeypatch.setattr(build, "load_runtime", lambda path: runtime)
    return runtime


def _candidate(path: Path, body: str = "return 0;") -> Path:
    return _write(
        path,
        f"""#include "HYPRE_parcsr_ls.h"
HYPRE_Int solver_create(HYPRE_Solver *solver,
                        HYPRE_Real relative_tolerance,
                        HYPRE_Real absolute_tolerance) {{
    (void) solver;
    (void) relative_tolerance;
    (void) absolute_tolerance;
    {body}
}}
""",
    )


def test_source_contract_is_small_and_explicit(tmp_path: Path) -> None:
    assert "solver_create" in build.validate_source(build.EXAMPLE_SOURCE)
    bad = _write(
        tmp_path / "bad.c",
        '#include "HYPRE_parcsr_ls.h"\n#include <stdio.h>\n',
    )
    with pytest.raises(build.BuildError, match="include exactly"):
        build.validate_source(bad)

    link = tmp_path / "link.c"
    link.symlink_to(build.EXAMPLE_SOURCE)
    with pytest.raises(ValueError, match="non-symlink"):
        build.validate_source(link)


def test_solver_is_compiled_audited_and_linked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _mock_runtime(tmp_path, monkeypatch)
    source = _candidate(tmp_path / "candidate.c")
    result = build.build_solver(source, tmp_path / "candidate", runtime)
    assert result.executable.is_file()
    assert result.source_sha256 == build._sha256(source)
    assert result.executable_sha256 == build._sha256(result.executable)
    subprocess.run((str(result.executable),), check=True)


def test_solver_audit_rejects_forbidden_calls_and_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _mock_runtime(tmp_path, monkeypatch)
    forbidden = _candidate(
        tmp_path / "forbidden.c",
        'extern int puts(const char *); return puts("not allowed");',
    )
    with pytest.raises(build.BuildError, match="forbidden symbols.*puts"):
        build.build_solver(forbidden, tmp_path / "forbidden", runtime)

    wrong_export = (
        _candidate(tmp_path / "wrong.c")
        .read_text()
        .replace("solver_create", "old_solver_create")
    )
    wrong = _write(tmp_path / "wrong.c", wrong_export)
    with pytest.raises(build.BuildError, match="global exports"):
        build.build_solver(wrong, tmp_path / "wrong", runtime)


def test_runtime_manifest_detects_tampering(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    _write(root / "include" / "nsl_hypre_solver.h", "policy\n")
    _write(root / "lib" / "libHYPRE.a", "library\n")
    _write(root / "trusted" / "driver.o", "driver\n")
    _write(root / "trusted" / "allowed-symbols.txt", "symbol\n")
    relative_files = {
        "policy_header": "include/nsl_hypre_solver.h",
        "hypre_library": "lib/libHYPRE.a",
        "driver_object": "trusted/driver.o",
        "allowed_symbols": "trusted/allowed-symbols.txt",
    }
    body = {
        "schema_version": 1,
        "build_kind": build.BUILD_KIND,
        "candidate_abi": build.CANDIDATE_ABI,
        "factory_export": build.FACTORY_EXPORT,
        "hypre": {
            "repository": build.HYPRE_REPOSITORY,
            "commit": build.HYPRE_COMMIT,
            "tree": build.HYPRE_TREE,
        },
        "include_tree_sha256": build._tree_sha256(root / "include"),
        "files": {
            name: {"path": path, "sha256": build._sha256(root / path)}
            for name, path in relative_files.items()
        },
        "tool_versions": {},
    }
    manifest = {**body, "manifest_sha256": build._canonical_sha256(body)}
    _write(root / "manifest.json", json.dumps(manifest))
    runtime = build.load_runtime(root)
    assert runtime.root == root.resolve()

    _write(root / "trusted" / "driver.o", "changed\n")
    with pytest.raises(ValueError, match="driver_object"):
        build.load_runtime(root)


def test_only_the_fixed_reference_gets_trusted_symbols(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert build._sha256(build.REFERENCE_SOURCE) == build.REFERENCE_SOURCE_SHA256
    runtime = _mock_runtime(tmp_path, monkeypatch)
    changed = _write(
        tmp_path / "changed-reference.c",
        build.REFERENCE_SOURCE.read_text().replace("GMRES(200)", "GMRES(100)"),
    )
    with pytest.raises(build.BuildError, match="fixed reference"):
        build.build_reference(changed, tmp_path / "reference", runtime)
