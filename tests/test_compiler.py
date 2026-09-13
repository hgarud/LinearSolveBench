from __future__ import annotations

import pathlib
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from linear_solver_bench.compiler import (
    CandidateBuildError,
    build_candidate,
    validate_source,
)


def write(path: pathlib.Path, value: str) -> pathlib.Path:
    path.write_text(value, encoding="utf-8")
    return path


def mock_runtime(tmp_path: pathlib.Path):
    cc = shutil.which("cc")
    cxx = shutil.which("c++")
    nm = shutil.which("nm")
    ar = shutil.which("ar")
    if not all((cc, cxx, nm, ar)):
        pytest.skip("native compiler toolchain is unavailable")
    include = tmp_path / "include"
    include.mkdir()
    write(
        include / "HYPRE_parcsr_ls.h",
        """
#ifndef MOCK_HYPRE_HEADER
#define MOCK_HYPRE_HEADER
typedef int HYPRE_Int;
typedef double HYPRE_Real;
typedef void *HYPRE_Solver;
#endif
""".lstrip(),
    )
    policy_header = write(
        include / "nsl_hypre_solver.h",
        """
#include "HYPRE_parcsr_ls.h"
HYPRE_Int solver_create(HYPRE_Solver *, HYPRE_Real, HYPRE_Real, HYPRE_Int);
""".lstrip(),
    )
    driver_source = write(
        tmp_path / "driver.cpp",
        """
extern "C" int solver_create(void **, double, double, int);
int main() {
    void *solver = nullptr;
    return solver_create(&solver, 1.0e-4, 0.0, 10);
}
""".lstrip(),
    )
    driver_object = tmp_path / "driver.o"
    subprocess.run(
        (cxx, "-c", str(driver_source), "-o", str(driver_object)), check=True
    )
    unused_source = write(tmp_path / "unused.c", "int lsb_unused = 0;\n")
    unused_object = tmp_path / "unused.o"
    library = tmp_path / "libHYPRE.a"
    subprocess.run((cc, "-c", str(unused_source), "-o", str(unused_object)), check=True)
    subprocess.run((ar, "rcs", str(library), str(unused_object)), check=True)
    return SimpleNamespace(
        cc=cc,
        cxx=cxx,
        nm=nm,
        include_dir=include,
        policy_header=policy_header,
        driver_object=driver_object,
        hypre_library=library,
    )


def test_source_contract_accepts_starter_and_rejects_extra_include(
    tmp_path: pathlib.Path,
) -> None:
    validate_source(pathlib.Path("submissions/starter.c"))
    bad = write(
        tmp_path / "bad.c",
        '#include "HYPRE_parcsr_ls.h"\n#include <stdio.h>\n',
    )
    with pytest.raises(CandidateBuildError, match="include exactly"):
        validate_source(bad)


def test_candidate_is_compiled_audited_and_linked(tmp_path: pathlib.Path) -> None:
    runtime = mock_runtime(tmp_path)
    source = write(
        tmp_path / "candidate.c",
        """
#include "HYPRE_parcsr_ls.h"
HYPRE_Int solver_create(HYPRE_Solver *solver,
                        HYPRE_Real relative_tolerance,
                        HYPRE_Real absolute_tolerance,
                        HYPRE_Int maximum_iterations) {
    (void) relative_tolerance;
    (void) absolute_tolerance;
    (void) maximum_iterations;
    *solver = (HYPRE_Solver) 0;
    return 0;
}
""".lstrip(),
    )
    build = build_candidate(source, tmp_path / "candidate-build", runtime=runtime)
    assert build.executable.is_file()
    assert build.executable.stat().st_mode & 0o111
    assert build.exported_symbols == ("solver_create",)
    subprocess.run((str(build.executable),), check=True)

    old_source = write(
        tmp_path / "old-candidate.c",
        source.read_text().replace("solver_create", "nsl_create"),
    )
    with pytest.raises(CandidateBuildError, match="global exports must be exactly"):
        build_candidate(old_source, tmp_path / "old-build", runtime=runtime)


def test_forbidden_undefined_symbol_is_rejected(tmp_path: pathlib.Path) -> None:
    runtime = mock_runtime(tmp_path)
    source = write(
        tmp_path / "candidate.c",
        """
#include "HYPRE_parcsr_ls.h"
extern int puts(const char *);
HYPRE_Int solver_create(HYPRE_Solver *solver,
                        HYPRE_Real relative_tolerance,
                        HYPRE_Real absolute_tolerance,
                        HYPRE_Int maximum_iterations) {
    (void) solver;
    (void) relative_tolerance;
    (void) absolute_tolerance;
    (void) maximum_iterations;
    return puts("forbidden");
}
""".lstrip(),
    )
    with pytest.raises(CandidateBuildError, match="forbidden symbols"):
        build_candidate(source, tmp_path / "candidate-build", runtime=runtime)
