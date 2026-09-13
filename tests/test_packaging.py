from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tarfile
import zipfile

import pytest

from linear_solver_bench import paths

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_resource_and_cache_overrides(monkeypatch, tmp_path):
    root = tmp_path / "assets"
    root.mkdir()
    (root / "benchmark.toml").write_text("schema_version = 1\n", encoding="utf-8")
    monkeypatch.setenv("LINEAR_SOLVER_BENCH_ROOT", str(root))
    monkeypatch.delenv("LINEAR_SOLVER_BENCH_DATA", raising=False)
    assert paths.repository_root() == root
    assert paths.data_dir() == root / "data"
    assert paths.native_dir() == root / "native"

    monkeypatch.setenv("LINEAR_SOLVER_BENCH_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("LINEAR_SOLVER_BENCH_CACHE", str(tmp_path / "cache"))
    assert paths.data_dir() == tmp_path / "data"
    assert paths.cache_dir() == tmp_path / "cache"

    monkeypatch.setenv("LINEAR_SOLVER_BENCH_ROOT", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="cannot locate benchmark.toml"):
        paths.repository_root()


def test_default_cache_is_outside_installed_assets(monkeypatch, tmp_path):
    monkeypatch.delenv("LINEAR_SOLVER_BENCH_CACHE", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    assert paths.cache_dir() == tmp_path / "xdg-cache" / "linear-solver-bench"
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert paths.cache_dir() == pathlib.Path.home() / ".cache" / "linear-solver-bench"


@pytest.fixture(scope="module")
def distributions(tmp_path_factory):
    output = tmp_path_factory.mktemp("distribution")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--outdir",
            str(output),
            str(ROOT),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return next(output.glob("*.tar.gz")), next(output.glob("*.whl"))


def test_archives_include_public_resources_only(distributions):
    sdist, wheel = distributions
    with tarfile.open(sdist, "r:gz") as archive:
        source_files = {
            pathlib.PurePosixPath(member.name).relative_to(
                pathlib.PurePosixPath(member.name).parts[0]
            ): archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isfile()
        }
    allowed_source_roots = {
        "CITATION.cff",
        "CONTRIBUTING.md",
        "LICENSE",
        "MANIFEST.in",
        "PKG-INFO",
        "README.md",
        "SECURITY.md",
        "benchmark.toml",
        "data",
        "docs",
        "modal_app.py",
        "pyproject.toml",
        "qualification_modal.py",
        "setup.cfg",
        "setup.py",
        "src",
        "submissions",
        "tests",
        "tools",
        "native",
    }
    assert {path.parts[0] for path in source_files} <= allowed_source_roots
    assert not any(
        path.parts[:2] in {("data", "downloads"), ("data", "prepared")}
        for path in source_files
    )
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert all(
            name.startswith("linear_solver_bench/")
            or (
                name.split("/")[0].startswith("linear_solver_bench-")
                and name.split("/")[0].endswith(".dist-info")
            )
            for name in names
        )
        resource_prefix = "linear_solver_bench/_resources/"
        packaged = {
            pathlib.PurePosixPath(name.removeprefix(resource_prefix)): archive.read(
                name
            )
            for name in names
            if name.startswith(resource_prefix)
        }
    required = {
        pathlib.PurePosixPath(name)
        for name in (
            "benchmark.toml",
            "data/catalogue.json",
            "data/dev-v1.json",
            "data/ranked-v1.json",
            "data/ns-mesh-pilot-dev.json",
            "data/ns-mesh-pilot-ranked.json",
            "data/flash-replay-dev-pilot.json",
            "data/flash-replay-ranked-pilot.json",
            "data/releases/flash-replay-dev-reference.json",
            "data/releases/flash-replay-ranked-reference.json",
            "data/releases/cpu-runtime-v2.json",
            "data/releases/README.md",
            "data/qualification.json",
            "data/qualification-inputs-v1.json",
            "data/NOTICE.md",
            "native/CMakeLists.txt",
            "native/src/driver.cpp",
            "native/include/nsl_hypre_solver.h",
            "native/include/NOTICE-HYPRE",
            "native/include/LICENSE-HYPRE-MIT",
        )
    }
    assert required <= packaged.keys()
    assert all(
        path.parts[0] in {"benchmark.toml", "data", "native"} for path in packaged
    )
    assert not any(
        path.parts[:2] in {("data", "downloads"), ("data", "prepared")}
        for path in packaged
    )
    assert all(content == source_files[path] for path, content in packaged.items())


def test_wheel_resources_work_outside_checkout(distributions, tmp_path):
    _, wheel = distributions
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"LINEAR_SOLVER_BENCH_ROOT", "LINEAR_SOLVER_BENCH_DATA"}
    }
    # A zip import also exercises resource extraction and its lifetime. Imports
    # must resolve inside the wheel, even if the test environment is editable.
    environment["PYTHONPATH"] = str(wheel)
    script = """
import pathlib
import json
import sys
import linear_solver_bench
from linear_solver_bench.dataset import load_split
from linear_solver_bench.manifests import load_release
from linear_solver_bench.paths import data_dir, native_dir, repository_root
from linear_solver_bench.pilot_scoring import score_pilot_report
assert linear_solver_bench.__file__.startswith(sys.argv[1])
assert load_split("dev")["case_count"] > 0
assert (repository_root() / "benchmark.toml").is_file()
assert (data_dir() / "NOTICE.md").is_file()
for name in ("ns-mesh-pilot-dev", "ns-mesh-pilot-ranked",
             "flash-replay-dev-pilot", "flash-replay-ranked-pilot"):
    release = load_release(data_dir() / (name + ".json"), official=True)
    assert release["cases"]
for split in ("dev", "ranked"):
    reference = json.loads((data_dir() / "releases" /
        ("flash-replay-" + split + "-reference.json")).read_text())
    score = score_pilot_report(reference["reference_report"], reference)
    assert score["official"] and score["speedup"] == 1.0
assert (native_dir() / "src" / "driver.cpp").is_file()
assert (native_dir() / "include" / "nsl_hypre_solver.h").is_file()
assert pathlib.Path.cwd() not in repository_root().parents
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(wheel)],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
