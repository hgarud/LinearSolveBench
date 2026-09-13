from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import pathlib
import zipfile
from collections import Counter

import numpy as np
import pytest
from scipy import sparse

from linear_solver_bench.models import CsrMatrix
from linear_solver_bench.workloads import matrix_digest, vector_digest


@pytest.fixture
def builder():
    path = pathlib.Path(__file__).resolve().parents[1] / "tools/build_flash_release.py"
    spec = importlib.util.spec_from_file_location("flash_release_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def catalogue(tmp_path):
    layout = [
        ("dev", "group-a", "run-a", "scored"),
        ("dev", "group-a", "run-a", "scored"),
        ("dev", "group-a", "run-b", "scored"),
        ("dev", "group-b", "run-c", "scored"),
        ("dev", "group-control", "run-control", "control"),
        ("ranked", "group-ranked", "run-ranked", "scored"),
    ]
    cases = []
    for index, (split, group, run, role) in enumerate(layout):
        matrix = CsrMatrix.from_scipy(
            sparse.csr_matrix([[3.0 + index, -1.0], [-1.0, 2.0]])
        )
        b, x0 = np.array([1.25, -3.5]), np.array([0.25, -0.75])
        members = {}
        for name, value in (("b.npy", b), ("x0.npy", x0)):
            buffer = io.BytesIO()
            np.save(buffer, value)
            members[name] = buffer.getvalue()
        buffer = io.BytesIO()
        sparse.save_npz(buffer, matrix.to_scipy())
        members["matrix.npz"] = buffer.getvalue()
        record = {
            "case_id": f"case-{index}",
            "split": split,
            "role": role,
            "provenance_group": group,
            "source_run": run,
            "n": matrix.n,
            "nnz": matrix.nnz,
            "tolerance": 1e-8,
            "step": index,
            "solve_index": 0,
            "matrix_sha256": matrix_digest(matrix),
            "b_sha256": vector_digest(b),
            "x0_sha256": vector_digest(x0),
            "path": f"cases/case-{index}.zip",
        }
        metadata = {
            "schema_version": 1,
            "family": "magnetic_diffusion_flash",
            "capture_kind": "scalar-magnetic-diffusion",
            "relative_tolerance": record["tolerance"],
            "absolute_tolerance": 0,
            **{
                key: record[key]
                for key in (
                    "case_id",
                    "provenance_group",
                    "source_run",
                    "n",
                    "nnz",
                    "step",
                    "solve_index",
                    "matrix_sha256",
                    "b_sha256",
                    "x0_sha256",
                )
            },
            "files": {
                name: hashlib.sha256(payload).hexdigest()
                for name, payload in members.items()
            },
        }
        members["case.json"] = json.dumps(metadata).encode("utf-8")
        archive_path = tmp_path / record["path"]
        archive_path.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive_path, "w") as archive:
            for name, payload in members.items():
                archive.writestr(name, payload)
        record.update(
            archive_bytes=archive_path.stat().st_size,
            archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        )
        cases.append(record)
    runs = {}
    counts = Counter(case["source_run"] for case in cases)
    for case in cases:
        runs[case["source_run"]] = {
            "source_run": case["source_run"],
            "problem": "Synthetic public test problem",
            "split": case["split"],
            "provenance_group": case["provenance_group"],
            "case_count": counts[case["source_run"]],
        }
    return {
        "schema_version": 1,
        "kind": "flash-replay-public-catalogue",
        "family": "magnetic_diffusion_flash",
        "case_count": len(cases),
        "split_counts": {"dev": 5, "ranked": 1},
        "cases": cases,
        "source_runs": list(runs.values()),
    }


def test_builds_pinned_splits_with_hierarchical_weights(builder, catalogue, tmp_path):
    revision = "a" * 40
    releases = builder.build_releases(
        catalogue, revision=revision, archive_dir=tmp_path
    )
    assert [case["weight"] for case in releases["dev"]["cases"]] == [
        0.125,
        0.125,
        0.25,
        0.5,
        0,
    ]
    for split, release in releases.items():
        assert release["release_id"] == "flash-replay-pilot"
        assert release["split"] == split
        assert release["execution"] == {
            "repetitions": 1,
            "cpus": 2,
            "memory_bytes": 4 * 1024**3,
            "case_timeout_seconds": 90.0,
            "max_iterations": 5000,
        }
        assert release["rhs"] is None
        for case in release["cases"]:
            assert case["qualification"] is None
            assert f"/resolve/{revision}/cases/" in case["source"]["url"]
            assert case["admission"]["citations"][0].endswith(
                f"/{revision}/catalogue.json"
            )
            assert "Synthetic public test problem" in case["admission"]["statement"]
    source, output = tmp_path / "catalogue.json", tmp_path / "manifests"
    source.write_text(json.dumps(catalogue), encoding="utf-8")
    assert (
        builder.main(
            [
                str(source),
                "--revision",
                revision,
                "--output-dir",
                str(output),
                "--archive-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert {path.name for path in output.iterdir()} == {
        "flash-replay-dev-pilot.json",
        "flash-replay-ranked-pilot.json",
    }
    for split, release in releases.items():
        assert (
            json.loads((output / f"flash-replay-{split}-pilot.json").read_text())
            == release
        )


@pytest.mark.parametrize("revision", ["main", "a" * 39, "A" * 40])
def test_requires_an_explicit_commit(builder, catalogue, revision):
    with pytest.raises(ValueError, match="40-character commit"):
        builder.build_releases(catalogue, revision=revision)


def test_archive_metadata_mismatch_leaves_no_manifests(builder, catalogue, tmp_path):
    record = catalogue["cases"][0]
    path = tmp_path / record["path"]
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    metadata = json.loads(members["case.json"])
    metadata["source_run"] = "incorrect-run"
    members["case.json"] = json.dumps(metadata).encode("utf-8")
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    record.update(
        archive_bytes=path.stat().st_size,
        archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    source, output = tmp_path / "catalogue.json", tmp_path / "manifests"
    source.write_text(json.dumps(catalogue), encoding="utf-8")
    assert (
        builder.main(
            [
                str(source),
                "--revision",
                "b" * 40,
                "--output-dir",
                str(output),
                "--archive-dir",
                str(tmp_path),
            ]
        )
        == 2
    )
    assert not output.exists()


def test_cross_split_operator_overlap_is_rejected(builder, catalogue):
    catalogue["cases"][-1]["matrix_sha256"] = catalogue["cases"][0]["matrix_sha256"]
    with pytest.raises(ValueError, match="operator exposure"):
        builder.build_releases(catalogue, revision="a" * 40)


def test_unsafe_archive_path_is_rejected(builder, catalogue):
    invalid = copy.deepcopy(catalogue)
    invalid["cases"][0]["path"] = "../case.zip"
    with pytest.raises(ValueError, match="relative ZIP"):
        builder.build_releases(invalid, revision="a" * 40)
