import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from linear_solver_bench.data import (
    _matrix_sha256,
    _vector_sha256,
    iter_cases,
    load_manifest,
)
from linear_solver_bench.families import resolve_family
from linear_solver_bench.models import CsrMatrix


def _archive_source(path: Path, **extra):
    payload = path.read_bytes()
    return {
        "url": path.as_uri(),
        "archive_bytes": len(payload),
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
        **extra,
    }


def _ns_manifest(tmp_path: Path):
    matrix_market = b"""%%MatrixMarket matrix coordinate real general
2 2 3
1 1 2
1 2 1
2 2 3
"""
    archive_path = tmp_path / "toy.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo("toy/toy.mtx")
        member.size = len(matrix_market)
        archive.addfile(member, io.BytesIO(matrix_market))
    matrix = CsrMatrix.from_scipy(sparse.csr_matrix(np.array([[2.0, 1.0], [0.0, 3.0]])))
    key = b"development-key!"
    case = {
        "index": 0,
        "case_id": "ns-toy",
        "n": 2,
        "nnz": 3,
        "tolerance": 1e-10,
        "role": "scored",
        "source": _archive_source(
            archive_path,
            kind="suitesparse",
            name="toy",
            group="test",
            matrix_id=1,
            matrix_sha256=_matrix_sha256(matrix),
        ),
    }
    return {
        "release_id": "ns-test",
        "family": "ns-mesh-pde",
        "rhs": {
            "scheme": "rademacher-hmac-sha256-v2",
            "key_id": hashlib.sha256(key).hexdigest(),
            "public_key_hex": key.hex(),
            "draw_index": 0,
        },
        "cases": [case],
    }


def _npy(value: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.save(output, value, allow_pickle=False)
    return output.getvalue()


def _flash_manifest(tmp_path: Path):
    scipy_matrix = sparse.csr_matrix(np.array([[4.0, 1.0], [0.0, 2.0]]))
    matrix = CsrMatrix.from_scipy(scipy_matrix)
    matrix_buffer = io.BytesIO()
    sparse.save_npz(matrix_buffer, scipy_matrix)
    files = {
        "matrix.npz": matrix_buffer.getvalue(),
        "b.npy": _npy(np.array([2.0, 3.0], dtype=np.float64)),
        "x0.npy": _npy(np.array([0.5, -0.5], dtype=np.float64)),
    }
    source_hashes = {
        "matrix_sha256": _matrix_sha256(matrix),
        "b_sha256": _vector_sha256(np.array([2.0, 3.0], dtype=np.float64)),
        "x0_sha256": _vector_sha256(np.array([0.5, -0.5], dtype=np.float64)),
    }
    metadata = {
        "schema_version": 1,
        "case_id": "flash-toy",
        "family": "magnetic_diffusion_flash",
        "capture_kind": "scalar-magnetic-diffusion",
        "n": 2,
        "nnz": 3,
        "relative_tolerance": 1e-9,
        "absolute_tolerance": 0,
        "provenance_group": "test-group",
        "source_run": "test-run",
        "step": 1,
        "solve_index": 0,
        **source_hashes,
        "files": {
            name: hashlib.sha256(value).hexdigest() for name, value in files.items()
        },
    }
    files["case.json"] = json.dumps(metadata).encode()
    archive_path = tmp_path / "flash.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name, value in files.items():
            archive.writestr(name, value)
    case = {
        "index": 0,
        "case_id": "flash-toy",
        "n": 2,
        "nnz": 3,
        "tolerance": 1e-9,
        "role": "scored",
        "provenance_group": "test-group",
        "source_run": "test-run",
        "admission": {"step": 1, "solve_index": 0},
        "source": _archive_source(
            archive_path,
            kind="huggingface",
            **source_hashes,
        ),
    }
    return {"family": "magnetic_diffusion_flash", "cases": [case]}


def test_bundled_manifests_are_authenticated():
    ns = load_manifest("ns_mesh_pde")
    flash = load_manifest(resolve_family("flash"))

    assert len(ns["cases"]) == 19
    assert len(flash["cases"]) == 96
    with pytest.raises(ValueError, match="choose"):
        resolve_family("anything-else")


def test_ns_archive_is_verified_and_target_is_isolated(tmp_path):
    manifest = _ns_manifest(tmp_path)
    cache = tmp_path / "cache"
    case = next(iter_cases(manifest, cache))

    assert set(case.target) <= {-1.0, 1.0}
    np.testing.assert_allclose(case.input.b, case.input.matrix.matvec(case.target))
    assert len(case.input_sha256) == 64
    assert not hasattr(case.input, "target")

    # A poisoned cache entry is discarded and fetched again from the source.
    cached = cache / manifest["cases"][0]["source"]["archive_sha256"]
    cached.write_bytes(b"wrong")
    assert next(iter_cases(manifest, cache)).input_sha256 == case.input_sha256


def test_flash_capture_preserves_vectors_and_hashes(tmp_path):
    case = next(iter_cases(_flash_manifest(tmp_path), tmp_path / "cache"))

    assert case.target is None
    np.testing.assert_array_equal(case.input.b, [2.0, 3.0])
    np.testing.assert_array_equal(case.input.x0, [0.5, -0.5])
    assert len(case.input_sha256) == 64


def test_case_selection_and_bad_download_digest(tmp_path):
    manifest = _ns_manifest(tmp_path)
    with pytest.raises(ValueError, match="unknown case"):
        list(iter_cases(manifest, tmp_path / "cache", ["missing"]))

    manifest["cases"][0]["source"]["archive_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        list(iter_cases(manifest, tmp_path / "other-cache"))
