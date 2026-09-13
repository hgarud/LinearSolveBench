"""Read the four-file public FLASH replay format and preserve captured arrays."""

from __future__ import annotations

import hashlib
import io
import json
import pathlib
import stat
import tarfile
import zipfile
from collections.abc import Mapping

import numpy as np

from ..manifests import strict_object
from ..models import CsrMatrix, EvaluationSystem, MatrixInput
from ..workloads import matrix_digest, vector_digest

CASE_FILES = {"matrix.npz", "b.npy", "x0.npy", "case.json"}


def _load_npy(payload: bytes, shape: tuple[int, ...], kinds: set[str]) -> np.ndarray:
    """Reject deceptive NPY shapes before NumPy can allocate their arrays."""
    handle = io.BytesIO(payload)
    version = np.lib.format.read_magic(handle)
    if version == (1, 0):
        actual, _, dtype = np.lib.format.read_array_header_1_0(
            handle, max_header_size=4096
        )
    elif version == (2, 0):
        actual, _, dtype = np.lib.format.read_array_header_2_0(
            handle, max_header_size=4096
        )
    else:
        raise ValueError("unsupported public NPY format version")
    if actual != shape or dtype.hasobject or dtype.kind not in kinds:
        raise ValueError("public array shape or dtype disagrees with its case")
    count = 1
    for dimension in actual:
        count *= dimension
    if len(payload) - handle.tell() != count * dtype.itemsize:
        raise ValueError("public array byte count disagrees with its header")
    return np.load(io.BytesIO(payload), allow_pickle=False, max_header_size=4096)


def _archive_files(path: pathlib.Path, case: Mapping[str, object]) -> dict[str, bytes]:
    limits = {
        "matrix.npz": 32 * case["nnz"] + 16 * (case["n"] + 1) + 65536,
        "b.npy": 8 * case["n"] + 4096,
        "x0.npy": 8 * case["n"] + 4096,
        "case.json": 65536,
    }
    result = {}
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                members = archive.infolist()
                if len(members) != len(CASE_FILES):
                    raise ValueError("FLASH archive must contain exactly four files")
                for member in members:
                    mode = member.external_attr >> 16
                    if (
                        member.filename not in CASE_FILES
                        or member.filename in result
                        or member.is_dir()
                        or stat.S_ISLNK(mode)
                        or member.file_size > limits[member.filename]
                    ):
                        raise ValueError("unsafe FLASH archive member")
                    result[member.filename] = archive.read(member)
        else:
            with tarfile.open(path, "r:gz") as archive:
                for member in archive:
                    if (
                        member.name not in CASE_FILES
                        or member.name in result
                        or not member.isfile()
                        or member.size > limits[member.name]
                        or member.size < 0
                    ):
                        raise ValueError("unsafe FLASH archive member")
                    handle = archive.extractfile(member)
                    if handle is None:
                        raise ValueError("FLASH archive member is unreadable")
                    with handle:
                        result[member.name] = handle.read(limits[member.name] + 1)
    except (OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
        raise ValueError("FLASH case archive is unreadable") from exc
    if set(result) != CASE_FILES:
        raise ValueError(
            "FLASH archive must contain exactly the four public case files"
        )
    return result


def _load_matrix(payload: bytes, n: int, nnz: int) -> CsrMatrix:
    # NPZ itself is compressed: validate its inner members before NumPy allocates
    # arrays. The outer archive limit alone does not bound decompression.
    limits = {
        "data.npy": nnz * 8 + 4096,
        "indices.npy": nnz * 8 + 4096,
        "indptr.npy": (n + 1) * 8 + 4096,
        "format.npy": 4096,
        "shape.npy": 4096,
        "_is_array.npy": 4096,
    }
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as inner:
            names = set()
            arrays = {}
            for info in inner.infolist():
                if (
                    info.filename not in limits
                    or info.filename in names
                    or info.file_size > limits[info.filename]
                ):
                    raise ValueError("unsafe compressed matrix member")
                names.add(info.filename)
                arrays[info.filename] = inner.read(info)
            if not {
                "data.npy",
                "indices.npy",
                "indptr.npy",
                "format.npy",
                "shape.npy",
            }.issubset(names):
                raise ValueError("FLASH matrix archive is incomplete")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("FLASH matrix archive is unreadable") from exc
    values = _load_npy(arrays["data.npy"], (nnz,), {"f"})
    columns = _load_npy(arrays["indices.npy"], (nnz,), {"i", "u"})
    rows = _load_npy(arrays["indptr.npy"], (n + 1,), {"i", "u"})
    shape = _load_npy(arrays["shape.npy"], (2,), {"i", "u"})
    form = _load_npy(arrays["format.npy"], (), {"S"})
    if "_is_array.npy" in arrays:
        _load_npy(arrays["_is_array.npy"], (), {"b"})
    if tuple(shape) != (n, n) or form.item() != b"csr" or values.dtype != np.float64:
        raise ValueError(
            "FLASH matrix must be float64 CSR with the declared dimensions"
        )
    if (
        np.any(columns < 0)
        or np.any(columns >= n)
        or np.any(rows < 0)
        or np.any(rows > nnz)
    ):
        raise ValueError("FLASH CSR index lies outside the declared dimensions")
    if np.any(values == 0):
        raise ValueError(
            "FLASH matrix must already be canonical with no explicit zeros"
        )
    return CsrMatrix(
        n=n,
        row_offsets=np.asarray(rows, dtype=np.uint64),
        column_indices=np.asarray(columns, dtype=np.uint32),
        values=values,
    )


def load_flash(path: pathlib.Path, case: Mapping[str, object]) -> EvaluationSystem:
    files = _archive_files(path, case)
    try:
        metadata = json.loads(files["case.json"].decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("FLASH case metadata is unreadable") from exc
    strict_object(
        metadata,
        {
            "schema_version",
            "case_id",
            "family",
            "capture_kind",
            "n",
            "nnz",
            "relative_tolerance",
            "absolute_tolerance",
            "provenance_group",
            "source_run",
            "step",
            "solve_index",
            "matrix_sha256",
            "b_sha256",
            "x0_sha256",
            "files",
        },
        "FLASH case metadata",
    )
    expected = {
        "schema_version": 1,
        "case_id": case["case_id"],
        "family": "magnetic_diffusion_flash",
        "capture_kind": "scalar-magnetic-diffusion",
        "n": case["n"],
        "nnz": case["nnz"],
        "relative_tolerance": case["tolerance"],
        "absolute_tolerance": 0,
        "provenance_group": case["provenance_group"],
        "source_run": case["source_run"],
        "step": case["admission"]["step"],
        "solve_index": case["admission"]["solve_index"],
        "matrix_sha256": case["source"]["matrix_sha256"],
        "b_sha256": case["source"]["b_sha256"],
        "x0_sha256": case["source"]["x0_sha256"],
    }
    if any(metadata[key] != value for key, value in expected.items()):
        raise ValueError("FLASH case metadata disagrees with the frozen release")
    strict_object(
        metadata["files"], {"matrix.npz", "b.npy", "x0.npy"}, "FLASH member digests"
    )
    for name, claimed in metadata["files"].items():
        if hashlib.sha256(files[name]).hexdigest() != claimed:
            raise ValueError("FLASH file digest mismatch")
    matrix = _load_matrix(files["matrix.npz"], case["n"], case["nnz"])
    b = _load_npy(files["b.npy"], (case["n"],), {"f"})
    x0 = _load_npy(files["x0.npy"], (case["n"],), {"f"})
    public = MatrixInput(matrix=matrix, b=b, x0=x0, tolerance=case["tolerance"])
    if (
        matrix_digest(matrix) != metadata["matrix_sha256"]
        or vector_digest(public.b) != metadata["b_sha256"]
        or vector_digest(public.x0) != metadata["x0_sha256"]
    ):
        raise ValueError("FLASH canonical numerical digest mismatch")
    return EvaluationSystem(
        case_id=case["case_id"], public=public, x_star=None, reference_kind="none"
    )
