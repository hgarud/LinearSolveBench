"""Read original SuiteSparse Matrix Market operators without changing the problem."""

from __future__ import annotations

import pathlib
import tarfile
from collections.abc import Mapping

import numpy as np
from scipy import sparse
from scipy.io import mmread

from ..models import CsrMatrix
from ..workloads import matrix_digest

MAX_MEMBERS = 128
MAX_EXPANDED_BYTES = 8 * 1024**3


def load_suitesparse(path: pathlib.Path, case: Mapping[str, object]) -> CsrMatrix:
    """Validate member paths, declared dimensions, and canonical matrix identity."""
    source = case["source"]
    expected = f"{source['name']}/{source['name']}.mtx"
    matrix = None
    seen = set()
    expanded = 0
    try:
        with tarfile.open(path, "r:gz") as archive:
            for count, member in enumerate(archive, start=1):
                name = pathlib.PurePosixPath(member.name)
                expanded += member.size
                if (
                    count > MAX_MEMBERS
                    or expanded > MAX_EXPANDED_BYTES
                    or member.size > 4 * 1024**3
                    or member.size < 0
                    or name.is_absolute()
                    or not name.parts
                    or ".." in name.parts
                    or member.name in seen
                    or not (member.isfile() or member.isdir())
                ):
                    raise ValueError("unsafe SuiteSparse archive member")
                seen.add(member.name)
                if member.name != expected:
                    continue
                if not member.isfile():
                    raise ValueError("expected Matrix Market member is not a file")
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError("Matrix Market member is unreadable")
                with handle:
                    header = handle.readline(1024).decode("ascii").lower().split()
                    if (
                        len(header) != 5
                        or header[:3] != ["%%matrixmarket", "matrix", "coordinate"]
                        or header[3] not in {"real", "integer", "pattern"}
                    ):
                        raise ValueError(
                            "SuiteSparse input must be a real sparse coordinate matrix"
                        )
                    line = handle.readline(4096)
                    while line.startswith(b"%"):
                        line = handle.readline(4096)
                    dimensions = [int(item) for item in line.split()]
                    if (
                        len(dimensions) != 3
                        or dimensions[:2] != [case["n"], case["n"]]
                        or not 0 < dimensions[2] <= member.size // 3
                    ):
                        raise ValueError(
                            "Matrix Market dimensions disagree with the declared case"
                        )
                    handle.seek(0)
                    loaded = mmread(handle, spmatrix=True)
                if not sparse.issparse(loaded) or np.iscomplexobj(loaded.data):
                    raise ValueError("SuiteSparse input must be real and sparse")
                matrix = CsrMatrix.from_scipy(loaded)
    except (OSError, tarfile.TarError, UnicodeError) as exc:
        raise ValueError("SuiteSparse archive is unreadable") from exc
    if matrix is None:
        raise ValueError("original Matrix Market operator is missing from archive")
    if (
        matrix.n != case["n"]
        or matrix.nnz != case["nnz"]
        or matrix_digest(matrix) != source["matrix_sha256"]
    ):
        raise ValueError(
            "canonical matrix disagrees with the release digest or dimensions"
        )
    # Scientific provenance is admitted in the public manifest. This independent
    # numerical check prevents symmetric operators being relabelled NS mesh.
    operator = matrix.to_scipy()
    scale = float(np.max(np.abs(operator.data), initial=0))
    difference = operator - operator.T
    asymmetry = float(np.max(np.abs(difference.data), initial=0))
    if scale == 0 or asymmetry <= 1e-12 * scale:
        raise ValueError("NS mesh requires a materially nonsymmetric operator")
    return matrix
