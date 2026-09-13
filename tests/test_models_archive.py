from __future__ import annotations

import io

import numpy as np
import pytest

from linear_solver_bench.archive import decode_system, encode_system
from linear_solver_bench.models import CsrMatrix


def test_archive_round_trip(small_system) -> None:
    decoded = decode_system(encode_system(small_system))
    assert decoded.case_id == small_system.case_id
    assert decoded.public.tolerance == small_system.public.tolerance
    np.testing.assert_array_equal(
        decoded.public.matrix.row_offsets, small_system.public.matrix.row_offsets
    )
    np.testing.assert_array_equal(decoded.public.b, small_system.public.b)
    np.testing.assert_array_equal(decoded.x_star, small_system.x_star)
    assert not decoded.x_star.flags.writeable


def test_archive_rejects_extra_fields(small_system) -> None:
    payload = encode_system(small_system)
    with np.load(io.BytesIO(payload), allow_pickle=False) as original:
        fields = {name: original[name] for name in original.files}
    fields["unexpected"] = np.asarray(1)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **fields)
    with pytest.raises(ValueError, match="schema"):
        decode_system(buffer.getvalue())


def test_csr_rejects_duplicate_or_unsorted_columns() -> None:
    with pytest.raises(ValueError, match="strictly sorted"):
        CsrMatrix(
            n=2,
            row_offsets=np.asarray([0, 2, 2], dtype=np.uint64),
            column_indices=np.asarray([1, 1], dtype=np.uint32),
            values=np.asarray([1.0, 2.0], dtype=np.float64),
        )
