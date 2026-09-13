from __future__ import annotations

import struct

import numpy as np
import pytest

from linear_solver_bench.protocol import (
    INPUT_HEADER,
    OUTPUT_HEADER,
    OUTPUT_MAGIC,
    PROTOCOL_VERSION,
    decode_output,
    encode_input,
)


def output_payload(solution: np.ndarray, *, status: int = 0, mutated: int = 0) -> bytes:
    return (
        OUTPUT_HEADER.pack(
            OUTPUT_MAGIC,
            PROTOCOL_VERSION,
            status,
            solution.size,
            600,
            100,
            200,
            300,
            mutated,
        )
        + np.asarray(solution, dtype="<f8").tobytes()
    )


def test_protocol_layout_and_decode(small_system) -> None:
    encoded = encode_input(small_system.public)
    assert INPUT_HEADER.size == 40
    assert OUTPUT_HEADER.size == 64
    expected = 40 + 8 * 3 + 4 * 4 + 8 * 4 + 8 * 2 + 8 * 2
    assert len(encoded) == expected

    result = decode_output(output_payload(small_system.x_star), expected_n=2)
    assert result.status == 0
    assert result.elapsed_s == pytest.approx(6.0e-7)
    assert result.elapsed_s == result.create_s + result.setup_s + result.solve_s
    np.testing.assert_array_equal(result.solution, small_system.x_star)


@pytest.mark.parametrize(
    "payload",
    [
        b"short",
        output_payload(np.asarray([1.0, 2.0])) + b"trailing",
        struct.pack("<8s", b"BADMAGIC") + b"\0" * 72,
    ],
)
def test_protocol_rejects_malformed_output(payload: bytes) -> None:
    with pytest.raises(ValueError):
        decode_output(payload, expected_n=2)
