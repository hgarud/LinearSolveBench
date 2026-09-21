"""Native wire format and raw execution evidence limits."""

import struct

INPUT_MAGIC = b"LSBIN001"
OUTPUT_MAGIC = b"LSBOUT01"
PROTOCOL_VERSION = 1
INPUT_HEADER = struct.Struct("<8sIIQQd")
OUTPUT_HEADER = struct.Struct("<8sIiQQQQQB7x")
DIAGNOSTIC_LIMIT = 8_000


def output_size(n: int) -> int:
    return OUTPUT_HEADER.size + n * 8
