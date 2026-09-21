"""Shared byte representations for persistent benchmark identities."""

import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()
