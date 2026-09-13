"""Bounded child-process execution with combined diagnostics."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    output: bytes
    wall_s: float


class ProcessTimeout(RuntimeError):
    def __init__(self, timeout_s: float, output: bytes):
        self.timeout_s = timeout_s
        self.output = output
        super().__init__(f"process exceeded {timeout_s:g} seconds")


def bounded_process(
    command: Sequence[str],
    *,
    timeout_s: float,
    env: Mapping[str, str] | None = None,
) -> ProcessResult:
    if not command or timeout_s <= 0.0:
        raise ValueError("command and positive timeout are required")
    start = time.monotonic()
    try:
        completed = subprocess.run(
            tuple(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout_s,
            env=dict(env) if env is not None else None,
        )
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or b"") + (exc.stderr or b"")
        raise ProcessTimeout(timeout_s, output) from exc
    return ProcessResult(
        completed.returncode, completed.stdout, time.monotonic() - start
    )
