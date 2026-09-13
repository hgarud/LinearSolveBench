"""Keep shipped reference evidence bound to the actual public solver and registry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from linear_solver_bench.dataset import identity_sha256
from linear_solver_bench.pilot_scoring import (
    score_pilot_report,
    validate_replay_reference,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("split,count,controls", [("dev", 96, 0), ("ranked", 248, 88)])
def test_published_reference_is_complete_and_matches_public_build(
    split, count, controls
):
    artifact = ROOT / "data" / "releases" / f"flash-replay-{split}-reference.json"
    reference = json.loads(artifact.read_text())
    validate_replay_reference(reference)
    report = reference["reference_report"]
    runtime = json.loads((artifact.parent / "cpu-runtime-v2.json").read_text())
    runtime_digest = runtime.pop("manifest_sha256")
    assert identity_sha256(runtime) == runtime_digest
    assert report["runtime_manifest_sha256"] == runtime_digest
    assert (
        report["source_sha256"]
        == hashlib.sha256(
            (ROOT / "submissions" / "gmres_amg.c").read_bytes()
        ).hexdigest()
    )
    assert report["case_count"] == report["solved_count"] == count
    assert sum(case["role"] == "control" for case in report["cases"]) == controls
    assert all(not case["execution"]["diagnostics"] for case in report["cases"])
    score = score_pilot_report(report, reference)
    assert score["eligible"] and score["official"]
    assert score["reference_status"] == "registered"
    assert score["speedup"] == 1.0
