from __future__ import annotations

import pathlib


def test_modal_venue_has_hard_limits_and_excludes_trusted_data() -> None:
    source = pathlib.Path("modal_app.py").read_text(encoding="utf-8")
    assert "cpu=(2.0, 2.0)" in source
    assert "memory=(4096, 4096)" in source
    assert "block_network=True" in source
    assert "MAXIMUM_SANDBOX_LIFETIME_S = 24 * 60 * 60" in source
    assert ".add_local_dir(ROOT, REMOTE_BENCHMARK" not in source
    assert '.add_local_dir(ROOT / "src"' in source
    assert '.add_local_dir(ROOT / "native"' in source
    assert '.add_local_dir(ROOT / "data"' not in source
    assert '.env({"LINEAR_SOLVER_BENCH_ROOT": REMOTE_BENCHMARK})' in source
