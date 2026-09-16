from linear_solver_bench.assets import asset_root


def test_runtime_assets_are_available():
    root = asset_root()
    assert (root / "families" / "ns_mesh_pde" / "dev.json").is_file()
    assert (root / "native" / "src" / "driver.cpp").is_file()
    assert (root / "reference" / "solver.c").is_file()
    assert (root / "examples" / "solver.c").is_file()
