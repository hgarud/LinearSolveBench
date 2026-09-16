from linear_solver_bench.cli import main, parser


def test_modal_is_the_default_venue():
    args = parser().parse_args(["run", "solver.c", "--family", "ns_mesh_pde"])
    assert args.venue == "modal"


def test_init_copies_the_starter(tmp_path):
    output = tmp_path / "solver.c"

    assert main(["init", str(output)]) == 0

    assert output.is_file()
    assert "solver_create" in output.read_text(encoding="utf-8")


def test_init_refuses_to_overwrite(tmp_path, capsys):
    output = tmp_path / "solver.c"
    output.write_text("mine", encoding="utf-8")

    assert main(["init", str(output)]) == 2

    assert output.read_text(encoding="utf-8") == "mine"
    assert "already exists" in capsys.readouterr().err
