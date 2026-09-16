# LinearSolverBench

LinearSolverBench measures the correctness and speed of C/HYPRE linear solvers.
Every candidate is compared with a fixed reference solver in the same process
environment—or the same Modal CPU sandbox—so results are normalized across
different hosts.

The benchmark has two public development families:

| Family | Input systems |
| --- | --- |
| `ns_mesh_pde` | Nonsymmetric SuiteSparse mesh and PDE matrices with manufactured right-hand sides |
| `flash` | Captured FLASH magnetic-diffusion matrices, right-hand sides, and initial guesses |

## Quick start

Python 3.12, a C/C++ compiler, CMake, Ninja, Git, and `nm` are required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

linear-solver-bench init solver.c
linear-solver-bench run solver.c --family ns_mesh_pde --venue local \
  --case ss-225-rademacher-0
```

The local venue is a development smoke test. Its first run downloads the selected
inputs and builds the pinned HYPRE runtime; both are cached under
`~/.cache/linear-solver-bench`.

## Standardized Modal evaluation

Install the Modal extra and configure your Modal account, then select the Modal
venue:

```bash
python -m pip install -e '.[modal]'
linear-solver-bench run solver.c --family ns_mesh_pde --venue modal \
  --output result.json
```

Modal is the default and the only official, cross-candidate venue. The reference
and candidate are compiled once in one network-disabled sandbox.
For every case, the evaluator launches a fresh reference process followed by a
fresh candidate process and independently checks both solutions. A passing case
reports:

```text
speedup = reference seconds / candidate seconds
```

The report records the immutable image, runtime, reference, candidate, and
manifest identities. Exact sandbox hardware is not part of the score because the
reference is remeasured beside every candidate.

> **FLASH data status:** publishing the dev-only FLASH archive at the URLs pinned
> by its manifest remains a release prerequisite. The implementation and tests do
> not require private or mixed-split data.

> **Reference status:** the fixed reference compiles and passes native smoke tests
> plus the first 10 NS cases. Full NS and FLASH qualification is still a release
> gate; the current ILUT reference is too slow on `ss-896` to accept yet.

## Write a solver

Candidates are one C file exporting `solver_create`. Start with
[`examples/solver.c`](examples/solver.c) and read [BENCHMARK.md](BENCHMARK.md) for
the ABI, allowed HYPRE operations, timing boundary, and accuracy rules.

## Submit

```bash
linear-solver-bench submit solver.c --name my-solver \
  --family ns_mesh_pde --license Apache-2.0 --submitter 'Your Name' \
  --output submissions/my-solver
```

Commit the generated three-file directory and open a pull request. Private test
data and result publication are maintained outside this participant repository.

## Code map

The normal path is deliberately short:

```text
CLI/API -> benchmark.py -> data.py -> build.py or modal.py -> verify.py
```

Family-specific input recipes live in `families.py`; native build and protocol
code live in `build.py`. There is no plugin system, compatibility layer, training
framework, or public operator service.

## License

Benchmark code is Apache-2.0. HYPRE-derived declarations retain their MIT notice.
SuiteSparse matrices retain their original attribution; see the benchmark-family
README files.
