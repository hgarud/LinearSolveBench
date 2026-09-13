# LinearSolverBench

LinearSolverBench measures how well AI models and systems can produce accurate,
fast solvers for sparse linear systems. A submission is one C source file using
approved [HYPRE](https://github.com/hypre-space/hypre) operations and
preconditioners. The evaluator compiles it once, runs independent cases, and
checks each returned solution against the original matrix and right-hand side.

The pilot supports two separate tracks:

| Family | Track | Result |
| --- | --- | --- |
| `ns-mesh-pde` | `coverage` | Number of cases solved accurately within fixed budgets; equal counts tie |
| `magnetic_diffusion_flash` | `replay` | Speedup for individual captured solves against one qualified reference |

Each pilot case runs **once in one fresh native process**. Setup and solve time
both count. NS coverage needs no timing reference. FLASH replay requires every
scored case and correctness control to pass before reporting aggregate speedup.
SPD, NS performance, and FLASH trajectory tracks are not supported yet. The
original SuiteSparse v1 benchmark remains available under its separate identity
and retains three repetitions per case.

## Get started with NS coverage

From a checkout:

```bash
uv venv --python 3.12
uv pip install -e '.[dev]'
linear-solver-bench dataset families
linear-solver-bench dataset summary --release ns-mesh-dev-pilot
linear-solver-bench dataset list --release ns-mesh-dev-pilot
linear-solver-bench dataset prepare \
  --release ns-mesh-dev-pilot --output data/prepared/ns-dev
linear-solver-bench runtime build --output build/runtime
linear-solver-bench candidate validate submissions/starter.c --runtime build/runtime
linear-solver-bench run submissions/starter.c \
  --runtime build/runtime --cases data/prepared/ns-dev \
  --output results/ns-dev.json
linear-solver-bench score results/ns-dev.json
```

The development pilot contains two real mesh/PDE matrices: `DRIVCAV/cavity01`
(317 unknowns) and `FEMLAB/poisson2D` (367 unknowns). It is an integration and
solver-development set, not a full NS family inventory. Preparation downloads
original SuiteSparse matrices and generates one deterministic manufactured
right-hand side per matrix.

`--release` accepts an installed manifest name without `.json`, or an existing
manifest path such as `data/ns-mesh-dev-pilot.json`. Optional `--family` and
`--track` selectors must agree with that release; `ns_mesh_pde` is an accepted
alias for `ns-mesh-pde`. The `run` command selects the evaluator from the prepared
manifest, so it needs no repeated family or track flags.

Local runs are labeled `local-uncontrolled`; their CPU and memory allocations
are not an official evaluation venue. For isolated execution from a checkout:

```bash
modal run modal_app.py \
  --source submissions/starter.c --cases data/prepared/ns-dev \
  --output results/ns-modal-dev.json
linear-solver-bench score results/ns-modal-dev.json
```

The trusted evaluator loads cases one at a time. The Modal route streams only
the current numerical input into a network-disabled sandbox and verifies the
returned solution outside it. See [operator instructions](docs/OPERATIONS.md)
for trusted release checks, resource limits, and reference calibration.

## FLASH replay data

FLASH cases are available from the public, ungated
[LinearSolveBench dataset on Hugging Face](https://huggingface.co/datasets/hgarud/LinearSolveBench).
The planned release contains 96 development cases and 248 ranked cases, including
88 correctness controls. Each downloadable case contains `matrix.npz`, `b.npy`,
`x0.npy`, and `case.json`. The numerical inputs retain their captured values;
using them requires neither FLASH nor access to its source.

The manifests `flash-replay-dev-pilot` and `flash-replay-ranked-pilot` pin dataset
commit `3da5eda0ce3e85da1808d4b62d28d9111127e354`. Prepare either split with
`dataset prepare --release <manifest ID> --output <directory>`.

Publication and qualification of a public timing reference are separate steps.
Use a published, commit-pinned release manifest once available. A replay report
without a matching qualified reference has `speedup: null` and reason
`awaiting_reference`. See [dataset details](docs/DATASET.md) and the
[FLASH task statement](docs/task-flash-replay.md).

## Submit a solver

Candidate source includes exactly `HYPRE_parcsr_ls.h` and exports only:

```c
HYPRE_Int solver_create(HYPRE_Solver *solver,
                        HYPRE_Real relative_tolerance,
                        HYPRE_Real absolute_tolerance,
                        HYPRE_Int maximum_iterations);
```

The evaluator force-includes `native/include/nsl_hypre_solver.h`; candidates
must not include it explicitly. The factory provides setup, solve, and destroy
callbacks and owns its allocations and preconditioners. Approved HYPRE
preconditioners are building blocks; stock outer solvers are excluded from the
candidate allowlist. Only numerical inputs reach candidate code, without case
names, provenance, manufactured targets, or numerical reference vectors.

Read the [specification](docs/SPEC.md),
[NS coverage task statement](docs/task-ns-coverage.md), and
[FLASH replay task statement](docs/task-flash-replay.md) before submitting.

## Existing v1 benchmark

Commands without `--release` keep their existing v1 meaning:

```bash
linear-solver-bench dataset summary
linear-solver-bench dataset prepare --split dev --output data/prepared/dev-v1
```

V1 has its own accuracy, repetition, qualification, and scoring rules. A pilot
selector cannot reinterpret v1 prepared data. Its 203-case ranked inventory is
not an NS mesh coverage release.

## License

Benchmark code is Apache-2.0. HYPRE-derived declarations retain HYPRE's MIT
notice. Preserve SuiteSparse metadata and matrix-specific attribution when
using its matrices; see [dataset attribution](docs/DATASET.md). The Hugging Face
dataset has its own dataset card and license metadata.
