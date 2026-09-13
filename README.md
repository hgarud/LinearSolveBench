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
source .venv/bin/activate
uv pip install -e '.[dev]'
linear-solver-bench dataset families
linear-solver-bench dataset summary --release ns-mesh-cohort-v2-dev
linear-solver-bench dataset list --release ns-mesh-cohort-v2-dev
linear-solver-bench dataset prepare \
  --release ns-mesh-cohort-v2-dev --output data/prepared/ns-dev
linear-solver-bench runtime build --output build/runtime
linear-solver-bench candidate validate submissions/starter.c --runtime build/runtime
linear-solver-bench run submissions/starter.c \
  --runtime build/runtime --cases data/prepared/ns-dev \
  --output results/ns-dev.json
linear-solver-bench score results/ns-dev.json
```

NS cohort v2 contains 49 qualified SuiteSparse mesh/PDE matrices: 19 development
cases in nine provenance groups and 30 ranked cases in 15 disjoint groups. It retains the
original 43 matrices and adds six source-reviewed operators. A frozen
matrix-only policy covers every shared application, spatial method, size band,
structural regime, and method-by-size combination supported by multiple groups
on both sides. Semiconductor devices and boundary elements are development-only.
The [cohort review](docs/NS_COHORT_V2.md) publishes the selection, sources, and
remaining single-source joint gaps; it does not claim to cover all NS PDE
problems or equalize solver difficulty.

Every fixed workload meets the unchanged tenfold qualification margins, using
46 refined sparse-LU witnesses, one PyAMG/GMRES witness with inexact refinement,
and two exact row-dominance certificates. All 49 numerical preparations also
reproduce with the minimum supported NumPy/SciPy versions. Full Modal runs give
**11/19 development and 14/30 ranked** for the public GMRES+AMG candidate.
All 49 cases run once; the two 90-second timeouts remain in the results without
retry. The 1.6-million-row Transport case passes in 5.81 seconds within 4 GiB.
Partial coverage is a valid benchmark result and requires no all-case timing
reference.

The historical 43-case `ns-mesh-pilot` remains frozen with its original 11/32
partition and GMRES+AMG results of 10/11 development and 14/32 ranked. The
two-case `ns-mesh-dev-pilot` remains a separate integration check.

`--release` accepts an installed manifest name without `.json`, or an existing
manifest path such as `data/ns-mesh-cohort-v2-dev.json`. Optional `--family` and
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
The published release contains 96 development cases and 248 ranked cases, including
88 correctness controls. Each downloadable case contains `matrix.npz`, `b.npy`,
`x0.npy`, and `case.json`. The numerical inputs retain their captured values;
using them requires neither FLASH nor access to its source.

The manifests `flash-replay-dev-pilot` and `flash-replay-ranked-pilot` pin dataset
commit `3da5eda0ce3e85da1808d4b62d28d9111127e354`. Prepare either split with
`dataset prepare --release <manifest ID> --output <directory>`.

The fixed public reference `submissions/gmres_amg.c` passes all 344 cases in the
official Modal venue, including every correctness control. Candidate runs reuse
the frozen [development](data/releases/flash-replay-dev-reference.json) or
[ranked](data/releases/flash-replay-ranked-reference.json) reference timings;
there is no reference rerun for each submission. Follow the
[replay instructions](docs/OPERATIONS.md#run-and-score-flash-replay) to evaluate
and score a candidate with the matching calibration. Local timings remain
development results. See [validation evidence](docs/PILOT_VALIDATION.md),
[dataset details](docs/DATASET.md), and the
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
