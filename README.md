# LinearSolverBench

LinearSolverBench measures the ability of an AI model or a coding agent to write fast, accurate, and general numerical solvers for sparse linear systems in C. The goal is to:

1. Encourage algorithmic advances in solving large sparse linear systems of equations.
2. Measure and improve the ability of AI models and systems at discovering algorithms.

## Motivation
Large systems of linear equations underpin some of the most important scientific and engineering problems across multiple domains. Some examples, together with problem families represented in the SuiteSparse Matrix Collection, include:

1. Power Grid Optimization, where large sparse linear systems form the core of numerical optimization algorithms for:
    * Optimal Power Flow (OPF)
    * Unit Commitment (UC), including LP/MILP relaxations and decomposition methods
2. Diffusive processes in Magnetic / Magneto-Inertial Confinement Fusion power devices:
    * Heat Conduction
    * Magnetic Diffusion
    * Radiation Transport
    * Alpha Energy Diffusion
3. Quantitative finance:
    * Multi-asset Black–Scholes, Heston, and Heston–Hull–White pricing PDEs, whose implicit discretizations require large sparse linear solves
    * American-option pricing under Black–Scholes or Heston models, where free-boundary methods repeatedly solve sparse linear systems
4. Structural mechanics and materials engineering:
    * Finite-element models of buildings, bridges, aircraft, plates, shells, and mechanical components
    * Buckling, fracture, contact, plasticity, and composite or porous-material simulations
5. Computational fluid dynamics, transport, and thermal science:
    * Navier–Stokes and Stokes flow, driven-cavity and coating-flow models, and atmospheric or ocean simulations
    * Heat transfer, combustion, diffusion, and coupled multiphysics discretizations
6. Electrical engineering, electromagnetics, acoustics, and semiconductor design:
    * Circuit and transistor simulation, power-network analysis, and semiconductor device or process modeling
    * Maxwell and Helmholtz equations, microwave structures, antennas, and vibroacoustic systems
7. Optimization, optimal control, and economic modeling:
    * Constraint matrices from linear programs and sparse KKT or Newton systems from nonlinear and constrained optimization
    * Model-predictive and optimal-control problems, economic models, and sparse least-squares systems
8. Imaging, computer vision, and inverse problems:
    * Tomographic and MRI reconstruction, image registration or segmentation, and other ill-posed inverse problems
    * Bundle adjustment, surface reconstruction, mesh deformation, and geometric optimization
9. Computational chemistry, physics, and reduced-order modeling:
    * Quantum chemistry, density-functional theory, lattice quantum chromodynamics, molecular structure, and protein models
    * Eigenvalue problems and reduced-order models for large dynamical systems
10. Graphs and network science:
    * Web, social, citation, collaboration, road, traffic, recommendation, and biological networks
    * Sparse adjacency, incidence, and graph-Laplacian matrices used for ranking, clustering, partitioning, and diffusion on networks

These applications do not all produce the same kind of matrix.
* PDE and finite-element problems may be symmetric positive definite, symmetric indefinite, or unsymmetric;
* flow, circuit, nonlinear-Jacobian, and KKT systems are often unsymmetric or saddle-point structured;
* least-squares and linear-programming data may be rectangular; and
* network data may encode a graph rather than an equation system.

Algorithmic progress in solving linear systems directly translates into advances in respective fields and is a net positive for society. Therefore, it is imperative to invest human and computational resources on discovering novel algorithms for our toughest problems. Large language models make it possible for the first time to do this at scale.

## The Benchmark

The language of choice for this benchmark is C with [HYPRE](https://github.com/hypre-space/hypre) as the main linear algebra library. HYPRE is one of the most widely used library, both in academia and industry, for building high-preformance solvers for large sparse linear systems and offers excellent low-level primitives, APIs, and stock preconditioning and solver algorithms that can be used to build novel solver algorithms. In other words, it offers the best batteries-included search space for a model or an agent without having to invent a new DSL for this task.

A submission is a single C source file built using an allow-listed set of low-level HYPRE primitives and stock preconditioners.
The `LinearSolverBench/native/include/nsl_hypre_solver.h` header file contains the list of primitives a submitted solver code is allowed to use.

The evaluation framework compiles the file once, runs it on linear systems derived from real [SuiteSparse Matrix Collection](https://sparse.tamu.edu/) matrices, verifies every returned solution independently, and reports both the number of systems solved and wall-clock time per system.

## Evaluation Tracks

## Install

```bash
uv venv --python 3.12
uv pip install -e '.[dev]'
```

Inspect the frozen dataset:

```bash
linear-solver-bench dataset summary
linear-solver-bench dataset list --split dev
```

Prepare public development matrices:

```bash
linear-solver-bench dataset prepare \
  --split dev \
  --output data/prepared/dev-v1
```

Build the pinned HYPRE runtime and validate a submission:

```bash
linear-solver-bench runtime build --output build/runtime
linear-solver-bench candidate validate submissions/starter.c \
  --runtime build/runtime
```

Run the public development set locally:

```bash
linear-solver-bench run submissions/starter.c \
  --runtime build/runtime \
  --cases data/prepared/dev-v1 \
  --output results/starter-dev.json
```

Official evaluations use `modal_app.py`, the ranked manifest, and an
operator-held RHS key. Public development uses the explicit public key in the
development manifest. The ranked key is never placed in candidate-visible
storage.

After preparing cases on the trusted operator machine, run the fixed venue:

```bash
modal run modal_app.py \
  --source submissions/starter.c \
  --cases data/prepared/dev-v1 \
  --output results/starter-modal-dev.json
```

The Modal image contains benchmark code and the pinned HYPRE runtime, but no
prepared archives. The local orchestrator streams one public case input at a
time into a single network-disabled Sandbox and verifies returned solutions
outside it.

## Submission boundary

Candidate source must include exactly `HYPRE_parcsr_ls.h` and export only:

```c
HYPRE_Int solver_create(HYPRE_Solver *solver,
                        HYPRE_Real relative_tolerance,
                        HYPRE_Real absolute_tolerance,
                        HYPRE_Int maximum_iterations);
```

> **Note:** Do not explicitly include `nsl_hypre_solver.h` in submitted solver
> code. The evaluator force-includes it during compilation using the equivalent
> of the following command:

```bash
cc -std=c17 -O3 -DNDEBUG -fno-omit-frame-pointer \
  -fvisibility=hidden \
  -include build/runtime/include/nsl_hypre_solver.h \
  -I build/runtime/include \
  -c policy.c \
  -o policy.o
```

The factory returns a HYPRE solver object with setup, solve, and destroy
callbacks. Candidate objects own and release all allocations and
preconditioners they create. Stock outer solvers are deliberately absent from
the allowlist; approved stock preconditioners are available as components.

See [docs/SPEC.md](docs/SPEC.md) for the complete scientific contract and
[docs/DATASET.md](docs/DATASET.md) for provenance and preparation. Evaluator
operators should also follow [docs/OPERATIONS.md](docs/OPERATIONS.md).

## Project status

This repository implements the standalone v1 contract and evaluation path.
Before opening the official leaderboard, operators must run venue calibration,
freeze its content-addressed timing artifact, and prepare the ranked systems
with an operator-held key. Prepared systems remain on the trusted operator
machine and are not baked into an image or mounted into the Sandbox.

## License

Benchmark code is Apache-2.0. HYPRE-derived declarations retain HYPRE's MIT
notice. SuiteSparse metadata and matrices are CC BY 4.0; matrix-specific
metadata and citations must be retained.
