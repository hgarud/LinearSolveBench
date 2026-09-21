# LinearSolverBench

LinearSolverBench measures the ability of an AI model or a harness to write
fast, accurate, and general numerical solvers for large sparse linear systems in
C. The goal is to:

1. Encourage algorithmic advances in solving large sparse linear systems of
   equations.
2. Measure and improve the ability of AI models and systems at discovering
   algorithms.

## Motivation

Large systems of linear equations underpin some of the most important scientific
and engineering problems across multiple domains. Some examples, together with
problem families represented in the
[SuiteSparse Matrix Collection](https://sparse.tamu.edu/), include (but not
limited to):

<details>
<summary>1. Power Grid Optimization, where large sparse linear systems form the core of numerical optimization algorithms</summary>

- Optimal Power Flow (OPF)
- Unit Commitment (UC), including LP/MILP relaxations and decomposition methods

</details>

<details>
<summary>2. Diffusive processes in Magnetic / Magneto-Inertial Confinement Fusion power devices</summary>

- Heat Conduction
- Magnetic Diffusion
- Radiation Transport
- Alpha Energy Diffusion

</details>

<details>
<summary>3. Quantitative finance</summary>

- Multi-asset Black–Scholes, Heston, and Heston–Hull–White pricing PDEs, whose
  implicit discretizations require large sparse linear solves
- American-option pricing under Black–Scholes or Heston models, where
  free-boundary methods repeatedly solve sparse linear systems

</details>

<details>
<summary>4. Structural mechanics and materials engineering</summary>

- Finite-element models of buildings, bridges, aircraft, plates, shells, and
  mechanical components
- Buckling, fracture, contact, plasticity, and composite or porous-material
  simulations

</details>

<details>
<summary>5. Computational fluid dynamics, transport, and thermal science</summary>

- Navier–Stokes and Stokes flow, driven-cavity and coating-flow models, and
  atmospheric or ocean simulations
- Heat transfer, combustion, diffusion, and coupled multiphysics discretizations

</details>

These applications do not all produce the same kind of matrices.

* PDE and finite-element problems may be symmetric positive definite, symmetric
  indefinite, or unsymmetric;
* flow, circuit, nonlinear-Jacobian, and KKT systems are often unsymmetric or
  saddle-point structured;
* least-squares and linear-programming data may be rectangular; and
* network data may encode a graph rather than an equation system.

Algorithmic progress in solving linear systems directly translates into advances
in respective fields and is a net positive for society. Therefore, it is
imperative to invest human and computational resources on discovering novel
algorithms for our toughest problems. Large language models make it possible for
the first time to do this at scale.

## The Benchmark

The language of choice is C with
[HYPRE](https://github.com/hypre-space/hypre) as the main linear algebra
library.
HYPRE is one of the most widely used library, both in academia and industry, for
building high-preformance solvers for large sparse linear systems and offers
excellent low-level primitives, APIs, and stock preconditioning and solver
algorithms that can be used to build novel solver algorithms. In other words, it
offers the best batteries-included search space for a model or an agent without
having to invent a new DSL for this task.

The benchmark currently supports the following families of matrices:

<table>
  <thead>
    <tr>
      <th>Family</th>
      <th>Input systems</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>ns_mesh_pde</code></td>
      <td>
        Nonsymmetric SuiteSparse mesh and PDE matrices; scored by the number of
        cases solved
      </td>
    </tr>
    <tr>
      <td><code>magnetic_diffusion_flash</code></td>
      <td>
        Captured magnetic diffusion replays from FLASH; scored by speedup over the
        fixed reference
      </td>
    </tr>
  </tbody>
</table>

## Quick start

Python 3.12, a C/C++ compiler, CMake, Ninja, Git, and `nm` are required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

linear-solver-bench run solver.c --family ns_mesh_pde --venue local \
  --case ss-225-rademacher-0
```

Here, `solver.c` is your candidate source file. The fixed solver at
[`reference/solver.c`](reference/solver.c) shows the interface but uses HYPRE
operations reserved for the reference, so it cannot be run as a candidate.

The local venue is mainly for testing. Its first run downloads the selected
inputs and builds the HYPRE runtime; both are cached under
`~/.cache/linear-solver-bench`.

## Evaluating a candidate solver

Configure your Modal account, then select the Modal venue:

```bash
linear-solver-bench run solver.c --family ns_mesh_pde --venue modal \
  --output result.json

linear-solver-bench run solver.c --family magnetic_diffusion_flash --venue modal \
  --output magnetic-diffusion-result.json
```

Candidates are one C file exporting `solver_create`. See
[`reference/solver.c`](reference/solver.c) for the interface and read
[BENCHMARK.md](BENCHMARK.md) for the ABI, allowed HYPRE operations, timing
boundary, and accuracy rules.

## Submit

```bash
linear-solver-bench submit solver.c --name my-solver \
  --family ns_mesh_pde --submitter 'Your Name' \
  --output my-solver-submission
```

Commit the generated three-file directory and open a pull request. Private test
data and leaderboard are maintained outside this repository.

## License

Benchmark code is Apache-2.0. HYPRE-derived declarations retain their MIT
notice.
SuiteSparse matrices retain their original attribution; see the benchmark-family
README files.
