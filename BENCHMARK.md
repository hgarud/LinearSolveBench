# Benchmark details

## Candidate solver interface

A candidate is one UTF-8 C file, includes exactly
`HYPRE_parcsr_ls.h` and exports:

```c
HYPRE_Int solver_create(HYPRE_Solver *solver,
                        HYPRE_Real relative_tolerance,
                        HYPRE_Real absolute_tolerance);
```

`solver_create` acts as a factory.
It does not return the solver directly but writes a HYPRE_Solver handle into *solver.
That handle represents an object containing three function pointers:

- `setup`: prepares solver state for matrix \(A\), such as constructing a preconditioner.
- `solve`: solves $Ax=b$ and writes into the supplied solution vector.
- `destroy`: releases all memory and HYPRE objects owned by the solver.
The driver uses them in this order:

```
solver_create(...)
HYPRE_SolverSetup(...)
HYPRE_SolverSolve(...)
HYPRE_SolverDestroy(...)
```

See [driver.cpp](native/src/driver.cpp) and the callback layout in
[nsl_hypre_solver.h](native/include/nsl_hypre_solver.h).

A custom solver can implement its own object. The reference demonstrates this by assigning:
```
data->base.setup = reference_setup;
data->base.solve = reference_solve;
data->base.destroy = reference_destroy;
*solver = (HYPRE_Solver) data;
```

See [reference/solver.c](reference/solver.c).

Candidates are not allowed to perform I/O, networking, process creation, dynamic loading, or timing. Inputs must not be modified.

The permitted external functions are listed in
[native/allowed-symbols.txt](native/allowed-symbols.txt).


## Execution and scoring for each family

For `ns_mesh_pde`, the evaluator builds and runs only the candidate. The score is
the number of cases in the development set that pass the accuracy threshold.

For `magnetic_diffusion_flash`, the reference and candidate are built against
the same HYPRE runtime inside one environment. Each numerical case is run in two fresh processes:

1. Run the reference in a fresh process.
2. Run the candidate in a fresh process.
3. Independently verify both returned solutions on the host, then compute
   `speedup = reference_seconds / candidate_seconds` if both pass.

A failed fixed reference invalidates the evaluation. A failed candidate case
has no speedup and prevents a complete aggregate speedup score.

Timing measures solver creation, setup, and solve. It excludes
compilation, input loading, transport, destruction, and numerical verification.
For magnetic diffusion, the reference is measured again for every candidate in each new sandbox to make sure the timing is comparable.

The complete magnetic diffusion score is the geometric mean of per-case
speedups.


## Accuracy

For residual `r = b - A x`, the evaluator computes normwise and
componentwise backward error and the relative residual. `ns_mesh_pde` cases also compare against the manufactured target.

`ns_mesh_pde` requires:

- normwise backward error ≤ `1e-10`;
- componentwise backward error ≤ `1e-8`;
- relative L2 and Linf forward errors ≤ `1e-5`.

`magnetic_diffusion_flash` requires:

- normwise backward error ≤ `max(1e-10, 20 * float64_epsilon)`;
- relative residual ≤ `1e-10 * (1 + 1e-6)`.

Both families require a finite solution of the correct size, a successful solver
status, and unchanged inputs.


## Modal sandboxes for evaluation

Each Modal sandbox receives the CPU and memory specified by its family manifest.
The requested amount is also the enforced limit.

| Family | CPUs | Memory |
| --- | ---: | ---: |
| `ns_mesh_pde` | 2 | 10 GiB |
| `magnetic_diffusion_flash` | 2 | 4 GiB |
