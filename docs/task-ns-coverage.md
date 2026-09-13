# NS mesh coverage: candidate task

Write one C solver that accurately solves as many supplied nonsymmetric mesh/PDE
linear systems as possible within the release's fixed per-case budgets. NS means
nonsymmetric; the systems are not restricted to Navier–Stokes equations. The
initial public development pilot has two matrices and does not define a complete
family inventory.

Submit a single UTF-8 C source file, at most 65,536 bytes. Include exactly
`HYPRE_parcsr_ls.h` and export only:

```c
HYPRE_Int solver_create(HYPRE_Solver *solver,
                        HYPRE_Real relative_tolerance,
                        HYPRE_Real absolute_tolerance,
                        HYPRE_Int maximum_iterations);
```

The evaluator force-includes the public solver header. Use its approved HYPRE
vector/matrix operations and preconditioners to implement setup, solve, and
destroy callbacks. Stock outer solvers, I/O, clocks, networking, process
creation, and dynamic linking are unavailable. Free all owned workspace and
preconditioners in destruction. `submissions/starter.c` demonstrates the ABI.

The solver receives a real square float64 sparse matrix `A`, a vector `b`, and
zero initial guess. Use the supplied relative tolerance and iteration limit;
the development release requests `1e-12` and 5,000 iterations. Absolute tolerance
is zero. Preserve `A,b` and write the solution to the supplied solution vector.
Case names, provenance, and manufactured targets are not supplied to the solver.

Trusted verification computes `r = b - A x` and requires all four limits:

```text
||r||inf / (||A||inf ||x||inf + ||b||inf) <= 1e-10
max_i |r_i| / (|A||x| + |b|)_i <= 1e-8
||x-x_target||2 / ||x_target||2 <= 1e-5
||x-x_target||inf / ||x_target||inf <= 1e-5
```

The target has entries ±1, with `b = fl(A x_target)`. Offline qualification
accounts for formation uncertainty; candidate acceptance always checks the
stored target. Relative residual is diagnostic. A solver's stopping declaration
or requested tolerance does not replace these fixed acceptance gates. Output
must be finite, correctly shaped, successful, and within the deadline.

Your source is compiled once and each case runs exactly once in a fresh native
process. Factory creation, setup, preconditioner construction, and solve all
count toward its measured solve time. No state persists between cases.

The score is the number of cases passing every required check. Equal counts tie;
timing does not break ties. All expected cases are evaluated, including after a
solver failure. No timing reference or calibration is used for coverage.

See [SPEC.md](SPEC.md) for the full contract and [README.md](../README.md) for
preparation and execution commands.
