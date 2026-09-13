# FLASH replay: candidate task

Write one C solver that accurately solves captured scalar magnetic-diffusion
systems as quickly as possible. Each case is an independent replay of one
captured solve. Factory and preconditioner setup costs count toward speedup.
No solver state carries between cases, and the simulation is not rerun.

Submit a single UTF-8 C source file, at most 65,536 bytes. Include exactly
`HYPRE_parcsr_ls.h` and export only:

```c
HYPRE_Int solver_create(HYPRE_Solver *solver,
                        HYPRE_Real relative_tolerance,
                        HYPRE_Real absolute_tolerance,
                        HYPRE_Int maximum_iterations);
```

The evaluator force-includes the public solver header. Implement setup, solve,
and destroy callbacks using the approved HYPRE operations and preconditioners.
Stock outer solvers, I/O, clocks, networking, process creation, and dynamic
linking are unavailable. Release all owned workspace and preconditioners in
destruction. `submissions/starter.c` demonstrates the ABI.

Each input preserves the captured float64 matrix `A`, right-hand side `b`,
initial guess `x0`, and relative tolerance `tau`. The initial guess may be useful;
do not assume it is zero. The factory receives the release's iteration limit
and absolute tolerance zero. Preserve `A,b` and write the solution to the
supplied solution vector. Case IDs, source-run names, scored/control roles,
weights, and numerical reference vectors are not supplied to candidate code.

For `r = b - A x`, trusted verification requires:

```text
||r||2 / ||b||2 <= tau * (1 + 1e-6)
||r||inf / (||A||inf ||x||inf + ||b||inf) <= max(tau, 20 eps)
```

`eps` is float64 machine epsilon. With a zero RHS, relative residual is zero
only when the residual is exactly zero. Output must have the expected shape,
be finite, return success, leave inputs unchanged, and complete within its
fixed deadline. Componentwise backward error is diagnostic, including for weak
rows. Forward errors against any available numerical reference are also
optional diagnostics, not acceptance gates.

The evaluator compiles the source once and runs each case once in one fresh
native process. The measured time includes factory creation, setup, and solve.
Each scored case's speedup is qualified reference time divided by candidate
time. The aggregate is a weighted geometric mean: equal mass per provenance
group, then source run, then scored case within that run.

Every scored case and mandatory correctness control must pass for aggregate
speedup eligibility. Controls have no aggregate timing weight. One failed case
makes speedup null; successful subsets are not averaged. The benchmark reuses
frozen timings from one reference solver qualified on the complete split.
Until that reference exists, reports provide accuracy diagnostics and score as
`awaiting_reference`.

The public dataset and its timing reference are separate release artifacts.
See [DATASET.md](DATASET.md) for download formats and [OPERATIONS.md](OPERATIONS.md)
for qualification and scoring commands.
