# Benchmark specification

## Scope

The v1 track evaluates a single-process, float64, nonsymmetric sparse linear
solver. It is intended to measure algorithm construction rather than selection
among stock outer solvers. A future `open-hypre` track may admit the complete
public HYPRE solver API, and future distributed tracks may change MPI and venue
contracts without altering v1.

## Candidate artifact

One regular file named `policy.c` is accepted. It must be UTF-8 with LF line
endings, no more than 65,536 bytes, contain exactly one include directive for
`HYPRE_parcsr_ls.h`, and define exactly one global export, `solver_create`.

The object file's undefined symbols are audited before linking. The allowlist
contains selected native Krylov vector, reduction, and matvec operations;
matrix shape queries; standard C memory/math intrinsics emitted by the pinned
compiler; and approved HYPRE preconditioner lifecycle and scalar configuration
calls. It excludes I/O, clocks, processes, networking, dynamic linking, stock
outer solvers, and configuration parsers.

The operating-system sandbox remains a second security boundary. Link-time
auditing is a capability restriction, not a replacement for isolation.

## Input system

Every system has a canonical square float64 CSR matrix `A`, right-hand side
`b`, zero initial guess `x0`, public tolerance `tau`, and evaluator-only exact
solution `x_star`. Ranked systems use a fixed operator-held HMAC key to derive
`x_star`; development systems use a published key. In both cases:

```text
x_star[j] is either -1 or +1
b = A @ x_star
x0 = 0
```

Only `A`, `b`, `x0`, and `tau` reach candidate-controlled code. Matrix names,
case IDs, source URLs, digests, and `x_star` do not.

## Timed boundary

Matrix decoding and construction of trusted HYPRE IJ/ParCSR objects are not
timed. The monotonic timer covers:

1. `solver_create`;
2. the candidate setup callback; and
3. the candidate solve callback.

Candidate destruction, solution extraction, mutation checks, and numerical
verification occur after the timer. Each repetition is a fresh process. A case
passes only if all three repetitions return status zero and pass every trusted
gate before the case deadline.

## Accuracy

For residual `r = b - A x`, the evaluator computes:

```text
eta_norm = ||r||inf / (||A||inf ||x||inf + ||b||inf)
eta_comp = max_i |r_i| / (|A||x| + |b|)_i
eta_fwd  = max(||x-x_star||2/||x_star||2,
               ||x-x_star||inf/||x_star||inf)
```

Safe zero denominators use the conventional exact-zero result; a nonzero
numerator over a zero denominator is infinity. For public tolerance `tau`, the
limits are:

```text
eta_norm <= max(tau, 20 eps)
eta_comp <= max(5 tau, 100 eps)
eta_fwd  <= min(2e-1, max(5 tau, 2e-12))
```

Each ranked case uses the smallest published tolerance tier in `1e-4`, `1e-3`,
`1e-2`, and `4e-2` supported by its frozen condition estimate.
The uniform 1-norm condition estimate selects `tau`; it does not alter or
replace any of the direct accuracy calculations above.

The output must also have the expected shape and contain only finite values.
The trusted driver checks that candidate setup/solve did not change the matrix
or right-hand side.

## Results and ranking

Every case records all repetition outcomes, trusted metrics, phase timings,
failure reason, source artifact hash, runtime identity, and venue identity.

The leaderboard is lexicographic:

1. maximize the number of completely solved cases;
2. minimize the penalized geometric mean across all cases.

For the tie-breaker, a solved case contributes its median wall time divided by
the case's calibrated reference time. An unsolved case contributes twice its
deadline divided by that reference time. Calibration is a separately frozen,
content-addressed artifact and cannot change within a benchmark version.

## Venue

The official v1 venue is a network-disabled Modal Sandbox with a request and
hard limit of 2.0 physical CPU cores and 4096 MiB memory. HYPRE is the pinned
sequential float64/int32 static build, MPI has one rank, and OpenMP is disabled.
One sandbox evaluates one submission; individual cases run in child processes.
