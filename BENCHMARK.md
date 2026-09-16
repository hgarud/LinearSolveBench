# Benchmark contract

## Candidate interface

A candidate is one UTF-8 C file of at most 65,536 bytes. It includes exactly
`HYPRE_parcsr_ls.h` and exports:

```c
HYPRE_Int solver_create(HYPRE_Solver *solver,
                        HYPRE_Real relative_tolerance,
                        HYPRE_Real absolute_tolerance);
```

The returned object provides setup, solve, and destroy callbacks. Candidates may
assemble a solver from the HYPRE operations allowlisted by the benchmark. Stock
HYPRE Krylov outer solvers are reserved for the fixed reference. Candidates may
not perform I/O, networking, process creation, dynamic loading, or timing. Inputs
must not be modified.

Submissions use a reviewed-source threat model. The compiler and symbol audit
enforce the ordinary C contract; they are not a security boundary against hostile
inline assembly or raw system calls. Modal still isolates the evaluation from the
host and blocks sandbox networking.

## Paired execution

The fixed reference and candidate are built against the same pinned HYPRE runtime
inside one environment. Each numerical case is run in two fresh processes:

1. Run and verify the fixed reference.
2. Run and verify the candidate.
3. Compute `speedup = reference_seconds / candidate_seconds` if both pass.

A reference failure invalidates the evaluation. A candidate failure is recorded
with `passed: false` and no speedup. Failed or omitted cases never disappear from
an aggregate.

The native monotonic clock measures factory creation, setup, and solve. It excludes
compilation, input loading, transport, destruction, and trusted verification.
The reference is measured again for every candidate; timings are never reused
between sandboxes. Each solver process has a 600-second wall-time limit.

The complete-split score is the geometric mean of scored-case speedups, provided
the candidate passes every scored and control case. Partial evaluations and
evaluations with failures have no aggregate score.

## Accuracy

For residual `r = b - A x`, the trusted evaluator computes normwise and
componentwise backward error and the relative residual. NS cases also compare
against the evaluator-held manufactured target.

NS requires:

- normwise backward error ≤ `1e-10`;
- componentwise backward error ≤ `1e-8`;
- relative L2 and Linf forward errors ≤ `1e-5`.

FLASH requires:

- normwise backward error ≤ `max(tolerance, 20 * float64_epsilon)`;
- relative residual ≤ `tolerance * (1 + 1e-6)`.

Both families require a finite solution of the correct size, a successful solver
status, and unchanged inputs.

## Modal venue

Modal is the default and the only official scoring venue. Local runs are useful
development checks, but their reports are marked `official: false`.

The Modal venue uses a versioned immutable image, exact CPU and memory
request/limit pairs, and a network-disabled sandbox. It compiles the two solvers
once and transfers one public case at a time. Only the numerical `CaseInput`
crosses the execution boundary; targets and correctness remain on the trusted
client.

Changing the reference source, HYPRE runtime, native driver, image, resources,
case membership, timing boundary, or accuracy rules creates a new venue or
benchmark version.
