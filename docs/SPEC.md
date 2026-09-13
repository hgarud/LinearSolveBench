# Benchmark specification

## Supported contracts

The pilot benchmark identity is `linear-solver-bench-pilot-cpu-v2`. It supports:

| Family / track | Accuracy contract | Scoring contract |
| --- | --- | --- |
| `ns-mesh-pde` / `coverage` | `ns-mesh-accuracy-v1` | `ns-coverage-count-v1` |
| `magnetic_diffusion_flash` / `replay` | `flash-replay-accuracy-v1` | `flash-replay-speedup-v1` |

Both use `independent-case-once-v1`: one execution per case with no retained
solver state between cases. The release fixes case order, roles, weights,
tolerances, deadlines, iteration requests, and resources. Unsupported family
and track combinations are rejected. The existing v1 contract is described
separately below and retains its original identity.

## Candidate artifact and inputs

A submission is one regular UTF-8 C source file with LF endings, at most 65,536
bytes. It contains exactly one include directive, for `HYPRE_parcsr_ls.h`, and
exports exactly `solver_create`:

```c
HYPRE_Int solver_create(HYPRE_Solver *solver,
                        HYPRE_Real relative_tolerance,
                        HYPRE_Real absolute_tolerance,
                        HYPRE_Int maximum_iterations);
```

The evaluator force-includes `nsl_hypre_solver.h` and audits object symbols
before linking. Allowed capabilities include selected vector, reduction, and
matrix-vector operations and approved preconditioner lifecycle/configuration
calls. I/O, clocks, processes, networking, dynamic linking, and stock outer
solvers are excluded. Operating-system isolation remains a separate boundary.

Each input contains a square float64 CSR matrix `A`, vectors `b,x0`, and relative
tolerance `tau`. Matrix dimensions and nonzero counts must fit the native int32
build. The factory receives the release's iteration limit and absolute tolerance
zero. The candidate must honor those requests and preserve `A,b`.

NS uses a manufactured target with entries in `{−1,+1}`, exactly unit RMS,
`b = fl(A x_target)`, and `x0 = 0`. FLASH preserves captured `b,x0,tau`; captures
requiring nonzero absolute tolerance are incompatible with this pilot. Case IDs,
source names, provenance, keys, and reference vectors stay outside candidate code.

## Lifecycle and timing

Compile the candidate once. For every pilot case, start one fresh native process
and construct new solver objects and workspace. There are no warm-ups, repeated
evaluations, or candidate-failure retries. A candidate failure on one case does
not prevent evaluation of the remaining cases.

Input decoding and trusted HYPRE object construction precede timing. The measured
elapsed time covers factory creation, setup/preconditioner construction, and
solve. Destruction, solution extraction, input-mutation checks, and trusted
verification follow timing. Reports retain elapsed time and phase timings from
that single execution. Local and Modal evaluators load numerical cases one at
a time and retain scalar results between cases.

## Independent accuracy checks

Verification always uses the original serialized operator and RHS. For
`r = b - A x`, compute:

```text
relative_residual = ||r||2 / ||b||2
eta_norm = ||r||inf / (||A||inf ||x||inf + ||b||inf)
eta_comp = max_i |r_i| / (|A||x| + |b|)_i
forward_l2 = ||x - x_reference||2 / ||x_reference||2
forward_linf = ||x - x_reference||inf / ||x_reference||inf
```

| Metric | NS coverage | FLASH replay |
| --- | --- | --- |
| Relative residual | Diagnostic | `<= tau * (1 + 1e-6)` |
| Normwise backward error | `<= 1e-10` | `<= max(tau, 20 eps)` |
| Componentwise backward error | `<= 1e-8` | Diagnostic, without row floors |
| Relative forward error, L2 | `<= 1e-5` against manufactured target | Optional numerical-reference diagnostic |
| Relative forward error, Linf | `<= 1e-5` against manufactured target | Optional numerical-reference diagnostic |

Here `eps` is float64 machine epsilon. NS acceptance limits are fixed across
cases; the solver stopping request, initially `1e-12`, is a separate setting.
Relative-residual guidance of `1e-8` does not replace the NS backward/forward gates.

Manufactured targets are required for NS and cannot be replaced by numerical
references or omitted. Floating-point RHS formation means a target need not be
the exact solution of the stored system. Offline qualification checks empirical
feasibility at one tenth of every required error limit and records RHS formation
uncertainty. Numerical references are labeled as such; they carry no exact-truth
claim and do not add forward-error gates to FLASH.

Zero divided by zero is defined as zero for these ratios. A nonzero numerator
over zero is infinity; in particular, a zero RHS passes the residual test only
with an exactly zero residual. Stable norm ratios avoid overflow from squaring
large values. Nonfinite residuals or required denominators cannot yield a pass.
Unavailable or nonfinite diagnostics serialize as JSON `null` with a reason.

Output must have the expected shape, be finite, return status zero, leave inputs
unchanged, and complete within the fixed deadline. Required numerical failures
make the case fail even if the candidate declares convergence.

## Scores and reference identity

NS coverage reports `solved_count`, `coverage_fraction`, and ranking key
`-solved_count`. Equal counts tie; timing is diagnostic. Scoring requires the
complete expected split and no calibration artifact.

For FLASH scored cases, `speedup_i = reference_elapsed_i / candidate_elapsed_i`.
The aggregate is the weighted geometric mean, with equal mass assigned to
provenance groups, then source runs within each group, then scored cases within
each run. Weights are frozen before evaluation. Correctness controls execute
and must pass, but have zero aggregate timing weight.

One declared reference must pass every scored case and control on the complete
split. Qualification freezes its single measured time for each case. Candidate
runs reuse these timings. Reference artifacts bind the source, numerical inputs,
case order, roles, weights, contracts, runtime, and venue. A collection of per-case
portfolio winners is not a reference solver.

Any failed candidate case makes replay `eligible = false` and `speedup = null`.
Missing reference calibration gives reason `awaiting_reference`. No failure is
replaced by a budget time, and successful subsets are not scored. Infrastructure
failures invalidate evaluation rather than removing expected cases.

## Venue and release trust

Local evaluation is `local-uncontrolled`: it supports development, with no claim
of hard CPU or memory enforcement. Modal uses `modal-sandbox-pilot-cpu-v2`, with
CPU request/limit and memory request/limit taken from the release, networking
disabled, and one sandbox per submission. Native child processes handle cases
individually. HYPRE is the pinned sequential float64/int32 build.

An official score requires the full split, enforced venue limits, and the exact
release digest in the packaged trusted registry. A self-consistent user-authored
manifest or report does not establish official release identity. A local timing
reference cannot be mixed with Modal candidate timings. Official replay also
requires the registered calibration digest for that release, runtime, and venue;
a custom passing reference remains useful for development but is not official.

## Original SuiteSparse v1 compatibility

V1 retains benchmark identity `linear-solver-bench-cpu-v1` and accuracy contract
`three-gate-v1`. It uses manufactured RHS and a zero initial guess, three fresh
processes per case, and requires every repetition to pass. Its limits remain:

```text
eta_norm <= max(tau, 20 eps)
eta_comp <= max(5 tau, 100 eps)
max(forward_l2, forward_linf) <= min(2e-1, max(5 tau, 2e-12))
```

Frozen condition estimates select one tolerance tier from `1e-4`, `1e-3`, `1e-2`,
and `4e-2`. They do not replace measured errors. Ranking maximizes completely
solved cases, then minimizes a penalized geometric mean: successful cases use
the median time divided by calibrated reference time, and failures use twice
the deadline divided by that reference time. V1 scoring requires calibration.
Its Modal venue remains two physical CPU cores and 4096 MiB memory. Existing
v1 archive decoding and manufactured archive bytes remain supported.
