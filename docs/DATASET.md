# Dataset and provenance

## Pilot releases

A release manifest fixes its family, track, development or ranked split, ordered
case inventory, scientific admission, numerical identities, and execution and
scoring contracts. Names alone do not establish scientific qualification.

The NS pilot release `ns-mesh-pilot` contains 43 real mesh/PDE matrices, each
with one newly qualified manufactured workload:

| Installed manifest | Split | Cases | Provenance groups | Unknowns |
| --- | --- | ---: | ---: | ---: |
| `ns-mesh-pilot-dev` | Development | 11 | 4 | 317–26,068 |
| `ns-mesh-pilot-ranked` | Ranked | 32 | 17 | 240–1,489,752 |

Groups do not cross the split boundary. The largest operator has 10,319,760
nonzeros. Both splits use the same numerical gates and execution contract:
90 seconds per case, two CPU cores, 4 GiB memory, a 5,000-iteration request, and
relative stopping tolerance `1e-12`. Every case was freshly qualified against
its frozen target and RHS; gates and numerical inputs were not relaxed to
obtain passing qualification.

These 43 cases define the public pilot's scientific scope; they are not an
exhaustive NS mesh inventory. The word NS means nonsymmetric and is not
restricted to Navier–Stokes equations. The existing v1 catalogue's generic
nonsymmetry screen does not establish mesh/PDE provenance.

The earlier `ns-mesh-dev-pilot` is retained separately as a two-case integration
check using `DRIVCAV/cavity01` (317 unknowns) and `FEMLAB/poisson2D` (367 unknowns).
It is not the main pilot's development or ranked split.

Preparation obtains original archives from the
[SuiteSparse Matrix Collection](https://sparse.tamu.edu/), checking both archive
and canonical matrix hashes. Each case preserves its public matrix identifier,
source citation, and scientific admission statement. Sparse storage is
canonicalized by expanding declared symmetry, summing duplicate coordinates,
removing exact zeros, and sorting each row's `(column index, value)` pairs.
Sorting storage pairs does not permute matrix columns or change the operator.
Lossy numeric conversions, nonfinite entries, and unsupported dimensions are
rejected. No scaling, transpose, shift, or variable reordering is introduced.

Each matrix receives one deterministic Rademacher target with entries ±1 and
exactly unit RMS. The versioned HMAC recipe separates draws by release, family,
canonical matrix, RHS kind, and draw index. Development keys are public; ranked
keys remain with the operator. The target and `b = fl(A x_target)` are frozen
in trusted prepared archives. A new draw changes the numerical system identity
and requires new qualification.

NS qualification distinguishes two kinds of evidence, both tied to the exact
stored system and requiring a tenfold margin against the four acceptance gates:

- **41 refined sparse-LU cases.** An independent float64 factorization solution
  receives exactly one correction using a higher-precision residual and the
  same factors. Extended precision is used when available, otherwise 80-digit
  decimal arithmetic. The final metrics and RHS formation audit are empirical
  numerical evidence, not verified forward-error bounds.
- **Two strict row-dominance cases.** Exact integer accumulation of binary64
  coefficients establishes every row's dominance margin and the exact RHS
  formation residual. Their ratio certifies the exact stored-system solution's
  relative L2 and Linf distance from the manufactured target. Separately,
  binary64 verifier metrics at the target are measured as a feasible witness;
  they are not claimed to be interval bounds or an independent factorization.
  The certified forward bound must also satisfy the tenfold margin.

The two certified forward bounds are below `1.2e-9`, against a required
qualification ceiling of `1e-6`. The original unrefined LU method remains
supported for older frozen development evidence. See
[the accuracy contract](SPEC.md#independent-accuracy-checks) for the common
acceptance gates and the distinction between targets and exact solutions.

## FLASH replay downloads

The publication destination is the public, ungated
[LinearSolveBench dataset on Hugging Face](https://huggingface.co/datasets/hgarud/LinearSolveBench).
The published inventory contains 344 admitted captured systems: 96 development
cases and 248 ranked cases, of which 88 are correctness controls. Manifests
`flash-replay-dev-pilot` and `flash-replay-ranked-pilot` share release identity
`flash-replay-pilot` and pin dataset commit
`3da5eda0ce3e85da1808d4b62d28d9111127e354`. Archive and canonical matrix/vector
hashes are frozen in each manifest.

```bash
linear-solver-bench dataset prepare \
  --release flash-replay-dev-pilot --output data/prepared/flash-dev
```

The captures come from FLASH 4.8 AlWire, driven two-dimensional ZPinch, and
MagDiff test-problem variants, primarily during startup. There are 13 source
runs and 10 provenance groups. Dimensions range from 4,224 to 196,608 and
nonzero counts from 16,420 to 2,153,480. The catalogue documents the limited
near-duplicate audit: it compares identical canonical sparsity patterns and
does not claim permutation invariance.

Every case is a standalone ZIP archive with exactly:

```text
matrix.npz    # float64 CSR, readable by scipy.sparse.load_npz
b.npy         # captured float64 right-hand side
x0.npy        # captured float64 initial guess
case.json     # public identity, provenance grouping, tolerance, and hashes
```

Downloading one case requires neither the benchmark package nor FLASH. The
catalogue maps public case IDs to archive paths, byte sizes, SHA-256 hashes,
dimensions, source-run and provenance-group IDs, time steps, and within-step
solve positions. A release assigns scored/control roles and freezes aggregation
weights. Additional Matrix Market exports and bulk bundles are outside the
pilot format.

The captured matrix, RHS, initial guess, and relative tolerance are preserved.
Cases requiring nonzero absolute tolerance are rejected. This four-file pilot
format contains no reference solution. The numerical verifier can compute
diagnostics against a separately supplied numerical reference, without treating
it as exact truth.
Captured and offline numerical solutions establish case feasibility. The
fixed public reference `submissions/gmres_amg.c` additionally passes all 344
cases locally, including every correctness control. Full Modal qualification
and registration remain in progress; local timings are not official venue
reference times.

Both development and ranked numerical inputs are public. Related source runs,
refinements, exact duplicates, and documented near-duplicate relationships must
remain within the same split. Ranked is an evaluation designation, not a claim
that those input matrices are hidden. Replay evaluates individual captured
solves and does not claim complete-trajectory coverage or rerun the simulation.

## Preparation and cache

```bash
linear-solver-bench dataset prepare \
  --release ns-mesh-pilot-dev --output data/prepared/ns-dev
linear-solver-bench dataset prepare \
  --release ns-mesh-pilot-dev --offline --output data/prepared/ns-dev-offline
```

Preparation validates source identity, safely decodes each source archive, and
writes a trusted schema-v2 prepared archive per case. NS archives declare a
manufactured reference; FLASH pilot archives declare no reference. Candidate
input payloads contain only numerical solve inputs. Frozen NS qualification is
reused after verifying its exact numerical identity, so ordinary preparation
does not repeat offline factorizations.

Downloads use a content-addressed cache with bounded sizes, resumable transfer,
and hash validation before reuse. `--offline` uses verified cached archives and
fails clearly for missing or corrupt data. Set `--cache` or
`LINEAR_SOLVER_BENCH_CACHE` to choose writable storage. Installed package
resources are read-only inputs and need no checkout for the Python CLI.

`--case CASE_ID` may select a case for a development check. A subset is marked
incomplete and cannot receive a complete-split score. Local and trusted Modal
evaluation load prepared cases one at a time to limit evaluator memory.

## Existing SuiteSparse v1 dataset

The following inventory and qualification procedure describe the original v1
benchmark, not the NS mesh pilot. Its manifests and archive identities remain
supported without relabeling their scientific scope.

### Source snapshot

The source catalogue is the SuiteSparse Matrix Collection
[`ss_index.mat`](https://sparse.tamu.edu/files/ss_index.mat); field definitions
are documented on the collection's
[statistics page](https://sparse.tamu.edu/statistics). The frozen snapshot has:

- `LastRevisionDate`: `31-Oct-2023 18:12:37`
- SHA-256: `f040a883a523c4621c5986aa66c39eeaa21bcae79c45596e953e7ea5e7ccbdb4`

The complete metadata screen is frozen in `data/catalogue.json`. All filters
are cumulative:

```text
RBtype == "rua"
nrows == ncols
0 <= numerical_symmetry < 0.99
isGraph == 0
sprank == nrows
500 <= nrows <= 1,000,000
nnz <= 20,000,000
3 <= nnz / nrows <= 500
nnzdiag / nrows >= 0.50
1 <= nblocks <= 0.25 * nrows
amd_vnz >= 0
amd_rnz >= 0
amd_vnz + amd_rnz <= 100,000,000
```

The result contains 350 matrices. `RBtype == "rua"` means real-valued,
unsymmetric, assembled; the explicit square check remains in the frozen filter
for readability and replay safety.

### Numerical qualification

`data/qualification.json` freezes numerical rank and condition evidence for all
350 matrices. Every matrix is processed by the same method. SuiteSparseQR uses
an explicit cutoff of
`40 * n * float64_epsilon * max_column_2_norm(A)` and reports 213 numerically
full-rank matrices and 137 rank-deficient matrices. For each full-rank matrix,
three deterministic `onenormest` runs (`t=4`, `itmax=10`) reuse the QR factor's
ordinary and transpose solves to estimate `norm(A^-1, 1)`. The largest estimate
is multiplied by a guard factor of 3 and then by `norm(A, 1)` to produce the
frozen condition estimate.

The ranked v1 set contains the 203 full-rank cases supported by at least one
accuracy tier. Each case receives the smallest tolerance in `1e-4`, `1e-3`,
`1e-2`, and `4e-2` for which:

```text
100 * estimated_condition * float64_epsilon <= 5 * case_tolerance
```

The tier counts are 183, 9, 7, and 4 respectively. Ten other full-rank matrices
exceed the `4e-2` tier's condition ceiling and are excluded from v1. The
evaluator continues to compute actual relative forward error in the 2-norm;
it also computes relative infinity-norm forward error and requires both gates
to pass. The 1-norm condition estimate assigns the tolerance but does not
replace either direct check or claim to upper-bound the 2-norm condition
number.

`data/qualification-inputs-v1.json` freezes every downloaded archive identity,
and `qualification_modal.py` reproduces the qualification campaign. The
content hash of the resulting evidence is bound into both dataset manifests.
The runner first fills a hash-addressed Modal volume over authenticated HTTPS,
then performs all QR factorizations and condition estimates in network-disabled
workers. Qualification may use more memory and CPU than the solver-evaluation
venue because it is offline dataset preparation, not measured candidate work.

```bash
modal run qualification_modal.py \
  --output data/qualification-replay.json
```

### Matrix transformation

Preparation downloads `https://sparse.tamu.edu/MM/{Group}/{Name}.tar.gz`,
verifies the frozen archive digest, safely extracts the Matrix Market member,
expands declared symmetry, converts to float64 CSR, sums duplicate entries,
removes exact zeros, and sorts column indices. It performs no reordering,
scaling, or transposition.

Every prepared case is content-addressed. Ranked right-hand sides are derived
only during trusted preparation from an operator-held key. The key and manufactured
target are never copied into the candidate sandbox workspace.

## Attribution

SuiteSparse matrices are licensed CC BY 4.0. Redistributions must preserve the
original Matrix Market metadata and matrix-specific citation instructions,
identify the canonicalization and manufactured-right-hand-side modifications,
and cite:

- Timothy A. Davis and Yifan Hu, *The University of Florida Sparse Matrix
  Collection*, ACM TOMS 38(1), 2011, DOI 10.1145/2049662.2049663.
- Scott P. Kolodziej et al., *The SuiteSparse Matrix Collection Website
  Interface*, JOSS 4(35), 2019, DOI 10.21105/joss.01244.
