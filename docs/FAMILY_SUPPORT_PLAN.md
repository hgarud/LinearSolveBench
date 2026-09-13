# First pilot: NS mesh coverage and FLASH replay

Status: implemented and validated pilot, updated 13 September 2026. NS has 43
qualified cases split into 11 development and 32 ranked cases, all evaluated in
the official Modal venue. FLASH has 344 publicly downloadable captures and a
fixed public reference that passes every case in that venue. Releases, runtime,
and replay calibration identities are frozen. The checks and scientific scope
below define this completed pilot; future tracks remain separate work.

## Implementation snapshot

| Area | Verified pilot evidence | Requirement for later changes |
| --- | --- | --- |
| Family and execution contracts | Two explicit pairs; `ns_mesh_pde` alias; schema-v2 identities; one execution per case; registered split digests and fixed venue | Version changed inputs, contracts, and venues |
| Numerical verification | Fixed NS gates, FLASH residual/backward gates, explicit reference semantics, stable ratios, JSON-safe diagnostics, v1 compatibility | Retain qualification and identity checks for later releases |
| NS coverage | All 43 qualified SuiteSparse workloads prepare and verify; full Modal execution gives 10/11 development and 14/32 ranked passes for the public GMRES+AMG solver | Requalify new manufactured inputs; preserve failures in coverage scores |
| FLASH inputs | 344 standalone archives published on Hugging Face; commit-pinned manifests; captured arrays verified; fresh installed public download succeeds | Preserve immutable bytes and provenance split boundaries |
| FLASH reference and scores | Fixed public `gmres_amg.c` passes all 344 Modal cases; registered single-execution timings and weighted scoring | Requalify when reference, inputs, runtime, or venue changes |
| Execution and distribution | Full Modal split runs; hard resource limits; fresh wheel/sdist builds, isolated installed resources, downloads, offline reuse, and native compilation | Repeat package/content and venue checks for each release |

The NS release `ns-mesh-pilot` contains 43 matrices in 21 provenance groups,
with 11 development cases in four groups and 32 ranked cases in 17 groups.
Dimensions range from 240 to 1,489,752. Qualification uses one fixed
higher-precision refinement after sparse LU for 41 cases and exact
row-diagonal-dominance certificates for two large cases. Every frozen workload
meets the unchanged tenfold qualification margins. This is a declared pilot
corpus, not an exhaustive scientific family inventory. The earlier two-case
`ns-mesh-dev-pilot` remains a separate integration check; the generic v1
inventory remains a separate benchmark.

The full NS Modal check completed every case once without crashes, timeouts,
infrastructure failures, or retries. Both matrices with over a million unknowns
passed within 4 GiB. A partial solved-case count is a valid coverage result;
NS coverage does not require discovery of an all-case performance reference.

The published FLASH splits are 96 development scored cases and 248 ranked cases
(160 scored, 88 correctness controls). Public inputs cover 13 source runs in 10
provenance groups. Captured solutions and stored independent numerical references
pass the public accuracy gates for all 344 cases. The fixed public reference
also passes all 344 in the official Modal venue. Complete reports and reference
timings are frozen in the [release artifacts](../data/releases/README.md).
The ungated
[Hugging Face dataset](https://huggingface.co/datasets/hgarud/LinearSolveBench)
is pinned by both manifests to commit
`3da5eda0ce3e85da1808d4b62d28d9111127e354`.

## Pilot scope

The implementation supports exactly two family/track pairs through one
independent-case evaluator:

| Family | Track | Primary result | Timing reference |
| --- | --- | --- | --- |
| `ns-mesh-pde` | `coverage` | Number of cases solved accurately within fixed budgets | Not required |
| `magnetic_diffusion_flash` | `replay` | Speedup on individual captured solves | One frozen reference qualified on the entire replay split |

`ns_mesh_pde` is accepted as a CLI alias. Manifests and reports use the public
ID `ns-mesh-pde`.

SPD mesh, NS mesh performance, and FLASH trajectory are later phases. Their
agreed designs remain in [FAMILY_SUPPORT_ROADMAP.md](FAMILY_SUPPORT_ROADMAP.md).
They do not add pilot implementation or acceptance requirements. In particular,
the pilot needs no SPD qualification pipeline, trajectory datasets or driver,
persistent solver ABI, retained-matrix API, or NS performance activation flow.
The CLI rejects unsupported family/track pairs.

Keep family, track, release, accuracy contract, and venue identities explicit
so later additions can use new versioned contracts. Implement only the two
supported pairs; do not build a generic plugin system or empty future adapters.
Preserve the existing public v1 manifests and replay behavior under their
original identity. Give the pilot a distinct release and benchmark identity.

## Shared execution and accuracy

Both pilot tracks fit the existing `solver_create` interface and numerical input
protocol. Compile one C submission once, then evaluate each case exactly once
in one fresh native process. Solver objects, workspace, and preconditioners are
recreated for every case. No state carries between cases. Freeze
`repetitions = 1` in the pilot execution contract to keep evaluation costs down;
do not add timed warm-ups, hidden repetitions, or candidate-failure retries.

The timed boundary remains factory creation, candidate setup/preconditioner
construction, and solve. Input decoding and HYPRE object construction precede
timing. Candidate destruction, output extraction, mutation checks, and trusted
verification follow timing. Report phase timings and the measured complete
native solve time from that single execution. Count a case as passing when
that execution satisfies every required accuracy and execution check.

Use the existing compiler capability restrictions, pinned sequential float64
HYPRE runtime, and candidate isolation. Only numerical `A,b,x0,tau` inputs reach
candidate code. Keep source names, public case IDs, provenance, reference
vectors, and preparation keys outside the candidate boundary.

Preserve original operators after documented storage canonicalization: expand
declared Matrix Market symmetry, sum duplicates, remove exact zeros, and sort
each row's `(column index, value)` storage pairs. This does not permute matrix
columns or variables. Do not silently scale, transpose, reorder, shift, or replace the
operator with another problem during preparation. Verification uses the original
serialized matrix and RHS, regardless of the candidate's internal algorithm.

| Input or required metric | NS mesh coverage | FLASH replay |
| --- | --- | --- |
| Scientific admission | Real, materially nonsymmetric primary systems with verified mesh/PDE provenance and numerical qualification | Valid captured scalar magnetic-diffusion systems with qualified capture evidence |
| RHS | One Rademacher manufactured target per matrix at unit RMS; `b = fl(A x_target)` | Preserve captured `b` |
| Initial guess | Zero | Preserve captured `x0` |
| Relative residual `\|\|b-Ax\|\|2 / \|\|b\|\|2` | Diagnostic | `<= tau * (1 + 1e-6)` |
| Normwise backward error | `<= 1e-10` | `<= max(tau, 20 eps)` |
| Componentwise backward error | `<= 1e-8` | Diagnostic |
| Relative forward error in L2 and Linf | Each `<= 1e-5` against the manufactured target | Numerical-reference diagnostics only |

Use the normwise/componentwise definitions in [SPEC.md](SPEC.md). Both tracks
also require valid shape, finite output, successful native status, unchanged
inputs, and compliance with time, memory, and iteration limits. Invalid required
metrics fail; unavailable or infinite diagnostics use JSON `null` with a reason.

`ns` means nonsymmetric, not exclusively Navier–Stokes. The current generic
SuiteSparse v1 screen does not establish mesh/PDE provenance and must not be
relabelled as this family. Validate scientific source evidence and numerical
feasibility independently; solver failure alone is not grounds for excluding
an otherwise qualified NS coverage case.

Distinguish manufactured targets from numerical reference solutions. Because
RHS formation rounds in float64, the manufactured target is not necessarily the
exact solution of the stored `A,b`; qualify formation uncertainty relative to
the forward-error limits. NS forward checks must never be disabled by missing
truth. FLASH numerical references, when available, support diagnostics without
becoming exact-solution claims or required forward-error gates.

The NS acceptance limits above are fixed benchmark requirements shared by every
case. Numerical qualification checks their feasibility for each frozen
`A,b,x_target`; it does not calculate a different tolerance for each matrix.
Require offline qualification evidence at one tenth of every required error
limit. Record whether that evidence consists of independently computed
reference-solve metrics or conservative mathematical bounds. A successful
float64 reference solve establishes empirical feasibility; it does not certify
the distance between the exact stored-system solution and the manufactured
target. Audit sensitive cases with higher-precision residuals/refinement or
verified error bounds before release, documenting the remaining uncertainty.
Condition estimates are diagnostics, not certified upper bounds or formulas for
the acceptance limits. This preparation adds no candidate evaluation processes.

Implemented qualification methods distinguish their evidence explicitly. Sparse
LU can perform one fixed higher-precision residual correction using the same
factors; its measured result remains empirical. An alternate strict row-diagonal
dominance path accumulates coefficients and RHS formation residuals exactly as
binary64 integers and certifies a bound on stored-system forward discrepancy.
It separately checks measured target-witness metrics at the tenfold limits and
also requires the certified forward bound to meet that margin. It does not claim
that those measured floating-point metrics are themselves interval bounds.

For FLASH, retain raw componentwise diagnostics even when weak rows make them
large; do not introduce row floors or reject on that diagnostic. Define a zero
RHS residual ratio as zero only for an exactly zero residual, otherwise infinity.
The pilot supports zero absolute tolerance; reject incompatible captures rather
than silently discarding their requested absolute tolerance.

Keep the frozen NS acceptance limits separate from the solver's stopping
request. Start NS feasibility runs with a supplied relative tolerance of
`1e-12` and 5,000 iterations; freeze those execution settings with the release.
Neither that stopping request nor the diagnostic relative-residual guidance of
`1e-8` replaces or changes the backward/forward acceptance limits. FLASH receives
its captured relative tolerance and a separately published iteration cap. Use
numerically stable norm evaluation and detect overflowing residuals or
denominators before accepting any metric.

## Track scores and the replay reference

NS coverage ranks by raw solved-case count:

```text
passed_i = the single execution of case i passes
solved_count = sum_i passed_i
coverage_fraction = solved_count / expected_case_count
ranking_key = -solved_count
```

Equal solved counts tie; timing is diagnostic and does not break ties. Retain
outcomes for every expected case. A candidate failure on one case does not stop
other cases from being evaluated. Provenance-group breakdowns can accompany the
raw score but do not change its meaning. This path requires no timing reference
or calibration artifact, although numerical qualification remains mandatory.

FLASH replay measures each captured solve independently. Where one physical
time step contains multiple magnetic solves, expose them as distinct cases
with public within-step ordering. The pilot does not reconstruct or score
complete trajectories.

Qualify one public reference solver/configuration on every case in the frozen
replay split, including correctness controls, under the same runtime, accuracy,
single-execution lifecycle, and venue as candidates. A stock GMRES+AMG
configuration requires measured passing evidence. The current public
`submissions/gmres_amg.c` implementation passes the full 344-case inventory
in the official Modal venue, including every correctness control.
Publish its source/configuration, complete qualification report, and timing
artifact. Reference qualification runs each case once and freezes that measured
time. Reuse the frozen reference timings for subsequent submissions rather than
rerunning the reference alongside each candidate. No private runtime or source
dependencies may be needed to reproduce it.

For a scored case, define `speedup_i = reference_time_i / candidate_time_i`,
using each solver's single measured execution time.
Aggregate with a weighted geometric mean, assigning equal mass to provenance
groups, equal source runs within each group, and equal scored cases within each
run. Freeze weights before timing or candidate evaluation. A value above one
means faster than the reference.

Require all scored cases and mandatory correctness controls to pass their
single candidate execution for an official aggregate speedup. A failure yields
`eligible = false` and `speedup = null`, while preserving per-case diagnostics.
Run correctness-only controls but exclude their timings from the speed mean.
Never average only successful cases, use a budget as a substitute reference,
or substitute per-case winners from a portfolio for the declared reference.

Bind calibration and reports to the exact family/track, ordered cases, roles and
weights, prepared numerical inputs, accuracy/execution/scoring contracts,
reference identity, compiler/runtime, execution count, and venue. Reject incomplete reports or
mismatches. Infrastructure failure invalidates a run under a fixed retry policy;
it does not silently shrink the scored set. Keep coverage and replay leaderboards
separate, with no combined score.

## Implemented modules

The pilot extends the existing evaluator through a fixed registry. The original
v1 dataset and scoring paths retain their identities and three-repetition
behavior. No generic plugin framework or future-family runtime is introduced.

| Area | Implemented pilot behavior |
| --- | --- |
| `families.py`, `manifests.py` | Two supported pairs, strict identities/settings, source and qualification validation, published-release digest checks |
| `pilot_dataset.py`, `sources/`, `workloads.py` | Public SuiteSparse and FLASH loaders, deterministic manufactured vectors, qualification, resumable verified downloads and offline cache reuse |
| `models.py`, `archive.py` | Required manufactured targets, optional numerical references, explicit schema-v2 archives, legacy decoding and byte compatibility |
| `accuracy.py`, `verify.py` | Shared stable metrics with separate NS/FLASH gates and diagnostic handling |
| `pilot_runner.py`, `runner.py`, `modal_app.py` | One execution per case, shared verification/report construction, release budgets and streamed numerical inputs |
| `pilot_scoring.py` | Reference-free coverage and all-pass qualified-reference replay speedup with frozen group/run/case weights |
| `cli.py`, `paths.py`, packaging | Release selection, automatic prepared-schema dispatch, installed resources, public cache locations and CLI documentation |
| `submissions/gmres_amg.c` | Fixed public reference passes all 344 FLASH cases in the official venue; timings are frozen for reuse |

Load prepared cases one at a time in local and trusted operator execution.
Retain a small case specification, family/track contract, optional reference
solver identity, and release record. Do not add `TrajectorySpec`, trajectory
callbacks, or placeholder SPD runtime types to pilot code. Retain the current
single-case C ABI and force-included header; change the native driver only if
pilot validation exposes a necessary correctness or settings-propagation fix.

Centralize effective settings so manifest values, Python execution, and native
limits cannot disagree. A user-authored self-consistent manifest is insufficient
for official evaluation: bind it to a trusted published pilot release. Keep
acquisition and qualification dependencies out of the candidate runtime.

## Public data and publication boundary

Freeze a declared pilot inventory and provenance-based development/ranked
splits. Use tiny synthetic fixtures and a small real case from each family for
the first integration checks; then validate all cases in the declared pilot.
Exact pilot counts are chosen after asset inventory and qualification, not
implicitly inherited from another corpus or chosen from candidate success.
Document the pilot's limited scope rather than claiming full-family coverage.

For NS mesh, download original SuiteSparse archives by public identifiers and
URLs, verify archive and canonical-matrix checksums, preserve source metadata,
and generate the declared manufactured workloads using a versioned recipe.
Freeze numerical bytes and qualification evidence. Domain-separate new draws by
release, family, matrix, RHS kind, and draw index. Keep operator-held ranked
keys and reference vectors outside candidate-visible storage. New draws require
new prepared identities and numerical qualification.

For FLASH replay, publish a standalone numerical dataset in a public, ungated
Hugging Face dataset repository with commit-pinned downloads. Hugging Face is
the only required publication host; Zenodo can be added later if needed.
Publish a catalogue mapping stable public IDs
to URLs, byte sizes, SHA-256 hashes, matrix dimensions, tolerances, scientific
provenance, roles, and citations. Provide one downloadable archive per case:

```text
matrix.npz    # Canonical CSR, readable with scipy.sparse.load_npz
b.npy         # Captured float64 RHS
x0.npy        # Captured float64 initial guess
case.json     # Public metadata, tolerance, identity, and hashes
```

These per-case archives are the required public format for FLASH replay;
additional Matrix Market exports and bulk bundles are outside the pilot scope.
Users must be able to download a single case directly without installing the
benchmark or obtaining FLASH. Preparation
resolves IDs, verifies downloads and canonical arrays, extracts safely, and
caches by content hash. Support download resumption and offline cache reuse.
[Pinned Hub downloads](https://huggingface.co/docs/huggingface_hub/main/guides/download).

Publish FLASH development and ranked numerical inputs; ranked denotes the
evaluation split, not secret data. Keep related source runs, time steps,
refinements, exact duplicates, and documented near-duplicates in the same split.
Preserve public source-run grouping so a future trajectory release can respect
these exposure boundaries, but do not require full-trajectory capture,
publication, or execution for the replay pilot. Publicly downloadable inputs
cannot be claimed to be hidden merely because reference vectors are private.

All 43 NS sources are downloadable and numerically qualified, and public
preparation has verified every frozen workload. FLASH assets are publicly
downloadable with immutable commit identities and frozen manifests. NS coverage
can become available independently if replay reference qualification is
delayed. A pilot claiming both tracks complete requires both working public
data paths and a qualified reference in the declared comparison venue.

Construct new public manifests and qualification reports through explicit
allowlisted exports. Exclude private package imports, repository paths/names,
account/deployment/storage identifiers, experiment receipts, prompts, model
outputs, and private source hashes. Keep conversion tools and source-to-public
mappings outside the public repository. Audit the source tree, Git history,
wheel/sdist, container context, embedded archive metadata, and documentation.
Public CI can validate allowed schemas and imports; literal private identifiers
used in release scans must remain in private tooling.

Preserve legitimate upstream citations and notices. SuiteSparse matrices are
CC BY 4.0, with matrix metadata retained and modifications identified.
[SuiteSparse licensing guidance](https://sparse.tamu.edu/about).
Establish redistribution terms for captured FLASH numerical data and metadata
before uploading. Do not bundle FLASH source or assume that its capture patches
are redistributable. [FLASH license agreement](https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node3.html).

## Pilot sequence and acceptance criteria

| Step | Status | Completed evidence |
| --- | --- | --- |
| 1. Freeze contracts and inventory | Implemented; declared inventories qualified | Two pairs and public schemas are explicit. NS has 43 qualified cases in disjoint 11/32 splits. FLASH has 344 cases in published 96/248 splits. |
| 2. Deliver NS coverage end to end | Implemented for the full pilot | All 43 public sources prepare and verify; bound qualification evidence, one fresh process per case, failure-preserving reports, and calibration-free scoring are available. |
| 3. Deliver FLASH replay end to end | Implemented; public assets published | Captured arrays round-trip unchanged across 344 archives; both manifests pin an immutable Hugging Face commit. |
| 4. Complete replay qualification and scoring | Complete | One fixed public solver passes every case and control in both Modal splits; complete reports and registered timings are frozen. |
| 5. Validate corpus and venue | Complete | All 43 NS and 344 FLASH cases execute in the declared venue; the largest NS cases fit 4 GiB. |
| 6. Validate distribution and publish | Implemented and checked | Fresh wheel/sdist builds, isolated installation, installed trusted manifests, public downloads, offline reuse, and native compilation pass; public data and reference artifacts have frozen identities. |

Begin with NS coverage, then FLASH replay. Keep numerical-reference quality
separate from performance-reference qualification. Do not wait for an NS
all-case solver and do not implement its performance track during this pilot.
The replay reference qualification path should be small and reusable later,
without adding an unused NS activation workflow now.

Test the existing venue against the declared pilot before retaining or changing
its CPU, memory, and deadline settings. Measure the largest admitted cases as
well as small controls. Freeze one resource contract for every compared
candidate/reference pair; local timings remain diagnostic. Do not expand pilot
work into full-corpus or trajectory capacity studies. Changes to official
resources or numerical inputs require a new calibration identity.

Update `README.md`, `SPEC.md`, `DATASET.md`, `OPERATIONS.md`, `benchmark.toml`,
CLI help, packaging, and examples together. Provide one versioned AI task
statement per pilot pair: available development data, permitted tools/libraries,
accuracy gates, C artifact contract, and run commands. Record model/system and
prompt versions, generation budget, attempt count, feedback access, and artifact
selection rule in experiment reports. Select final artifacts using development
feedback before ranked evaluation; model invocation adapters stay optional.

Implemented NS development commands:

```bash
linear-solver-bench dataset families
linear-solver-bench dataset prepare --release ns-mesh-pilot-dev \
  --output data/prepared/ns-mesh-dev
linear-solver-bench run submissions/starter.c \
  --runtime build/runtime --cases data/prepared/ns-mesh-dev \
  --output results/ns-coverage.json
linear-solver-bench score results/ns-coverage.json
```

The same CLI accepts `flash-replay-dev-pilot` and `flash-replay-ranked-pilot`
by installed name or file path. Prepare the complete split and evaluate the
candidate using the bundled calibration for that split. Reuse those reference
times when scoring; reference regeneration is an optional operation for
development or a new release. Concrete operator commands are in
[OPERATIONS.md](OPERATIONS.md). No invented commit or placeholder release ID is
an official data source.

Require the selected pair to match its verified release, prepared case set,
accuracy/scoring contracts, and reference artifact. CLI options cannot override
scientific gates or reinterpret another track's results.

## Pilot verification and completion

Implemented tests cover required/diagnostic metric separation, malformed and
rehashed input rejection, reference semantics, deterministic draws, resume/offline
cache behavior, one-case streaming and object release, failure continuation,
report/weight/reference identity binding, and Modal transport using a test
sandbox. Native development checks and public-array round trips complement
those tests. A mocked sandbox is not evidence of a full official venue run.

Completed pilot validation includes:

1. All 43 qualified NS workloads prepared from public sources and executed once
   in the official venue. The public GMRES+AMG solver passes 10/11 development
   and 14/32 ranked cases, with both large atmospheric matrices fitting 4 GiB.
2. The fixed public replay reference passes all 96 development and all 248
   ranked cases, including controls. Complete reports, single-execution timings,
   published split digests, and the runtime identity are frozen and registered.
3. Fresh wheel and source-distribution builds, isolated wheel installation,
   all four installed release IDs, public NS/FLASH single-case downloads,
   identical offline preparation, and installed native compilation pass.
   Archive scans detect no private identifiers, credential patterns, prepared
   data, or raw matrix payloads. See [PILOT_VALIDATION.md](PILOT_VALIDATION.md).

The acceptance coverage to retain as implementation evolves is:

- Numerical: hand-computed required gates; each NS backward/forward failure;
  FLASH residual failures and non-rejecting diagnostics; nonzero captured warm
  starts, zero RHS, rounding uncertainty, nonfinite values and overflow.
- Data: deterministic manufactured inputs, exact captured-array preservation,
  missing NS targets, hash corruption, invalid CSR/dtypes/dimensions, unsafe
  archive paths, duplicate case IDs, incompatible contracts and split overlap.
- Native: an actual verified solve from each pilot family, not just compilation;
  exactly one fresh evaluation process per case; status, mutation, timeout, memory and invalid-output
  failures; release iteration settings reaching the native execution.
- Scores: raw NS solved-count ties; no calibration needed for coverage; complete
  replay reference qualification with one execution per case; speedups from
  single measured times; controls, partial
  solutions, missing cases and mismatched identities handled correctly.
- Distribution: documented downloads and direct numerical loading, offline
  cache reuse, clean wheel/resource lookup, minimal container context, reference
  isolation, and no dependency on private repositories or artifacts.

Use small synthetic fixtures for offline CI and small real inputs for separate
integration checks. Run the full declared pilot in the official venue before
publishing reference timing claims. Do not add SPD, NS performance, or trajectory
execution tests to the pilot acceptance gate.

Pilot completion means NS mesh coverage and FLASH replay can both be prepared
from documented public sources, evaluated with their correct numerical gates,
and reported reproducibly from a clean installation. Coverage ranks accurate
case counts; replay reports qualified-reference speedups. Implementation may
be reviewed before data publication, but a two-track public pilot release also
requires available numerical assets, established data-use terms, frozen
manifests/resources, and a passing replay reference. Future work remains in the
separate roadmap and does not block this milestone.
