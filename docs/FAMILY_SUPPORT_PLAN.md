# First pilot: NS mesh coverage and FLASH replay

Status: implementation and release plan, updated 13 September 2026. The shared
pilot code is implemented. NS has a qualified two-case development release;
FLASH has a validated numerical inventory and public export. Publication,
complete reference qualification, and official venue validation remain launch
gates. The snapshot below distinguishes implemented support from a finished
scientific benchmark release.

## Implementation snapshot

| Area | Current evidence | Remaining work |
| --- | --- | --- |
| Family and execution contracts | Two explicit pairs; `ns_mesh_pde` alias; schema-v2 release/prepared/report identities; one native execution per pilot case | Freeze each published split's exact manifest and venue |
| Numerical verification | Fixed NS gates, FLASH residual/backward gates, explicit reference semantics, stable ratios, JSON-safe diagnostics, v1 compatibility | Audit sensitive cases when expanding the declared NS inventory |
| NS coverage | `ns-mesh-dev-pilot`: two public SuiteSparse matrices, manufactured workloads, bound qualification evidence, native evaluation, reference-free coverage | Declare and qualify a scientifically useful pilot inventory and provenance-based ranked split |
| FLASH inputs | 344 admitted cases exported as standalone archives; original CSR, RHS, and initial-guess bytes verified by the public loader | Finish Hugging Face publication, freeze commit-pinned manifests, verify clean public downloads |
| FLASH reference and scores | Public `gmres_amg.c` reference candidate; representative checks across 13 source runs; all-pass qualification and weighted scoring implemented | Qualify that fixed solver on every case/control in both declared splits and freeze venue-matched timings |
| Execution and distribution | Local and Modal case streaming, release settings propagation, hard Modal resource configuration, package resources, CLI/task documentation and regression tests | Complete full-split official venue runs and final checkout/wheel/container/release audit |

The NS development matrices are `DRIVCAV/cavity01` (317 unknowns) and
`FEMLAB/poisson2D` (367 unknowns). They demonstrate real mesh/PDE inputs and the
complete coverage path. Two small cases do not establish broad scientific
coverage, difficult-case behavior, or a ranked NS benchmark. No larger NS
inventory or ranked release is claimed complete, and the existing generic v1
inventory must not be used as a substitute.

The intended FLASH splits are 96 development scored cases and 248 ranked cases
(160 scored, 88 correctness controls). Public inputs cover 13 source runs in 10
provenance groups. Captured solutions and stored independent numerical references
pass the public accuracy gates for all 344 cases. This is feasibility evidence,
not qualification of a public timing solver under the new runtime and venue.
The destination is the ungated
[Hugging Face dataset](https://huggingface.co/datasets/hgarud/LinearSolveBench);
its existence alone does not establish an immutable, reproducible release.

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
configuration is a candidate for qualification, not assumed passing evidence.
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
| `submissions/gmres_amg.c` | Public reference candidate for qualification; complete-split passing evidence is still required |

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

The NS development sources are downloadable and numerically qualified. FLASH
assets have been obtained, checked, and exported; immutable public downloads
and release manifests still need final publication verification. NS coverage
can become available independently if replay publication or reference
qualification is delayed. A pilot claiming both tracks complete requires both
working public data paths and a qualified replay timing reference. A two-case
NS development release must remain labeled with that limited scope.

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

| Step | Status | Evidence and remaining acceptance criteria |
| --- | --- | --- |
| 1. Freeze contracts and inventory | Contracts implemented; inventory release work remains | Two pairs and public schemas are explicit. NS has only a two-case development inventory; its broader pilot and ranked split remain undeclared. FLASH inventory is fixed for publication. |
| 2. Deliver NS coverage end to end | Implemented for the development pilot | Real SuiteSparse preparation, bound qualification, one fresh process per case, failure-preserving reports, and calibration-free scoring are available. |
| 3. Deliver FLASH replay end to end | Loader and evaluator implemented; public path pending publication verification | Captured arrays round-trip unchanged across 344 exports; diagnostic metrics remain diagnostic. Release downloads need immutable commit identities. |
| 4. Complete replay qualification and scoring | Scoring implemented; full reference qualification pending | Representative reference checks are insufficient: one fixed public solver must pass every expected case and control in each split under the comparison venue. |
| 5. Validate corpus and venue | Pending full declared-split runs | Complete NS inventory qualification and split review; run the full FLASH reference; measure resource needs and establish local/official verdict agreement. |
| 6. Validate distribution and publish | Resources, CI paths, and user/task documentation implemented; final release audit pending | Verify downloads from a clean installation and freeze trusted release digests, source/configuration, calibration, packaging, and content audit. |

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
linear-solver-bench dataset prepare --release ns-mesh-dev-pilot \
  --output data/prepared/ns-mesh-dev
linear-solver-bench run submissions/starter.c \
  --runtime build/runtime --cases data/prepared/ns-mesh-dev \
  --output results/ns-coverage.json
linear-solver-bench score results/ns-coverage.json
```

The same CLI accepts a published FLASH manifest by installed name or file path.
Once that manifest is available, prepare its complete split, evaluate the fixed
public reference candidate, and run `calibrate` on its passing report. Evaluate
other candidates under the same runtime and venue, then `score` their reports
with that calibration. Concrete operator commands are in
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

The following remain required before claiming a complete public benchmark:

1. Declare the scientifically justified NS pilot inventory and its development
   and ranked splits; qualify every frozen manufactured workload. Preserve the
   current two-case set as a clearly labeled development release until then.
2. Finish FLASH publication, verify the immutable download commit, and freeze
   both split manifests and exact trusted release digests.
3. Qualify one fixed public replay reference on all 96 development and all 248
   ranked cases, including controls, under the declared comparison venue.
   Freeze complete reports and single-execution timings.
4. Complete official venue resource and correctness checks for every declared
   split, then verify the final checkout, wheel, source distribution, container
   context, and documentation against the release identities.

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
