# Matrix-family roadmap beyond the first pilot

Status: future-track design roadmap, updated 13 September 2026. The active
implementation and launch gates are recorded in
[FAMILY_SUPPORT_PLAN.md](FAMILY_SUPPORT_PLAN.md). NS coverage and FLASH replay
are implemented in the shared pilot evaluator. SPD, NS performance, and FLASH
trajectory remain unsupported; their design below is not a pilot launch gate.

The public NS release currently contains two qualified development cases, not a
complete scientific inventory or ranked corpus. FLASH has 344 validated exports
for intended 96-case development and 248-case ranked splits; publication and
full public timing-reference qualification are in progress. Representative
reference checks do not establish complete-split or official venue readiness.
The pilot plan tracks these remaining launch requirements explicitly.

Pilot cases and replay reference qualification each use exactly one fresh
native process per case. The original v1 benchmark retains three repetitions.
Any repetition/median proposals below for deferred tracks are provisional and
must be decided in those tracks' own versioned contracts. They do not change
implemented pilot execution.

## Retained broader design

Add `magnetic_diffusion_flash`, `ns-mesh-pde`, and `spd-mesh` as three explicit
dataset families served by one evaluator. Give each family a documented input
and accuracy contract, with separate evaluation tracks:

| Family | Track | Primary measurement | Reference requirement |
| --- | --- | --- | --- |
| `ns-mesh-pde` | `coverage` | Number of cases solved accurately within fixed budgets | No performance reference required |
| `ns-mesh-pde` | `performance` | Speedup relative to one frozen solver that solves every case | Awaiting discovery and qualification of that solver |
| `magnetic_diffusion_flash` | `replay` | Speedup for individual captured time-step solves | A qualified reference under the replay lifecycle |
| `magnetic_diffusion_flash` | `trajectory` | Speedup over a complete trajectory | A qualified reference under the trajectory lifecycle |
| `spd-mesh` | Proposed independent-case design | Previously proposed accuracy and timing results | Qualify independently; no additional tracks requested |

Share compiler restrictions, the HYPRE runtime, numerical metrics, and trusted
verification. Give each track its own scoring and solver-lifecycle contract.
In particular, trajectory execution needs persistent state within a trajectory,
while replay and NS mesh cases are independent solves.

Introduce these contracts as a new benchmark version. Preserve the existing v1
manifests and results; do not relabel its nonsymmetric SuiteSparse corpus as
`ns-mesh-pde`. Matrix family, dataset release, submission track, and execution
venue are separate concepts and need separate identities.

The public implementation must stand alone. Its dependencies are public
libraries and independently published data artifacts. Public source files,
documentation, tests, manifests, packages, and container images must contain
only material intended for publication.

## 1. Implemented foundation and remaining extensions

The pilot now provides the shared pieces that later tracks should reuse:

- Strict float64 CSR inputs, preserved initial guesses, native int32 bounds,
  and separate manufactured/numerical/absent reference semantics.
- Restricted `solver_create` C interface, symbol auditing, pinned sequential
  HYPRE, timing of creation/setup/solve, and trusted mutation and accuracy checks.
- Explicit family/track contracts, release/prepared/report hashes, one native
  execution per pilot case, and shared local/Modal result construction.
- Public SuiteSparse and FLASH loaders, deterministic manufactured workloads,
  resumable verified downloads, offline cache reuse, and case-by-case streaming.
- Reference-free NS coverage, qualified-reference FLASH replay scoring, package
  resources, release-aware CLI commands, and per-track candidate documentation.

| Area | Implemented pilot behavior | Remaining future work |
| --- | --- | --- |
| Registry and schemas | Two independent-case pairs with strict versioned release identities | Add only requested future tracks and their actual contracts |
| References and metrics | Required NS targets, optional FLASH references, separate required/diagnostic gates | SPD qualification and any independently justified new accuracy contracts |
| Dataset preparation | NS/FLASH acquisition, new workload identities, qualification and streaming | Broader NS scientific inventory, SPD inventory, and complete trajectory acquisition |
| Native execution | One fresh process per pilot case; release iteration requests and deadlines reach execution | Ordered persistent-state trajectory ABI, ownership, and cumulative budgets |
| Scoring | NS raw coverage and all-pass FLASH replay reference qualification | NS performance activation and complete-trajectory scoring/reference lifecycle |
| Distribution | Installed resources and wheel/native CI paths | Validate each additional track's public data, package, venue, and release |

The original v1 SuiteSparse screen still has its own counts, digests, and
500–1,000,000 dimension bounds. Those filters do not define the new mesh
families. NS and SPD admission must rely on documented scientific provenance
and numerical qualification, including valid cases outside the old size range
when their released venue supports them.

## 2. Family contracts

Preserve the original operator after documented storage canonicalization:
expand declared Matrix Market symmetry, sum duplicates, remove exact zeros,
and sort `(column index, value)` pairs within each row. This canonicalizes
storage without permuting variables. Do not silently scale, transpose, shift, symmetrize, reorder,
or replace an operator with normal equations during dataset preparation.
Candidates may implement permitted mathematical transformations inside their
timed solve, but verification always uses the original serialized `A` and `b`.

| Public family ID | Scientific scope | RHS and initial guess | Reference |
| --- | --- | --- | --- |
| `magnetic_diffusion_flash` | Captured scalar magnetic-diffusion systems from FLASH | Preserve captured `b`, `x0`, and requested tolerance | Independently qualified numerical solution for diagnostics; never label it exact |
| `ns-mesh-pde` | Real, materially nonsymmetric primary linear systems with verified mesh/PDE provenance | Initially one Rademacher manufactured solution per matrix, at unit RMS; `b = fl(A x_target)`, `x0 = 0` | Manufactured target, accompanied by RHS formation and reference-quality evidence |
| `spd-mesh` | Real, exactly symmetric positive-definite mesh operators; describe any initial scalar-PDE cohort restriction explicitly | Rademacher and uniform `[-1,1]` manufactured targets, one draw each; `b = fl(A x_target)`, `x0 = 0` | Manufactured target, accompanied by qualification evidence |

`ns` means nonsymmetric, and does not restrict the family to Navier–Stokes.
SuiteSparse metadata is a discovery aid, not proof of mesh provenance,
nonsingularity, or positive definiteness. In particular, a positive diagonal
alone is not an SPD certificate. Require original-matrix symmetry checks and a
documented numerical positive-definiteness assessment, such as bounded sparse
Cholesky qualification, with failures and inconclusive cases recorded.

Use the following acceptance rules as the starting scientific specification,
then verify their feasibility before freezing a public release:

| Required quantity | `magnetic_diffusion_flash` | `ns-mesh-pde` | `spd-mesh` |
| --- | --- | --- | --- |
| Relative residual, `\|\|b-Ax\|\|2 / \|\|b\|\|2` | `<= tau * (1 + 1e-6)` | Diagnostic | `<= 1e-9` |
| Normwise backward error | `<= max(tau, 20 eps)` | `<= 1e-10` | `<= 1e-9` |
| Componentwise backward error | Diagnostic | `<= 1e-8` | `<= 1e-8` |
| Relative forward error in both L2 and Linf | Numerical-reference diagnostic | Each `<= 1e-5` | Each `<= 1e-6` |

Use the existing definitions of normwise and componentwise backward error in
`SPEC.md`. All families require valid output shape, finite solution values,
successful native status, unchanged inputs, and compliance with execution
budgets. Pilot cases require their single execution to pass. A future track
must specify its own execution count; no additional repetitions are implied.

For FLASH, small weak rows can make componentwise relative error large even
when residual accuracy is satisfactory. Preserve the diagnostic value without
adding row floors or turning it into an undocumented rejection gate. Numerical
reference disagreement is also diagnostic, rather than a forward-error
guarantee. State that limitation in the family documentation.

For a zero RHS, define the relative residual as zero only when the residual is
exactly zero; otherwise it is infinity. The first release supports zero absolute
tolerance. Reject captures requiring a nonzero absolute tolerance until a
versioned transport and evaluator contract support it. Never silently discard
such a tolerance.

Distinguish solver stopping guidance from acceptance criteria. Start mesh
feasibility runs with a solver tolerance of `1e-12`, with iteration caps of 5,000
for NS mesh and 16,000 for SPD mesh. These are proposed execution settings to
freeze after venue validation, not formulas for the terminal gates. FLASH uses
the captured relative tolerance and a separately declared iteration budget.

Compute norms with scaling where needed; detect overflowing residuals and
denominators rather than allowing an infinite denominator to produce a false
zero error. Serialize unavailable or unbounded diagnostics as JSON `null` with
an explicit reason. Invalid required metrics always fail.

## 3. A small shared implementation

Prefer a fixed registry and ordinary typed records over a plugin framework or
three copies of the evaluator. A family should select an input recipe, an
accuracy contract, and public documentation. A track selects the execution
lifecycle, scoring contract, and reference requirements. Neither should own a
duplicate compiler, sandbox implementation, or report serializer.

Implemented modules under `src/linear_solver_bench/` should be extended where
the new behavior fits; trajectory execution remains a separate future module:

```text
families.py             # Implemented pilot pairs and contracts
manifests.py            # Implemented strict release, case, and asset schemas
pilot_dataset.py        # Implemented preparation and streamed case loading
pilot_runner.py         # Implemented independent-case evaluation
pilot_scoring.py        # Implemented coverage and replay scoring
sources/
    suitesparse.py      # Implemented public acquisition and canonicalization
    flash.py            # Implemented public numerical capture loader
workloads.py            # Implemented manufactured vectors and NS qualification
trajectory.py           # Deferred: ordered sessions and whole-trajectory results
```

Retain and extend the existing `models.py`, `archive.py`, `dataset.py`,
`accuracy.py`, `verify.py`, `runner.py`, and `scoring.py`. Keep qualification
tools separate from evaluation dependencies. Sparse factorization libraries,
capture tooling, or file formats needed only to construct datasets should not
be required to run submissions against prepared numerical inputs.

The core records should express these responsibilities:

- `FamilyDefinition`: public family ID, workload kinds, accuracy contract ID,
  and permitted reference kinds.
- `TrackDefinition`: family/track pair, evaluation unit, state lifetime,
  scoring contract, execution contract, and performance-reference requirements.
- `DatasetRelease`: release ID, family manifests, split policy, ordered cases,
  public artifact identities, and execution/scoring contract references.
- `CaseSpec`: globally unique ID, matrix identity, split, provenance group,
  workload kind, asset references, and resolved solver settings. Timing roles
  belong to track membership: a replay control is still a timed trajectory step.
- `TrajectorySpec`: public trajectory ID, complete ordered step references,
  physical start/end, expected solve count, split/provenance group, and
  trajectory accuracy, lifecycle, and budget contracts. Each step identifies
  its physical time step, within-step solve ordinal, and mesh/order identity.
- `TrackRelease`: immutable family/track/release identity, allowed numerical
  corpus, readiness state, budgets, weights, and optional qualified reference.
- `ReferenceSolution`: `kind = manufactured | numerical`, values, and
  qualification identity. A missing reference is explicit and allowed only
  for FLASH diagnostics; it cannot disable required mesh forward checks.
- `ReferenceSolver`: source/configuration, runtime and compiler identities,
  qualification report, timing artifact, and exact dataset/track binding.
  This is distinct from a reference solution used to check numerical accuracy.
  It is absent for NS coverage and initially absent for NS performance.
- `EvaluationCase`: numerical candidate input plus evaluator-owned case
  specification, accuracy contract, and optional reference.
- `AccuracyContract`: named required thresholds and diagnostic definitions,
  validated against a closed set of supported contract versions.

Separate metric calculation from deciding whether a metric passes. Local and
Modal execution must call the same function to evaluate an outcome and build
its report. The transport layer should only launch work, enforce budgets, and
return native outputs.

Preserve `solver_create` and the current numerical input protocol where possible:
they already accept an arbitrary `x0`. The pilot's versioned case metadata and
references provide the independent-case foundation; future tracks need their own
trusted archive schema, not extra candidate-visible identifiers. Trajectory
adds a separate streaming protocol and lifecycle ABI, with public per-step
numerical settings and explicit ownership rules. Document the family
and its contract in the problem statement, while passing only numerical inputs
to candidate code. Resolve and validate the family/track pair before any
candidate is compiled. Keep leaderboards separate by family and track; a
coverage count and a trajectory speedup must not be collapsed into one score.

Give new hashes a source-neutral, versioned domain. Keep v1 hash domains and
decoders intact for v1 replay. Centralize effective execution settings so that
changing a configuration file cannot leave hard-coded Python and native values
silently inconsistent.

## 4. Public dataset publication

Create a new public schema instead of copying another system's manifests. Use
an explicit field allowlist. A public matrix record should contain only:

- A public source ID and URL or dataset accession, author citations, license,
  dimensions, nonzero count, and archive and canonical-array hashes.
- Scientific classification and the evidence supporting admission.
- A public provenance-group ID, split assignment, and duplicate relationships.
- Named canonicalization steps and versioned workload-generation information.

A release should bind matrix, `b`, and `x0` hashes; required reference and
qualification hashes; accuracy and execution contracts; case roles and weights;
and the ordered case list. Hashes establish identity, so an official evaluator
must also check them against the selected trusted release. A self-consistent
user-authored manifest alone must not qualify as an official benchmark run.

For SuiteSparse, acquire original archives from published upstream URLs and
retain their Matrix Market metadata. Recompute public qualification artifacts
using public scripts. Reuse scientific selection conclusions only when their
supporting public evidence is available and matches the operator bytes.

For FLASH, publish an independently documented numerical dataset with `A`, `b`,
`x0`, requested tolerances, and enough scientific provenance to understand the
capture. Put large arrays in immutable, checksummed dataset releases, outside
the code repository. Benchmark installation and evaluation must not require
building FLASH. Acquisition reproducibility and solver-evaluation
reproducibility are separate documented procedures.

### FLASH download and hosting design

The benchmark maintainers must publish and maintain this dataset: there is no
SuiteSparse accession or download endpoint assumed for these captures. The
publication host is a public, ungated Hugging Face dataset repository owned
by the benchmark organization. Hugging Face is the only required host; Zenodo
archival is an optional later addition if needed. No repository or downloadable
FLASH release has been created by this plan.

Hugging Face supports individual file downloads pinned to a full commit hash.
Use that capability for reproducible downloads; never resolve a benchmark
release through a mutable `main` branch. [Hugging Face download documentation](https://huggingface.co/docs/huggingface_hub/main/guides/download).
If Zenodo archival is added later, preserve the same payload bytes, cite the
specific version DOI, and publish its download URLs as optional mirrors.
Neither a DOI nor mirror support is required for the Hugging Face release.
[Zenodo DOI versioning](https://zenodo.org/help/versioning).

Publish the following small metadata files with the code release and dataset:

- `catalogue.json`: stable public matrix and case IDs, dimensions, nonzero
  counts, scientific provenance, license/citations, download URLs, byte sizes,
  SHA-256 checksums, and case-to-matrix relationships.
- `release.json`: frozen membership, split assignments, roles, tolerance and
  accuracy contracts, supported tracks, catalogue hash, and pinned host revision.
- `trajectories/<public-id>.json`: complete ordered case references, physical
  step and solve ordinals, start/end times, layout-change information, expected
  solve count, numerical checksums, and public capture-completeness evidence.
- `README.md` and `DATA_LICENSE`: numerical format documentation, a small
  download/load example, capture description, and established data-use terms.

An illustrative public case ID is `flash-magdiff-v1/000123`; its numerical
identity is also bound to hashes of `A`, `b`, and `x0`. Allocate IDs independently
of acquisition-directory names or private run identifiers. Repeated matrices
may share a matrix ID while different RHS/warm-start pairs have distinct case
IDs. Never reuse an ID for changed bytes.

Make each case downloadable separately as a compressed archive containing:

```text
matrix.npz    # Canonical float64 CSR, readable with scipy.sparse.load_npz
b.npy         # Captured float64 right-hand side
x0.npy        # Captured float64 initial guess
case.json     # Public case ID, tolerance, schema, scientific metadata, hashes
```

These per-case binary archives are the required public format for FLASH replay
and remain authoritative for benchmark identity. A user must be able to obtain
one case without downloading the full corpus. Additional Matrix Market exports
and bulk bundles are outside the replay pilot scope.

Publish full-trajectory bundles as well as individual cases. A trajectory
download must resolve its entire sequence and preserve every repeated entry;
content-addressed storage can avoid downloading the same array blob twice.
The existing replay selection is not a sufficient trajectory inventory: recover
or acquire missing steps and independently attest sequence completeness before
publication. A case omitted from replay merely because it is a duplicate must
still appear in its trajectory. A numerically invalid required step prevents
that entire trajectory from qualifying until resolved.

Extend `dataset prepare` with case and trajectory selectors. Resolve IDs from
the frozen catalogue, download only selected archives, verify compressed and
canonical-array hashes, extract safely, and cache by content hash. Support
resuming interrupted downloads. Mirror fallback can be added later if needed.
An offline cache must be sufficient to repeat preparation. For example:

```bash
linear-solver-bench dataset prepare --release mesh-cpu-v2 \
  --family magnetic_diffusion_flash --track replay \
  --case flash-magdiff-v1/000123 \
  --output data/prepared/flash-example
```

This is a proposed command and illustrative ID. Users should also be able to
follow a direct catalogue link and load the data without installing the
benchmark or obtaining FLASH. This package contains captured numerical inputs;
it does not contain FLASH source, raw simulation dumps, private acquisition
receipts, or evaluator reference solutions.

For the fully public release, publish numerical inputs for both development and
ranked splits. `ranked` denotes the evaluation split, not a secret dataset.
Preserve the provenance separation and document prior exposure, but do not
claim that publicly downloadable cases are hidden. Any future evaluation with
unreleased systems must be a separate, explicitly identified track. Keeping
reference vectors private does not make a published `A,b,x0` system secret.

Publication is complete only after an unauthenticated machine can download a
single case and a complete split, verify all identities, and solve the example
using the documented public commands. Require established redistribution terms
for every included numerical artifact before uploading it. Dataset files remain
outside Git; their catalogue, checksums, and format stay reviewable in the code
repository.

Publish development inputs and worked examples. Retain ranked evaluation
references and any ranked manufactured-vector key in trusted operator storage.
Domain-separate new manufactured draws by release, family, matrix, RHS kind,
and draw index. Freeze the algorithm, normalization, tool versions, vector
hashes, and serialized bytes. Do not put a library version into an implicit
seed recipe that changes the workload after a dependency upgrade.

Changing a seed or key produces different workloads and requires new numerical
qualification, prepared identities, and calibration. A manufactured target is
not necessarily the exact solution of the rounded serialized `A,b` system:
record formation error and verify that this uncertainty is small relative to
the forward-error requirement. For NS mesh, the acceptance limits are fixed
release requirements, not values calculated from each matrix's condition
estimate. Require offline qualification evidence at one tenth of each required
error limit and distinguish empirical reference-solve metrics from certified
bounds. A passing float64 reference alone does not certify the exact
stored-system solution's distance from the target. Audit sensitive cases using
higher-precision residuals/refinement or verified error bounds and document
remaining uncertainty. These are preparation checks, not additional candidate
executions.

Use provenance groups to split data before tuning or baseline selection. Keep
related simulation steps, mesh refinements, parameter variants, exact
duplicates, and documented near-duplicates together. Check overlaps across
families, tracks, and previously published development sets. A ranked FLASH
trajectory must not contribute steps to the replay development split: apply
one provenance-group split consistently to both FLASH tracks. New RHS draws do
not make a previously exposed matrix unseen. Public matrix availability also means
the benchmark cannot claim to rule out model pretraining exposure.

Publish a development split, a ranked split, and an exclusion ledger. Treat
quarantine as a disposition, not a scored split. FLASH correctness controls
must retain their role even if their initial guess already satisfies accuracy;
exclude them from replay speed aggregation and report their pass rate separately.
Every required step, including such controls, contributes to trajectory time.

## 5. Track execution, references, and scores

### NS mesh: coverage

The implemented pilot evaluates every case in the frozen NS mesh split once
in one fresh native process. Count a case as solved only when that execution
satisfies the fixed accuracy and execution requirements. A future performance
track must preserve the numerical contract and declare its execution lifecycle
separately; it must not change existing coverage results.

```text
solved_count = sum_i single_execution_passed_i
coverage_fraction = solved_count / expected_case_count
ranking_key = -solved_count
```

Use the raw solved-case count as the primary score, as requested. Equal counts
tie: execution time does not break coverage ties. Report timings, failure
reasons, matrix/group breakdowns, and optionally a provenance-balanced coverage
diagnostic, but these do not change the ranking. Candidate failure on one case
does not prevent accurate solutions on other cases from counting.

Coverage requires no performance reference solver or timing calibration.
Numerical targets and qualification evidence are still required to enforce the
accuracy gates. Freeze independent per-case execution budgets before evaluating
candidate submissions. Preserve hard cases even when available baselines fail.

### NS mesh: performance

Measure speedup against **one versioned reference solver that solves every
case** in the selected frozen split, under the same accuracy and execution
contract. Use the same numerical corpus as coverage; do not remove cases or
change their tolerances to obtain a passing reference.

The track initially has `status = awaiting_reference`, with no reference solver
or timing artifact. In the later NS performance phase, implement its execution, qualification,
scoring, CLI, and report schemas before discovering the reference. An official performance evaluation must fail clearly before
compilation when no qualified reference is registered; it must not fabricate
reference times or fall back to another scoring rule.

When a suitable solver is discovered:

1. Freeze its public source, configuration, compiler/runtime identity, and
   reference-solver ID. One reference artifact may contain adaptive algorithms,
   but all cases must run that artifact without case-name dispatch or an
   evaluator choosing a different best solver for each case.
2. Run it on the entire frozen split in the official venue for all three
   repetitions. Independently verify every solution and retain every result.
3. Require complete case membership, all passing repetitions, finite positive
   timings, and matching dataset, accuracy, lifecycle, and resource identities.
   Missing cases, timeouts, or numerical failures reject qualification.
4. Publish the qualification report and frozen reference timings, then create
   an active performance-track release bound to those artifacts. Coverage
   remains usable throughout and retains its existing release identity.

For case `i`, let `R_i` be the qualified reference median and `C_i` the candidate
median, both including creation, setup, and solve. Define `speedup_i = R_i/C_i`.
Require candidates to solve all scored cases and mandatory controls for an
official aggregate speedup. If any fails, retain coverage and timing diagnostics
but return `eligible = false` and `speedup = null`; never average only the
successful cases or replace failed timings with a purported speedup.

Use a weighted geometric mean of case speedups, with equal mass for provenance
groups, equal matrices within each group, and equal RHS workloads within each
matrix. Freeze these weights independently of timing and candidate results:

```text
speedup = exp(sum_i weight_i * log(R_i / C_i))
ranking_key = -speedup
```

The weights sum to one. A value above one means faster than the reference.
Changing the reference, resource budget, corpus, or accuracy contract creates
a new performance release; scores against different references are not mixed.
Use an all-passing synthetic reference in CI to test activation in that phase. Discovering
the real reference is a separate research task, not a prerequisite for
implementing the track or releasing coverage.

### FLASH: replay

Treat each case as one captured magnetic-diffusion solve at a time step. If a
physical step contains multiple magnetic solves, expose their identities and
ordering explicitly rather than silently combining or omitting them.

Each solve starts with its captured `A,b,x0,tau`, a fresh native process, and a
new solver object. Time factory creation, setup/preconditioner construction,
and solve. Destroy the object afterward. No solver state, preconditioner,
workspace, or candidate solution carries from one replay case to another.
The implemented pilot uses exactly one execution and its complete measured
solve time, including one execution per case during reference qualification.

Qualify one public reference solver across the replay split under this exact
lifecycle. A stock GMRES+AMG configuration is a starting candidate for that
qualification, not an assumed valid timing artifact. Compute per-case speedup
and a frozen provenance-balanced geometric mean, with equal weight for source
trajectories within a provenance group and scored steps within a trajectory.
Apply the same complete-accuracy eligibility rule as NS performance. Run
designated correctness-only controls but exclude their times from replay speed
aggregation. Require the reference and candidate to pass those controls.

Replay measures independent time-step solves. Its speedups cannot serve as
trajectory reference times or be summed to claim trajectory performance.

### FLASH: trajectory

The agreed scope is an **ordered captured sequence with persistent solver
state**. The driver streams each captured `A_t,b_t,x0_t,tau_t` in order to one
candidate process and solver state. Captured numerical inputs remain fixed;
candidate outputs do not regenerate future matrices or RHS vectors and do not
advance a live FLASH simulation. The reported quantity is cumulative linear
solver time across the complete captured trajectory. It makes no claim about
full simulation wall-clock speedup or error propagation in a recoupled PDE run.

Allow candidate-owned workspaces, preconditioners, Krylov recycling data, and
past numerical information to persist between steps. Supply the captured initial
guess at every step; a solver may choose an alternative initial iterate using
its own past state as part of its timed work. Do not automatically replace the
captured guess with the previous candidate solution, particularly when the mesh
dimension or degree-of-freedom ordering changes.

Require every magnetic solve from the declared simulation start through its
declared terminal time, in recorded order, including repeated matrices,
duplicate inputs, controls, substeps, and multiple solves per physical step.
A sampled set of replay cases or a cropped window is not a complete trajectory.
If the available capture omits necessary solves, acquire a complete trajectory
before enabling this track. Missing or invalid steps cannot be repaired by
dropping them. Duplicate numerical blobs may share storage, but all their
positions in the sequence remain executable.

Provide a small, separate versioned trajectory ABI, leaving the existing
single-case submission ABI unchanged. The interface needs three operations:

```text
create(state)                         # Once at trajectory start
step(state, A, b, x, solve_settings)    # Once per captured solve, in order
destroy(state)                        # Once at trajectory end
```

`x` is initialized from captured `x0`; `solve_settings` carries the current
relative/absolute tolerances and iteration cap. The step operation includes
all candidate decisions about retaining, rebuilding, or updating state, and
must work across changed values, patterns, dimensions, and orderings. Freeze
exact C signatures and exported symbols in a public header before implementation.
Audit the trajectory artifact against its own small entrypoint allowlist while
retaining the existing capability restrictions.

Specify HYPRE object ownership explicitly. Current-step inputs are borrowed;
retaining an old matrix for a cached preconditioner requires an explicit
read-only retain/release handle supported by the driver. Retained matrices and
candidate-owned allocations count toward the process memory limit, and the
driver audits retained inputs for mutation. Vector histories must be copied
into candidate-owned workspace. Do not keep every previous HYPRE object alive
implicitly or assume that a matching dimension means matching mesh ordering.
Test this lifetime contract with actual HYPRE preconditioners before freezing it.

Do not expose future step payloads to candidate code. Stream only the current
input, enforce its integrity and accuracy, then advance. Candidate state resets
between trajectories and between repetitions. Each repetition runs the entire
trajectory from a new process; a sandbox cannot silently restart the candidate
mid-trajectory or carry caches from another trajectory.

For each full-trajectory repetition, measure:

```text
T = create_time + sum_t step_time_t + destroy_time
```

This includes candidate setup, reuse decisions, matrix-retention calls, workspace
changes, preconditioner rebuilds, solves, and candidate cleanup. Exclude trusted
input decoding, HYPRE input assembly, transport, solution extraction, and
independent verification. Report these excluded phases separately. Compared to
single-case replay, final candidate destruction is explicitly included here
to account for the complete persistent solver lifecycle. Enforce a cumulative
native-time budget, a total process watchdog, and the trajectory memory limit.

Independently verify each returned step solution against that step's original
`A_t,b_t` and FLASH accuracy gates. A failed, missing, repeated-out-of-order, or
nonfinite output invalidates the trajectory; abort it and record the remaining
steps as unexecuted. No speedup is assigned to a partial trajectory. Correctness
controls and duplicates contribute their execution time in this track because
they are part of the full trajectory.

Qualify a reference using the same trajectory ABI, streamed inputs, allowed
reuse, memory limits, and timed boundary. A published stateless adapter can
serve as a reference, but all of its per-step creation/setup/destruction must
be charged inside its step callbacks. Qualify it by executing full sequences;
do not construct its timing from replay measurements.

For trajectory `j`, take the median of three complete totals for each solver:

```text
trajectory_speedup_j = median(T_reference_j) / median(T_candidate_j)
```

Do not sum per-step medians or average per-step speedups. Aggregate eligible
trajectory ratios geometrically, giving equal mass to provenance groups and
equal trajectories within each group. Require every trajectory and every step
to pass all repetitions for an official aggregate; report partial completion
only as diagnostics. Publish step traces and whole-trajectory totals so setup
amortization and regressions remain visible.

### SPD and shared venue rules

SPD retains its proposed single-case, coverage-first and calibrated-time
ordering, with its separate accuracy contract and qualification. No additional
SPD tracks are introduced by this update. Keep its results separate from the
four explicitly requested track leaderboards. Existing public v1 results and
ranking remain unchanged as well.

For clarity, the retained SPD proposal uses provenance/matrix/RHS-balanced
weights, ranks weighted passing coverage first, and uses the penalized geometric
mean of normalized times second. Its frozen normalization can use the best
passing median from a predeclared public baseline portfolio, or a budget value
explicitly labelled as such when none passes. Failures contribute twice their
deadline divided by that normalization. This SPD cost diagnostic is not the NS
performance or FLASH speedup formula, and budget normalization must never be
accepted by those speedup tracks.

Do a venue feasibility sweep before choosing final v2 limits. The current
4 GiB, 120-second venue is not evidence that all new workloads fit. Measure
peak memory, wall time, initialization overhead, and retained-state growth.
Start independent-solve feasibility runs with memory headroom up to 32 GiB and
cases up to 300 seconds; these are experiment allowances, not leaderboard
limits. Measure whole trajectories to establish their separate total budgets.

Load one prepared numerical input at a time. Budget total sandbox lifetime for
repetitions, compilation, transfer, and cleanup. Partition long evaluations
only at independent case or trajectory boundaries; never split a trajectory
to fit a shorter sandbox lifetime. Use the same frozen runtime and resources
for a candidate/reference comparison. Venue changes invalidate calibration.

Provide understandable public GMRES/FGMRES and PCG examples with appropriate
preconditioners. Any trusted stock-HYPRE calibration executable needs a separate
build route and published configuration; it must not broaden the candidate
allowlist. A reference solver is public benchmark infrastructure and must not
carry private source dependencies or identifiers.

Bind results to family, track, evaluation unit, release, prepared inputs,
accuracy/lifecycle/scoring contracts, reference identity when applicable,
compiler/runtime, venue, repetitions, budgets, and frozen weights. Infrastructure
failure invalidates a run and follows a fixed retry policy; it never changes
the scored case set. Candidate failures remain explicit outcomes. Local runs
are diagnostics and cannot be submitted as official venue results.

## 6. Open-source boundary

Prepare publication material through an explicit export into a clean directory.
Export numerical data and public scientific evidence, and write new public
schemas and documentation around them. Keep any one-time source-conversion
tool and source-to-public mapping outside the public repository. Public loaders
must consume the public format without knowing where it was assembled.

Exclude private package imports, repository names and paths, deployment or
account IDs, storage names, experiment logs, prompts, model outputs, historical
receipts, and private source hashes. Do not copy whole directories and then
attempt string replacement. Recompute hashes after constructing public
metadata; retain original upstream artifact hashes where they identify public
scientific sources. Preserve required third-party copyright notices.

Audit the actual release contents: source archive, wheel, container build
context, embedded archive metadata, generated documentation, and Git history.
Use public schema/import allowlists in CI. Keep any check containing literal
private identifiers in private release tooling, so the check itself cannot
publish those identifiers. Legitimate attribution is not an internal detail
and must not be removed.

Keep code and data licensing separate. SuiteSparse states that its matrices
are CC BY 4.0 and requires preserving matrix-specific metadata and identifying
modifications. [SuiteSparse licensing and redistribution guidance](https://sparse.tamu.edu/about).

FLASH's published license restricts redistribution of its source code and
components. Do not bundle that source or assume capture patches are freely
redistributable. Establish the redistribution terms for the numerical dataset
and its accompanying metadata before publishing it; source-code terms alone
do not establish the terms for those outputs. [FLASH license agreement](https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node3.html).

## 7. Future implementation sequence and acceptance criteria

This retained sequence describes the broader, deferred scope. Several shared
pieces are already delivered by the pilot, as listed in section 1; extend them
instead of recreating those changes. The pilot's remaining publication and
qualification gates are tracked only in the active implementation plan.
Each additional track should be a small, separately reviewable extension.

| Step | Deliverable | Acceptance criteria |
| --- | --- | --- |
| 1. Freeze the design and inventory | Family/track specifications, source list, provenance groups, asset availability, trajectory completeness, and redistribution status | Every proposed case/trajectory has a disposition; public data locations and unresolved rights are explicit; no assumed complete corpus |
| 2. Separate v1 from v2 configuration | Typed release/contracts, family/track registry, strict manifests, readiness states, version dispatch | Existing v1 behavior remains unchanged; unknown family/track pairs fail closed; NS performance represents a missing reference explicitly |
| 3. Extend reference and accuracy handling | New trusted archive schema, shared metrics and family gate selection | Tiny analytic cases exercise all contracts, warm starts, and reference types; missing mesh truth cannot pass |
| 4. Add public dataset preparation | SuiteSparse and FLASH loaders, generator, complete trajectory manifests, qualification commands | Prepare one small case per family and one complete sequence from public artifacts; hashes, order, completeness and provenance validate; captured inputs are preserved |
| 5. Complete independent-case execution | Streaming case loading, iteration limits, shared local/Modal reporting | NS coverage, FLASH replay and SPD native solves succeed; local and Modal verdicts agree; candidate sees no reference or identifying case metadata |
| 6. Implement trajectory execution | Versioned ABI and streaming driver, retained-object ownership, whole-sequence timing, trace validation | Actual HYPRE reuse works across changing inputs; all steps execute in order; state resets between repetitions; missing steps or memory/time failures invalidate the trajectory |
| 7. Implement track scores and reference qualification | Raw coverage count, qualified-reference speedups, activation command, track-bound reports | Coverage runs with no timing reference; a synthetic all-passing reference activates NS performance in CI; incomplete references and partial candidate speedups are rejected |
| 8. Freeze datasets and venue | Qualified case/trajectory releases, official execution budgets and available reference reports | All admitted numerical workloads qualify, full trajectories fit, splits are consistent across tracks, and each speedup track is explicitly active or awaiting data/reference |
| 9. Validate distribution and publish | Documentation, package resources, CI, citations and notices, readiness reporting | Clean checkout/wheel/container runs without sibling repositories; content audit passes; documented coverage/replay/trajectory paths work; missing NS reference is handled clearly |
| 10. Activate NS performance after discovery | Public all-case reference artifact, complete official qualification, new active track release | Every existing NS case passes every reference repetition; published speedups bind to the frozen reference; coverage data and results remain unchanged |

After the pilot, a first additional vertical slice can use a small SPD mesh
case with its own scientific qualification and the existing SuiteSparse loader.
A trajectory slice then needs a tiny complete sequence with repeated and changed
matrices, a stateful solver, and a reference adapter. Validate each new track
before publishing that track's corpus; these extensions do not delay release of
independent-case pilot data. Implement NS performance's activation mechanism
when implementing that deferred track. Discovering its real all-case reference
remains a separate milestone.

Update `README.md`, `SPEC.md`, `DATASET.md`, `OPERATIONS.md`, `benchmark.toml`,
packaging metadata, and CLI help together. Add three concise family documents,
track specifications, and both independent-case and trajectory submission
examples. Label scientific background,
candidate instructions, and evaluator operation procedures clearly.

For comparisons of AI models and systems, publish a versioned task statement
per family and track: available development cases, permitted tools and libraries,
accuracy requirements, artifact format, and evaluation commands. Record the
model/system version, prompt version, generation budget, number of independent
attempts, development feedback access, and final artifact-selection rule in
experiment reports. Select artifacts using development feedback before ranked
evaluation. Keep model invocation adapters optional and separate from the
numerical evaluator so any AI system can submit the same C artifact.

Proposed CLI behavior:

```bash
linear-solver-bench dataset families
linear-solver-bench dataset summary --release mesh-cpu-v2
linear-solver-bench dataset list --release mesh-cpu-v2 --family spd-mesh --split dev
linear-solver-bench dataset prepare --release mesh-cpu-v2 --family spd-mesh \
  --split dev --output data/prepared/spd-mesh-dev
linear-solver-bench run submissions/pcg.c --runtime build/runtime \
  --cases data/prepared/spd-mesh-dev --output results/pcg-spd.json

linear-solver-bench run submissions/gmres.c --family ns-mesh-pde \
  --track coverage --runtime build/runtime --cases data/prepared/ns-mesh-dev \
  --output results/ns-coverage.json
linear-solver-bench track status --family ns-mesh-pde --track performance

linear-solver-bench dataset prepare --release mesh-cpu-v2 \
  --family magnetic_diffusion_flash --track trajectory --split dev \
  --trajectory flash-trajectory-v1/000001 --output data/prepared/flash-trajectory
linear-solver-bench run submissions/trajectory.c \
  --family magnetic_diffusion_flash --track trajectory --runtime build/runtime \
  --cases data/prepared/flash-trajectory --output results/flash-trajectory.json

# After an NS reference has been discovered:
linear-solver-bench reference qualify references/ns-reference.c \
  --family ns-mesh-pde --track performance --runtime build/runtime \
  --cases data/prepared/ns-mesh-ranked --output references/ns-qualification.json
```

These commands and filenames are proposed interfaces, not currently available
features. Require an explicit `--track` when a family has multiple tracks. Runs
resolve that selection against the verified release and prepared manifest; the
CLI cannot override a gate, change case membership, use a reference from another
track, or silently reinterpret an existing release. The official reference
qualification command must use the registered official venue; local evidence
cannot activate an official track.

## 8. Verification needed for future-track completion

Existing pilot tests already cover many independent-case numerical, data,
streaming, report, and scoring checks. The list below describes the broader
acceptance surface after the additional families and lifecycles are implemented;
it does not establish that those future checks already run.

- Numerical tests: hand-computed residual/backward/forward errors; a perturbed
  mesh solution failing each required gate; FLASH diagnostic errors that do not
  reject an otherwise valid solution; zero RHS, nonzero warm starts, rounding
  uncertainty, nonfinite values, and overflowing denominators.
- Data tests: exact public-array preservation, deterministic generation,
  corrupted hashes, wrong dimensions/dtypes, unsafe paths, duplicate case IDs,
  missing references, unknown fields, and an intentionally overlapping split.
- Native tests: a successful solve from every family; status, mutation, timeout,
  memory-limit and invalid-output failures; repeated fresh-process execution;
  observed iteration settings matching the release.
- Track tests: identical NS accuracy verdicts in coverage and performance;
  coverage ranks raw case counts and retains time ties; absence of a timing
  reference never blocks coverage; absent, failing, mismatched, or incomplete
  references cannot activate speedup scoring; a synthetic qualified reference
  exercises activation and replacement via a new release.
- Trajectory tests: persistent workspace/preconditioner reuse; retained HYPRE
  matrix lifetimes and mutation checks; value/pattern/dimension/order changes;
  per-step tolerance changes; duplicates and controls retained; no future-input
  exposure; wrong order, missing terminal step, failure at the last step, and
  state leaking across repetitions; memory and cumulative-time limits.
- Scoring tests: exact reference/candidate speedup examples, invalid partial
  solves, failed controls, and distinct timing rules for replay and trajectory.
  Verify median-of-whole-trajectory sums, not sum-of-step-medians, and that no
  replay timing or budget fallback is accepted as a trajectory reference.
  Reject mismatched family/track/release/runtime/contract identities.
- Distribution tests: install from a wheel outside the repository; exercise
  package resource lookup; build a sandbox from its minimal allowlisted context;
  confirm no evaluator references are included; run all documented examples.

Keep routine CI small and offline using synthetic fixtures. Run public small
real-data integration checks separately from the full official-venue sweep.
The existing CI native job only compiles and audits the starter; extend it to
execute and independently verify actual solves.

Completion of this broader roadmap means all three families and the four
requested tracks have documented public formats, execution contracts, numerical checks,
scoring, distribution support, and tested readiness/activation behavior. Dataset
and leaderboard readiness are tracked separately: NS coverage can open before
NS performance, and NS performance remains awaiting reference until the real
all-case qualification succeeds. FLASH trajectory requires complete captured
sequences plus a separately qualified trajectory reference; independent replay
results do not establish its readiness. The main release dependencies are data
availability and redistribution terms, defensible split coverage, numerical
qualification, and measured venue capacity.
