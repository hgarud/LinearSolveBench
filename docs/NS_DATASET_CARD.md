# NS Mesh PDE — dataset card

**49 original sparse operators · 24 provenance groups · 19 development / 30 ranked cases**

NS Mesh PDE is the nonsymmetric mesh/PDE family of **LinearSolverBench**, a
benchmark for AI models and systems that produce linear solvers. Each case asks
a solver to solve `A x = b` accurately within a fixed resource budget. This
release measures **coverage: how many cases pass every accuracy check**.

**NS means nonsymmetric**, not exclusively Navier–Stokes. The matrices span
fluid flow and transport, reservoir simulation, diffusion and thermal problems,
biomedical models, semiconductor devices, and solid mechanics.

| Property | Frozen release |
| --- | --- |
| Release identity | `ns-mesh-cohort-v2` |
| Family / track | `ns-mesh-pde` / `coverage`; CLI alias `ns_mesh_pde` |
| Status | All 49 workloads qualified; both complete venue runs validated |
| Numerical data | Real square float64 sparse matrices; one manufactured RHS per matrix; zero initial guess |
| Matrix dimensions | 240–1,602,111 unknowns |
| Numerical nonzeros per matrix | 1,760–23,487,281, after storage canonicalization |
| Development | 19 cases in 9 provenance groups; fully reproducible with public inputs |
| Ranked | 30 cases in 15 disjoint groups; public matrices, operator-held RHS key and targets |
| Candidate budget | 2 CPU cores, 4 GiB, 90 seconds per case; 5,000-iteration request |
| Execution | One fresh native process per case; one execution, no retained solver state |
| Distribution | Original matrices from SuiteSparse; manifests, recipes, and evidence in this repository |
| Licensing | SuiteSparse matrices: CC BY 4.0; benchmark code: Apache-2.0 |

> **Representation limit.** The development split covers every shared declared
> application and method category, but has no confirmed finite-volume case with
> 10,000–100,000 unknowns. The ranked wind-tunnel case falls in that range.
> Coverage of categories does not establish equal solver difficulty or represent
> every nonsymmetric PDE problem.

[Access](#access-and-quickstart) · [Data format](#what-one-case-contains) ·
[Curation](#where-the-data-come-from) · [Splits](#development-and-ranked-splits) ·
[Evaluation](#accuracy-and-evaluation) · [Limitations](#limitations-and-responsible-interpretation) ·
[Citation](#license-attribution-and-citation)

## Intended use

Use development cases to design, debug, and compare solver-generation systems.
Reserve ranked cases for final evaluation and disclose any prior use of their
matrices or published results. A benchmark submission is one C source file
implementing the public `solver_create` interface with approved HYPRE operations
and preconditioners. See the [candidate task](task-ns-coverage.md) and
[specification](SPEC.md) for the executable contract.

The dataset also supports separately labeled numerical-method research. Runs
with different solvers, tolerances, resources, or workloads should report those
changes and their own results. This release does not measure speedup against an
all-case reference, physical simulation fidelity, or generalization to unseen
PDE families. It has no training split.

## Access and quickstart

All 49 **original matrices** are publicly downloadable from the
[SuiteSparse Matrix Collection](https://sparse.tamu.edu/). The two release
manifests specify exact matrix IDs, archive URLs, byte sizes, SHA-256 checksums,
canonical matrix identities, citations, and numerical qualification:

| Manifest | Cases | Original compressed archive bytes |
| --- | ---: | ---: |
| [ns-mesh-cohort-v2-dev](../data/ns-mesh-cohort-v2-dev.json) | 19 | 562,789,795 |
| [ns-mesh-cohort-v2-ranked](../data/ns-mesh-cohort-v2-ranked.json) | 30 | 253,520,899 |

These byte totals describe downloads, not peak RAM or prepared-data disk usage.
No raw matrix payloads are committed to this repository. The benchmark's
Hugging Face dataset hosts the separate FLASH family; it is not the download
source for these NS matrices.

From a checkout, prepare development data with Python 3.12 or newer:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
linear-solver-bench dataset list --release ns-mesh-cohort-v2-dev
linear-solver-bench dataset prepare \
  --release ns-mesh-cohort-v2-dev --output data/prepared/ns-dev
```

Preparation downloads and verifies the original archives, canonicalizes sparse
storage, generates the fixed workload, and checks its numerical identity against
frozen qualification. It does not repeat offline factorizations. Downloads are
cached by content; use `--offline` to reuse an already populated, verified cache.
Use `--cache` to choose its location.

For a small development check, add `--case ss-225-rademacher-0` to preparation
and choose a separate output directory. A subset cannot receive a complete-split
coverage score. The [README](../README.md#get-started-with-ns-coverage) continues
from preparation to compiling, running, and scoring a candidate.

**Ranked access differs from development.** Anyone can download the ranked
matrices and inspect the manifests and scalar qualification evidence. Preparing
the exact frozen ranked RHS and targets requires the operator-held key matching
the manifest commitment. It is not included in the package. Public matrix
availability therefore does not imply public access to every ranked numerical
input. See [operator preparation](OPERATIONS.md#prepare-and-run-ns-coverage).

## What one case contains

A case is one square operator and one numerical workload, not a mesh, a complete
simulation, or a collection of alternative right-hand sides. For example, the
first development case is `ss-225-rademacher-0`: SuiteSparse ID **225**,
[HB/orsirr_1](https://sparse.tamu.edu/HB/orsirr_1), with 1,030 unknowns and 6,858
numerical nonzeros, assigned to provenance group `reservoir-ors`.

The source archive contains the original Matrix Market file and source
metadata. Benchmark preparation produces `manifest.json` and one trusted
schema-v2 compressed NumPy archive per case. The prepared manifest maps each
case ID to its relative archive path, hashes, dimensions, and qualification.
This archive has the benchmark's own schema; it is not a SciPy `save_npz` file.

| Prepared archive field | Type / shape | Meaning |
| --- | --- | --- |
| `schema_version` | uint32 scalar | `2` |
| `case_id` | String scalar | Public workload identifier |
| `row_offsets` | uint64, `(n + 1,)` | Zero-based CSR row offsets |
| `column_indices` | uint32, `(nnz,)` | Zero-based columns, strictly increasing within each row |
| `values` | float64, `(nnz,)` | Original operator coefficients after canonicalization |
| `b` | float64, `(n,)` | Stored manufactured RHS |
| `x0` | float64, `(n,)` | All zeros |
| `tolerance` | float64 scalar | Relative stopping request, `1e-12` |
| `reference_kind` | String scalar | `manufactured` |
| `x_star` | float64, `(n,)` | Manufactured target, with entries −1 or +1 |

The native HYPRE build uses signed 32-bit indices, so admitted dimensions and
nonzero counts must also fit that limit. Metadata and `x_star` remain with the
trusted evaluator; candidate code receives the numerical solve inputs and
stopping requests. The trusted NPZ archive is never given to a candidate.

Read one prepared development case using the package's validating decoder:

```python
import json
from pathlib import Path

from linear_solver_bench.archive import decode_system

root = Path("data/prepared/ns-dev")
manifest = json.loads((root / "manifest.json").read_text())
record = manifest["cases"][0]
case = decode_system((root / record["relpath"]).read_bytes())
A = case.public.matrix.to_scipy()
b, x0 = case.public.b, case.public.x0
print(case.case_id, A.shape, A.nnz)
# ss-225-rademacher-0 (1030, 1030) 6858
```

## Where the data come from

The cohort reuses operators deposited by scientific contributors in SuiteSparse.
Source review identifies their PDE, mesh, discretization, and related models;
the benchmark does not regenerate their original simulations. Public citations
and grouping rationales are recorded in the
[source descriptors](analysis/ns-cohort-v2-descriptors.json), with computed
matrix properties in the [inventory](analysis/ns-cohort-v2-inventory.json).
Missing or ambiguous spatial methods remain labeled `unknown`.

Cohort v2 retains the 43 operators from the historical `ns-mesh-pilot` and adds
six: HB/plskz362, Hohn/fd12, Janna/CoupCons3D, Janna/Transport,
Grueninger/windtunnel_evap3d, and Goodwin/Goodwin_095. The additions address
documented gaps and incompatible group-coverage constraints. The
[cohort review](NS_COHORT_V2.md) records their primary sources and selection
history. All additions and the final split were selected before the fresh v2
targets and candidate evaluations; solver outcomes are not selector inputs.

Admission and preparation follow three separate steps:

1. **Scientific and matrix admission.** Require a real, square, original PDE
   operator with documented provenance and material nonsymmetry:
   `max(abs(A - A.T)) > 1e-12 * max(abs(A))`. Roundoff-scale asymmetry alone is
   insufficient. The original PLSK ocean operator is admitted with its historical
   eigenmode purpose disclosed; its squared PLAT relatives are excluded.
2. **Storage and workload construction.** Expand declared Matrix Market
   symmetry, sum duplicate coordinates, remove exact zeros, and sort each row's
   `(column index, value)` pairs. This sorting does not permute variables.
   No scaling, shift, projection, normal equations, or replacement operator is
   introduced. A versioned HMAC recipe generates one fixed Rademacher target
   `x_target` with entries ±1 and exactly unit RMS; float64 arithmetic forms
   `b = fl(A x_target)`, with `x0 = 0` and draw index zero.
3. **Numerical qualification.** Bind feasibility evidence to the complete stored
   system identity and require a tenfold margin against each acceptance limit.
   A new target or numerical operator changes that identity and needs new
   qualification. Inputs are not redrawn or altered to obtain a passing witness.

The target is a benchmark label, **not necessarily the exact solution of the
stored `A,b`**: rounding when forming `b` can separate the two. Qualification
audits RHS formation and tests feasibility relative to the gates; the available
guarantees depend on the recorded evidence type.
Of the 49 workloads, 46 use refined sparse-LU witnesses, one uses a PyAMG/GMRES
witness with inexact refinement, and two use exact strict-row-dominance
certificates with separate measured feasibility witnesses. The first two methods
provide empirical numerical evidence; they do not certify forward-error bounds.
See [qualification details](DATASET.md#pilot-releases) and the preserved
[offline attempt history](../data/releases/ns-cohort-v2-offline-reference-attempts.json).

## Development and ranked splits

The split unit is a **provenance group**: related models, studies, meshes, or
generators stay together. Shared authorship alone does not establish a shared
model. Unresolved overlap is documented and related cases are grouped
conservatively; group separation is not proof of complete code independence.

The [frozen selection policy](analysis/ns-cohort-v2-policy.json) first requires
both splits to contain each known application, spatial method, size band, and
structural regime supported by at least two groups. Single-group categories
go to development. The same both-split rule applies to method-by-size
combinations supported by multiple groups; single-group combinations are
reported as gaps. Unknown method labels do not establish known-method coverage.

Among feasible partitions, a deterministic selector balances five matrix
features, known-tag proportions, and development size. Its features are
`log10(n)`, `log10(nnz/n)`, relative Frobenius nonsymmetry, zero-diagonal fraction,
and weakly diagonally dominant row fraction. These are structural descriptors,
not difficulty labels. Exhaustive search considers 16,777,216 group partitions,
of which 6,531 satisfy the hard rules. The chosen partition has 19 development
and 30 ranked cases; the [selection report](analysis/ns-cohort-v2-selection.json)
also records alternative feasible partitions and remaining gaps.

| Size band: unknowns `n` | Development | Ranked |
| --- | ---: | ---: |
| `n < 10,000` | 11 | 15 |
| `10,000 <= n < 100,000` | 4 | 12 |
| `100,000 <= n < 1,000,000` | 3 | 1 |
| `n >= 1,000,000` | 1 | 2 |
| **Total** | **19** | **30** |

| Source-supported spatial method | Development | Ranked |
| --- | ---: | ---: |
| Finite element | 8 | 13 |
| Finite difference | 4 | 4 |
| Finite volume | 2 | 3 |
| Boundary element | 1 | 0 |
| Unknown | 5 | 10 |

Method counts may overlap within a case; columns need not sum to split sizes.
They describe spatial methods, so finite differences used only for time
integration do not count as spatial finite differences.

<details>
<summary>Complete group membership and SuiteSparse IDs</summary>

Each ID below identifies one matrix and one workload. Resolve names, source
URLs, and archive hashes through the linked release manifests or `dataset list`.

| Development provenance group | SuiteSparse IDs |
| --- | --- |
| `crack-log-grid` | 917, 918 |
| `driven-cavity-fem` | 380, 389, 395 |
| `finite-volume-rans` | 2334, 2335 |
| `janna-coupled-fe` | 2647, 2649 |
| `nonlinear-diffusion` | 1916 |
| `reservoir-ors` | 225, 227 |
| `semiconductor-devices` | 830, 832, 833 |
| `square-inlet-flow` | 444, 445 |
| `torso-electrophysiology` | 896, 897 |

| Ranked provenance group | SuiteSparse IDs |
| --- | --- |
| `airfoil-flow` | 1319 |
| `atmospheric-elliptic` | 2265, 2267 |
| `cavity-vorticity` | 828, 829 |
| `circle-surface-poisson` | 2562, 2565, 2567 |
| `femlab-langemyr` | 864, 925, 926, 927 |
| `fidap-flow-transport` | 428, 431, 435 |
| `goodwin-flow-transport` | 446, 2817, 2822, 2825 |
| `heart-quasistatic` | 891 |
| `platzman-ocean-model` | 231 |
| `reservoir-sherman` | 246 |
| `reservoir-steam` | 251, 252 |
| `thermal-fem` | 377 |
| `unstructured-euler` | 820, 821, 822 |
| `unstructured-stokes` | 1862 |
| `windtunnel-evaporation` | 2815 |

</details>

Reproduce the split without matrix downloads, numerical targets, or solver runs:

```bash
python tools/select_ns_cohort.py \
  --inventory docs/analysis/ns-cohort-v2-inventory.json \
  --policy docs/analysis/ns-cohort-v2-policy.json \
  --output results/ns-cohort-v2-selection.json
```

## Accuracy and evaluation

The trusted verifier uses the original stored matrix and RHS. For the returned
solution `x`, it computes `r = b - A x` and requires all four acceptance gates:

| Required metric | Candidate acceptance | Offline qualification |
| --- | --- | --- |
| Normwise backward error | `<= 1e-10` | `<= 1e-11` |
| Componentwise backward error | `<= 1e-8` | `<= 1e-9` |
| Relative target error, L2 | `<= 1e-5` | `<= 1e-6` |
| Relative target error, Linf | `<= 1e-5` | `<= 1e-6` |

```text
normwise backward error = ||r||inf / (||A||inf ||x||inf + ||b||inf)
componentwise backward error = max_i |r_i| / (|A||x| + |b|)_i
relative target error, L2 = ||x - x_target||2 / ||x_target||2
relative target error, Linf = ||x - x_target||inf / ||x_target||inf
relative residual = ||r||2 / ||b||2     [diagnostic only]
```

The requested relative stopping tolerance `1e-12` is guidance for the solver,
not a replacement for these acceptance gates. Output must also be finite,
correctly shaped, successful, leave inputs unchanged, and meet the deadline.
The [specification](SPEC.md#independent-accuracy-checks) defines zero-denominator
and nonfinite behavior.

Compile the submission once and execute each case once in a fresh process.
Factory creation, setup/preconditioner construction, and solve count toward
measured time. Cases are loaded one at a time. Coverage is `solved_count` out
of the complete expected split, with equal counts tied and timing diagnostic.
Every case has equal weight; provenance groups are not equally weighted.
Solver failures and timeouts remain in the denominator. Missing cases or
infrastructure failures prevent a valid complete-split score.

### Published example results

The public [GMRES+AMG candidate](../submissions/gmres_amg.c) has these frozen
results in the two-core, 4 GiB, 90-second Modal venue:

| Split | Solved | Coverage | Complete report |
| --- | ---: | ---: | --- |
| Development | 11 / 19 | 57.9% | [Development validation](../data/releases/ns-cohort-v2-dev-validation.json) |
| Ranked | 14 / 30 | 46.7% | [Ranked validation](../data/releases/ns-cohort-v2-ranked-validation.json) |

All 49 cases ran once. CoupCons3D and Goodwin_095 timed out; their failures were
retained without retry. There were no crashes or infrastructure failures.
The [runtime manifest](../data/releases/cpu-runtime-v2.json) binds the environment.
These results demonstrate a valid partial-coverage evaluation, not an all-case
reference or a claim about the best attainable score. Offline qualification may
use more resources and different methods than a candidate; its success does not
guarantee success under the candidate budget.

## Limitations and responsible interpretation

- **Conditional representation.** This is a curated SuiteSparse cohort, not a
  random sample of all PDE operators. Public availability, source documentation,
  numerical admission, and the chosen taxonomy shape its scope. The split policy
  balances declared descriptors; it does not establish matched condition numbers,
  equal solver difficulty, or coverage of every joint regime.
- **Known gaps.** The ranked 40,816-row finite-volume wind-tunnel case has no
  development counterpart in its size/method combination. Semiconductor and
  boundary-element categories are development-only. Other development-only
  method/size combinations are finite differences and boundary elements at
  100,000–1,000,000 unknowns, and finite elements at at least 1,000,000 unknowns.
  Unknown methods remain unknown. The [cohort review](NS_COHORT_V2.md#frozen-partition-and-remaining-gaps)
  gives exact support counts.
- **Public-data exposure.** Matrix IDs, coefficients, provenance, and example
  results are public and may have appeared in model training or prior research.
  Group separation and an operator-held ranked RHS key do not establish an
  uncontaminated test set. Disclose benchmark exposure and ranked-set tuning;
  use development data for iterative design.
- **Manufactured workloads.** One ±1 target and zero initial guess per matrix
  do not reproduce native physical forcing, multiple RHS behavior, warm starts,
  or trajectories. Target-based forward errors are not universally certified
  errors against the exact stored-system solution. The release publishes the
  evidence type and its uncertainty for every case.
- **Score interpretation.** Groups have different case counts, so related cases
  can contribute more total score than a singleton group. Compare scores only
  under the same frozen workloads and enforced venue. Single-execution timings
  are diagnostics, not estimates of timing variability; local runs are labeled
  `local-uncontrolled` and do not establish official resource enforcement.
- **Data subjects and context.** The records are numerical operators and
  scientific provenance, not person-level observations. Some sources model
  biomedical systems; that label describes the equations, not a clinical
  prediction task. Public contributor metadata are retained for attribution.
  This card does not certify the absence of sensitive information in every
  upstream source archive.

## Reproducibility and maintenance

The release manifests are the authority for ordered cases, scientific admission,
workload identities, gates, budgets, and scoring. Their `manifest_sha256` values
are semantic content identities checked by the package, not byte checksums of
the pretty-printed JSON files:

```text
ns-mesh-cohort-v2-dev
9927ea72bac431d871b0547b966f9ce78b550ca0d1285c7060cdcb2e3e688ecc

ns-mesh-cohort-v2-ranked
a86a2d42765364bcadd48fdef22dc2f0900d04e98fe4fe49fe551b84d591ad52
```

Per-case source-archive hashes, canonical-matrix hashes, and complete-system
hashes distinguish download integrity from numerical identity. Hashes bind
content; a self-authored consistent manifest does not establish an official
release. The package's trusted registry fixes the accepted release digests.

All 49 prepared systems reproduced the same canonical-matrix and complete-system
hashes with NumPy 2.0.2, SciPy 1.14.1, and Python 3.12.11 on macOS arm64. The
[reproduction receipt](../data/releases/ns-cohort-v2-preparation-reproduction.json)
records the comparison. This validates that tested preparation environment,
not every platform or the reproducibility of every offline solver's metrics.

LinearSolverBench maintainers curate the manifests, code, and this card;
SuiteSparse contributors and maintainers provide the original matrices and
source records. Report issues through the benchmark repository's issue tracker,
following the [contribution guidelines](../CONTRIBUTING.md) and including release
ID, case ID, the observed discrepancy, and public evidence.
Do not attach ranked keys or trusted target archives to public issues.

Documentation corrections can update this card. Changes to numerical inputs,
split membership, or evaluation contracts require a new frozen release and
relevant qualification/validation; they must not silently replace this release.
The historical 43-case `ns-mesh-pilot` remains available with its original 11/32
split. Its scores are not directly comparable to cohort v2: both membership and
manufactured workloads differ. There is no scheduled refresh cadence.

## License, attribution, and citation

SuiteSparse states that its matrices are distributed under **CC BY 4.0** and
requests preservation of matrix metadata and matrix-specific citations. See the
[collection's license and attribution guidance](https://sparse.tamu.edu/about#license).
Prepared workloads document their modifications: canonical float64 CSR storage
and a manufactured RHS/target instead of a native physical workload. Retain
original source attribution and identify prepared workloads as derivatives when
redistributing them. Benchmark software is separately
[Apache-2.0 licensed](../LICENSE); see [dataset notices](../data/NOTICE.md).

For publications, cite the benchmark using [CITATION.cff](../CITATION.cff), include
the code revision, release ID, split, and manifest digest, and cite SuiteSparse
and the original matrix sources:

- Timothy A. Davis and Yifan Hu. *The University of Florida Sparse Matrix
  Collection*. ACM Transactions on Mathematical Software, 38(1), 2011.
  [DOI: 10.1145/2049662.2049663](https://doi.org/10.1145/2049662.2049663).
- Scott P. Kolodziej et al. *The SuiteSparse Matrix Collection Website
  Interface*. Journal of Open Source Software, 4(35), 1244, 2019.
  [DOI: 10.21105/joss.01244](https://doi.org/10.21105/joss.01244).
- Matrix-specific references in the original Matrix Market headers and the
  [source descriptors](analysis/ns-cohort-v2-descriptors.json).

*Card last reviewed: 2026-09-13. Organization follows the motivation,
composition, provenance, use, and maintenance questions in
[Datasheets for Datasets](https://arxiv.org/abs/1803.09010) and the
[Hugging Face dataset-card guidance](https://huggingface.co/docs/hub/datasets-cards).
These are documentation references, not papers introducing this cohort.*
