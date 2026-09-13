# NS mesh cohort v2

The replacement release `ns-mesh-cohort-v2` contains 49 qualified original
operators in 24 reviewed provenance groups. It retains all 43 matrices from
`ns-mesh-pilot` and adds six public SuiteSparse operators. Selection,
manufactured-workload qualification, and execution validation are separate
steps; source admission alone does not make this an active benchmark release.
The original 11-development/32-ranked release remains reproducible, with its
limitations recorded in the [earlier split review](NS_SPLIT_REVIEW.md).

Current status: all 49 fixed workloads are qualified, both releases are
frozen, and both complete venue runs are validated. The public GMRES+AMG
candidate passes 11/19 development and 14/30 ranked cases.

## Scientific scope and added sources

This family uses real, materially nonsymmetric original PDE discretization
operators for manufactured linear-system workloads. Source provenance must
identify the PDE or mesh construction. Preparation canonicalizes storage and
preserves the operator; it does not square, shift, project, scale, or replace it.
A source may originally have used the operator for eigenmode analysis. That
history is disclosed, and the benchmark makes no claim to reproduce its native
physical right-hand side.

| Added operator | SuiteSparse ID | Unknowns | Numerical nonzeros | Source-supported spatial method and purpose |
| --- | ---: | ---: | ---: | --- |
| [HB/plskz362](https://math.nist.gov/MatrixMarket/data/Harwell-Boeing/platz/platz.html) | 231 | 362 | 1,760 | Finite-difference shallow-water ocean operator, originally used for natural-mode eigenanalysis |
| [Hohn/fd12](https://sparse.tamu.edu/Hohn/fd12) | 917 | 7,500 | 28,462 | Finite-difference single-material crack problem on a logarithmic grid |
| [Janna/CoupCons3D](https://sparse.tamu.edu/Janna/CoupCons3D) | 2647 | 416,800 | 17,277,420 | Finite-element, fully coupled three-dimensional poroelastic consolidation |
| [Janna/Transport](https://sparse.tamu.edu/Janna/Transport) | 2649 | 1,602,111 | 23,487,281 | Tetrahedral finite-element, density-driven coupled flow and transport |
| [Grueninger/windtunnel_evap3d](https://sparse.tamu.edu/Grueninger/windtunnel_evap3d) | 2815 | 40,816 | 803,978 | Coupled Navier–Stokes/Darcy evaporation with cell-centred finite volumes and a marker-and-cell scheme |
| [Goodwin/Goodwin_095](https://sparse.tamu.edu/Goodwin/Goodwin_095) | 2825 | 100,037 | 3,226,066 | Finite-element Navier–Stokes and related transport equations |

The original skew-symmetric PLSK ocean matrix is retained. The related squared
PLAT operators are excluded. CoupCons3D uses finite differences in **time**;
that does not establish spatial finite-difference coverage. Numerical nonzero
counts exclude explicitly stored zeros after canonicalization.

The Janna pair stays in one group because the sources share a specific study
and their mesh/code overlap is unresolved. Goodwin_095 joins the existing
Goodwin group, so all four related Goodwin cases stay together. Hohn/fd12 and
the existing fd15 case share one crack-model group. These groups
describe reviewed model, study, mesh, and generator exposure. A common author
or collector alone does not force unrelated physical models into one group.
The [source descriptors](analysis/ns-cohort-v2-descriptors.json) record the
evidence, unknown methods, and remaining independence uncertainties.

The input pool was revised from 47 to 48 operators before new manufactured
targets or candidate executions. A joint size-by-method review identified the
need for another finite-element source in the 100,000-to-1,000,000 band.
Botonakis/FEM_3D_thermal2 was investigated first, but its original coefficients
differ from their transpose only at roundoff scale. Its maximum relative
asymmetry is approximately `1.11e-16`, below the unchanged `1e-12` admission
threshold. The [screening record](analysis/ns-cohort-v2-screening.json) preserves
that exclusion and archive identities. Goodwin_095 passes the same gate and
provides another source group in that size-by-method stratum. No candidate
convergence or timing result entered either decision.

The 48-operator pool then exposed incompatible finite-difference support
constraints: small finite-difference cases, medium finite-difference cases,
and all-zero-diagonal cases tied three groups into pairwise opposite-split
requirements. Adding fd12 supplies another reviewed group in the small
finite-difference stratum, while keeping both Hohn matrices together. This
produces the 49-operator input pool without relaxing the coverage policy or
using a solver outcome to select a case.

## Selection contract

The [frozen policy](analysis/ns-cohort-v2-policy.json) applies to the
[matrix-only inventory](analysis/ns-cohort-v2-inventory.json). It keeps each
provenance group indivisible and retains every admitted operator. Development
must contain 25–50% of cases, with one third as the balance target; ranking must
retain at least eight groups.

Hard coverage rules precede numerical balance. Each known application, spatial
method, size band, and declared structural regime with at least two source
groups must occur in both splits. A category represented by only one group is
development-only. Unknown method labels never establish known-method coverage.
Size bands are below 10,000; 10,000 to below 100,000; 100,000 to below 1,000,000;
and at least 1,000,000 unknowns. The zero-diagonal and weak-row-dominance
fractions each distinguish zero, strictly between zero and one, and one.

The same both-split requirement applies to spatial-method-by-size combinations
supported by at least two groups. Single-group combinations are reported as
limitations, because marginal coverage does not establish every joint regime.
The weak-dominance feature is a floating-point structural diagnostic, separate
from exact certificates used in numerical qualification.

Among feasible group partitions, the selector minimizes an equally weighted
combination of marginal Wasserstein distance, multivariate energy distance,
known-tag incidence differences, and deviation from the development-count
target. The five matrix features are `log10(n)`, `log10(nnz/n)`, relative
Frobenius nonsymmetry, zero-diagonal fraction, and weakly diagonally dominant
row fraction. Feature ranges, energy normalization, rounding, and tie-breaking
are explicit in the policy. The selector examines every group partition;
candidate outcomes and reference solutions are not selector inputs.

These are declared benchmark design choices, not statistical estimates of an
optimal split. Meeting them establishes coverage conditional on this input
pool, taxonomy, and source review. It does not establish representativeness of
all nonsymmetric PDE systems, equal solver difficulty, complete code
independence, or coverage of every physical and numerical combination.

## Frozen partition and remaining gaps

The [selection report](analysis/ns-cohort-v2-selection.json) records all
16,777,216 group partitions considered and the 6,531 that satisfy the hard
rules. The selected partition contains 19 development cases in nine groups and
30 ranked cases in 15 groups. Development contains 38.8% of the cases.

| Unknowns | Development | Ranked |
| --- | ---: | ---: |
| Below 10,000 | 11 | 15 |
| 10,000 to below 100,000 | 4 | 12 |
| 100,000 to below 1,000,000 | 3 | 1 |
| At least 1,000,000 | 1 | 2 |

Development groups are `crack-log-grid`, `driven-cavity-fem`,
`finite-volume-rans`, `janna-coupled-fe`, `nonlinear-diffusion`, `reservoir-ors`,
`semiconductor-devices`, `square-inlet-flow`, and `torso-electrophysiology`.

Ranked groups are `airfoil-flow`, `atmospheric-elliptic`, `cavity-vorticity`,
`circle-surface-poisson`, `femlab-langemyr`, `fidap-flow-transport`,
`goodwin-flow-transport`, `heart-quasistatic`, `platzman-ocean-model`,
`reservoir-sherman`, `reservoir-steam`, `thermal-fem`, `unstructured-euler`,
`unstructured-stokes`, and `windtunnel-evaporation`. The report lists every
SuiteSparse matrix ID in each split.

All shared application and spatial-method categories, all four size bands,
both structural features' three regimes, and every method-by-size combination
supported by multiple groups occur in both splits. In particular, both sides
have all-zero-diagonal cases and finite-element cases in the
100,000-to-1,000,000 band. Semiconductor devices and boundary elements have only
one supporting source group each and are development-only; they are not
claimed as ranked regimes represented by independent development sources.

Four single-group size-by-method combinations remain one-sided:

| Combination | Development | Ranked | Source group |
| --- | ---: | ---: | --- |
| Finite volume, 10,000 to below 100,000 unknowns | 0 | 1 | Wind-tunnel evaporation |
| Finite difference, 100,000 to below 1,000,000 | 2 | 0 | Torso electrophysiology |
| Boundary element, 100,000 to below 1,000,000 | 1 | 0 | Torso electrophysiology |
| Finite element, at least 1,000,000 | 1 | 0 | Coupled flow/transport |

The ranked 40,816-row finite-volume case therefore has no development case
with the same size-and-method combination. Development's confirmed
finite-volume cases are smaller than 10,000 rows. Adding an independent source
in that joint regime is future cohort work; known-method marginal coverage
does not remove this limitation. Unknown spatial methods are kept unknown.

The selected weighted objective is approximately `0.0524530`: marginal
Wasserstein `0.0559072`, normalized energy `0.0128166`, tag-incidence difference
`0.0866667`, and development-fraction difference `0.0544218`. These numbers
are descriptive comparisons under the declared objective, not quality
certificates. The report includes the next five feasible partitions, whose
scores range from approximately `0.0527210` to `0.0551028`, with development
counts from 17 to 20. It also reports nearest development feature neighbours
for each ranked case. Those diagnostics do not establish equal difficulty or
robustness to every alternative policy.

Reproduce the exhaustive selection from a checkout with the package installed:

```bash
python tools/select_ns_cohort.py \
  --inventory docs/analysis/ns-cohort-v2-inventory.json \
  --policy docs/analysis/ns-cohort-v2-policy.json \
  --output results/ns-cohort-v2-selection.json
```

This command reads the frozen public feature inventory and policy. It needs
neither matrix downloads nor manufactured targets, RHS keys, or solver runs.
The expected `selection_sha256` is
`606ca7f230261c72d49287da8d4600ea123707e302928ecff13e3b374d6eb024`.

The underlying matrix properties and descriptors can also be audited directly
from public archives:

```bash
python tools/audit_ns_split.py \
  --release data/ns-mesh-cohort-v2-dev.json \
  --release data/ns-mesh-cohort-v2-ranked.json \
  --descriptors docs/analysis/ns-cohort-v2-descriptors.json \
  --output results/ns-cohort-v2-matrix-audit.json
```

Use `--cache` to choose writable archive storage and `--offline` to require
cached archives. This audit processes the original operators without ranked
RHS keys or candidate execution. Its descriptive distances do not replace the
selection policy or establish a universal acceptable threshold.

## Release qualification

Selection is followed by fresh deterministic manufactured targets under the
new release identity. Each fixed workload must meet the existing tenfold
qualification margins without changing its operator, target, or acceptance
thresholds. Removing a case that cannot be qualified requires an explicit
inventory/release revision; choosing another target because it passes is not
permitted. A failed reference algorithm does not establish that a workload is
infeasible: another documented reference method may qualify the same fixed
inputs, with the failed attempt retained.

The 49 completed qualifications comprise 46 independent sparse-LU witnesses
with one higher-precision residual correction each, one PyAMG-preconditioned
GMRES witness with two inexact corrections, and two exact
row-diagonal-dominance certificates. The solver witnesses establish empirical
feasibility at the tenfold margins; they are not verified forward-error bounds.
The dominance method separately certifies the exact stored-system solution's
forward discrepancy and measures verifier metrics at the manufactured target.
All four qualification gates retain their tenfold margins: normwise backward
error `1e-11`, componentwise backward error `1e-9`, and both target-relative
forward errors `1e-6`.

Offline qualification has a separate resource budget from candidate evaluation.
The large-case qualification venue allows up to 32 GiB and one hour; candidate
cases still have two CPU cores, 4 GiB, and a 90-second deadline. CoupCons3D's
fixed ILU reference attempt broke down, and a full-LU qualification on the same
stored inputs passed using 8.68 GiB and 268.54 seconds. That is valid offline
feasibility evidence, not a claim that this reference satisfies the candidate
budget. Failure receipts remain part of the qualification record. Neither
the target nor the acceptance gates changed between reference methods.

Transport qualified with the versioned
`independent-pyamg-sa-gmres-refined-v2` method. It keeps the initial relative
GMRES tolerance at `1e-13`, uses `1e-2` for each of two correction solves, and
keeps absolute tolerance zero throughout. The hierarchy and zero initial
guesses are fixed, and the manufactured target is never a reference-solver
input. Inexact correction solves are a standard iterative-refinement option;
the attainable improvement depends on the system and is checked independently
here. [Carson and Higham, 2018](https://eprints.maths.manchester.ac.uk/2629/1/cahi18.pdf)

Its final normwise and componentwise backward errors were approximately
`1.11e-16` and `3.31e-16`; L2 and Linf target errors were `2.06e-14` and
`5.65e-14`. Qualification took 174.38 seconds with 2,471,686,144 bytes peak RSS.
These are offline reference measurements, not candidate performance results.
The [public attempt history](../data/releases/ns-cohort-v2-offline-reference-attempts.json)
also preserves the earlier ILU timeout and the stopped PyAMG v1 attempt, which
had no final accuracy verdict. The cause of that attempt's slow correction
convergence was not established; no Transport-specific floating-point accuracy
floor is claimed. V2 changes the reference algorithm's inner stopping rule,
without changing the frozen system, cohort membership, or acceptance gates.

All 49 freshly generated numerical systems reproduce exactly with the minimum
supported NumPy 2.0.2 and SciPy 1.14.1 on Python 3.12.11, macOS arm64. The
[preparation receipt](../data/releases/ns-cohort-v2-preparation-reproduction.json)
binds each original archive, canonical matrix, and resulting numerical-system
digest to this inventory. This verifies deterministic preparation for those
versions and platform; it does not substitute for numerical qualification or
claim reproducibility on every possible platform.

## Candidate venue validation

The public `gmres_amg.c` candidate completed both full split evaluations in
the declared Modal venue:

| Split | Cases | Passed | Timeouts |
| --- | ---: | ---: | ---: |
| Development | 19 | 11 | 1 |
| Ranked | 30 | 14 | 1 |

All 49 cases received exactly one fresh native execution. CoupCons3D in
development and Goodwin_095 in ranking reached their 90-second deadlines and
remain failed coverage cases. Neither was retried or removed. There were no
crashes, infrastructure failures, or candidate retries. The complete
[development report](../data/releases/ns-cohort-v2-dev-validation.json) and
[ranked report](../data/releases/ns-cohort-v2-ranked-validation.json) preserve
every outcome and input/runtime identity. Both scores meet the registered
official release and venue requirements.

Transport, with 1,602,111 unknowns, passed in 5.81 seconds of native elapsed
time, including 2.59 seconds of setup and 3.22 seconds of solving, under the
normal two-CPU/4-GiB/90-second contract. This is a measured candidate result;
the offline qualification algorithm and its 174.38-second preparation cost
are separate. NS ranking uses solved counts, with equal counts tied. These
partial coverage results do not establish an all-case performance reference.
