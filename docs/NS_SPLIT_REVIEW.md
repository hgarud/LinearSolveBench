# NS development/ranked split review

Status: **the current 11-development/32-ranked partition is provisional for
representativeness**. Its 43 workloads remain numerically qualified and its full
Modal execution remains validated. Those checks establish feasible accuracy
requirements and working evaluation; they do not establish that development
feedback covers the scientific and numerical regimes encountered in ranking.
The representative-split requirement is not yet satisfied.

This review uses the frozen manifests, the
[matrix-property audit](analysis/ns-split-audit.json), and the
[public-source descriptors](analysis/ns-source-descriptors.json). The audit
reads matrix properties without evaluating candidate solvers. Source descriptors
record application and discretization evidence from original public metadata,
with unresolved claims labeled explicitly. These descriptors are a draft for
scientific review; the counts below reflect the evidence established so far,
and unresolved classifications require further source review.

## Gaps in the current partition

Development contains four provenance groups and ranked contains 17. Keeping
groups disjoint limits related-data exposure, but does not by itself balance
applications, discretizations, dimensions, or matrix structure.

| Number of unknowns | Development cases | Ranked cases | Available provenance support |
| --- | ---: | ---: | --- |
| Below 10,000 | 7 | 17 | Multiple groups |
| 10,000 to below 100,000 | 4 | 11 | Multiple groups |
| 100,000 to below 1,000,000 | 0 | 2 | Both cases belong to one torso group |
| 1,000,000 or more | 0 | 2 | Both cases belong to one atmospheric group |

The largest development matrix has 26,068 unknowns; the largest ranked matrix
has 1,489,752. Neither of the two largest size bands can appear on both sides
through a different partition of the current inventory while keeping provenance
groups disjoint: each band has only one group. Adding independent groups is
necessary if the replacement split is to represent both bands in development.

The source review also finds one-sided application coverage:

| Broad application tag | Development cases | Ranked cases |
| --- | ---: | ---: |
| Fluid flow and coupled transport | 4 | 17 |
| Elliptic, diffusion, or thermal systems | 3 | 6 |
| Reservoir simulation | 0 | 5 |
| Biomedical systems | 0 | 3 |
| Semiconductor devices | 3 | 0 |
| Solid mechanics or fracture | 1 | 1 |

These are coarse descriptions of the documented application, not solver
difficulty classes. Development has no reservoir or biomedical group. All three
semiconductor cases occupy one development group. A change in case counts alone
cannot address the absence of independent provenance support for a category.

Confirmed discretization evidence differs as well:

| Source-supported method | Development cases | Ranked cases |
| --- | ---: | ---: |
| Finite element | 8 | 10 |
| Finite volume | 0 | 4 |
| Finite difference | 0 | 6 |
| Boundary element | 0 | 1 |
| Unresolved by the reviewed metadata | 3 | 12 |

One ranked case has both finite-difference and boundary-element tags, so these
rows are not mutually exclusive. An unresolved method is not evidence that a
missing method is covered. A group introduction or generic collection label
must not override more specific matrix metadata.

Numerical structure also has uncovered extremes. The largest zero-diagonal
fraction in development is about 0.2628, while ranked includes
[Hohn/fd15](https://sparse.tamu.edu/Hohn/fd15), whose entire diagonal is zero.
The audit additionally compares dimension, average nonzeros per row, relative
nonsymmetry, and the fraction of weakly diagonally dominant rows. These traits
help identify gaps even when broad application names overlap.

## What the matrix audit establishes

[audit_ns_split.py](../tools/audit_ns_split.py) computes five features from each
canonical matrix: `log10(n)`, `log10(nnz/n)`, the relative Frobenius difference
between the matrix and its transpose, the zero-diagonal fraction, and the
weakly diagonally dominant row fraction. It reports split ranges, medians, and
per-feature Wasserstein distances normalized by the pooled feature range.

Reproduce the audit directly from public SuiteSparse matrices, without ranked
RHS keys or candidate executions:

```bash
python tools/audit_ns_split.py \
  --release data/ns-mesh-pilot-dev.json \
  --release data/ns-mesh-pilot-ranked.json \
  --descriptors docs/analysis/ns-source-descriptors.json \
  --output results/ns-split-audit.json
```

Use `--cache` to choose an archive cache and `--offline` to require cached
archives. Operators may instead supply the two existing prepared directories
with repeated `--prepared` arguments. Both input paths reproduced the same
audit for all 43 matrices. Matrices are processed individually; targets, RHS
values, qualification outcomes, and solver results do not enter the comparison.

These distances describe the observed marginal distributions. There is no
universal acceptable-distance threshold in this audit. An improved average
distance can still hide an absent application, an uncovered extreme, or a
different joint distribution. Matching individual feature distributions would
not establish equal solver difficulty. The audit provides no condition-number,
nonnormality, convergence, or forward-error certificate. Its floating-point
dominance diagnostic is separate from the exact dominance certificates used
to qualify two manufactured workloads.

## Requirements for a replacement split

The development count and number of development groups are open design choices;
neither 11 cases nor four groups is a target to preserve. Define the intended
scientific scope and preparation/evaluation budget before selecting a new
inventory and partition.
Declare the application/discretization taxonomy, size bands, and structural
regimes before selection so those definitions cannot be adjusted afterward to
make the chosen split appear balanced.

1. Keep related matrices, refinements, duplicates, and source runs within one
   provenance group, and keep those groups disjoint across development and
   ranking. Distinct collection group names alone do not establish independence.
2. For each broad application, confirmed discretization, size band, and
   numerical structural regime declared as represented in ranking, require
   independent development support wherever the release claims representative
   development coverage. Explicitly review important combinations, such as
   large finite-volume systems or zero-diagonal flow operators. Unknown labels
   do not satisfy a coverage requirement.
3. Identify requirements that the available provenance groups cannot satisfy.
   Acquire and qualify independent groups when necessary. Any scientific scope
   restriction must be explicit and justified before selection; moving related
   cases across the split boundary is not a solution.
4. Freeze a documented balance objective, multivariate review procedure, and
   resource budget before choosing the partition. Separate required regime
   coverage from descriptive distribution summaries, explain feature scaling
   and weights, and inspect tails and joint structure rather than optimizing
   only a mean marginal distance. Report unresolved tradeoffs and sensitivity
   to reasonable alternative partitions. No numerical cutoff is implied here.
5. Select from scientific provenance and matrix properties without candidate
   convergence, speed, or coverage outcomes. Qualify frozen manufactured inputs
   and validate their execution budgets separately; solver failures must remain
   valid coverage outcomes rather than becoming a selection rule.
6. Publish the proposed membership and evidence for review. A new partition
   requires a new release identity, fresh deterministic RHS generation and
   qualification, updated registered digests, and complete venue validation.
   Preserve the original frozen manifests and reports for reproducibility.

## Acquisition candidates

Three public matrices have useful source evidence for further investigation.
They have not been downloaded or numerically qualified for this replacement
release, and are not admitted benchmark cases:

| Candidate | SuiteSparse ID | Unknowns | Reported nonzeros | Source evidence and possible contribution |
| --- | ---: | ---: | ---: | --- |
| [Janna/Transport](https://sparse.tamu.edu/Janna/Transport) | 2649 | 1,602,111 | 23,487,281 | Real nonsymmetric, tetrahedral finite-element discretization of coupled flow and transport; a potential additional large-matrix provenance group |
| [Janna/CoupCons3D](https://sparse.tamu.edu/Janna/CoupCons3D) | 2647 | 416,800 | 17,277,420 | Real nonsymmetric, fully coupled three-dimensional poroelastic consolidation with spatial finite elements; potential support in the 100,000-to-1,000,000 band |
| [Grueninger/windtunnel_evap3d](https://sparse.tamu.edu/Grueninger/windtunnel_evap3d) | 2815 | 40,816 | 803,978 | Real nonsymmetric coupled Navier–Stokes/Darcy system with finite-volume and MAC discretization; potential additional finite-volume support |

The CoupCons3D source identifies finite differences for time discretization;
that does not establish a spatial finite-difference regime. Its reported
17,277,420 numerical nonzeros exclude 5,044,916 explicitly stored zeros. Use
matrix-specific descriptions and symmetry records when they differ from a
generic group introduction.

Transport and CoupCons3D share authors and a cited paper. Their mesh and code
overlap is unresolved, so conservatively treat them as one Janna provenance
group and keep them in the same split. If qualified and admitted together in
development, they could supply additional source support in both missing large
size bands while retaining the existing torso and atmospheric groups in
ranking. That would address size-band absence; it would not establish coverage
of the same physics, discretizations, or joint numerical regimes.

Each still needs original archive/hash verification, independent provenance
review, admission and numerical qualification, and measured resource checks.
Further semiconductor leads in the 100,000-to-1,000,000 band have unresolved
provenance or usage questions and are not ready data. Broader application and
discretization coverage remains unresolved. Additional acquisition and an
explicit partition review are required before claiming representative
development coverage.
