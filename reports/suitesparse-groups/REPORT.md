# SuiteSparse Matrix Collection groups: meaning, contents, and within-group diversity

Analysis date: 2026-09-02  
Public data revision served by the collection: 31-Oct-2023 18:12:37  
Scope: 2,904 primary matrix problems in 203 groups

## Executive answer

A SuiteSparse **group is a provenance namespace, not a mathematical matrix
class**. The official client documentation defines the group as the source of a
problem—a person, organization, or another collection—and defines the full
identifier as `Group/Name`. The collection paper makes the same distinction:
group classifies source, while `kind` describes the application domain.
[Official `ssget` help](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/ssget/sshelp.html),
[Davis and Hu (2011)](https://doi.org/10.1145/2049662.2049663)

That distinction explains the data:

- Most groups are fairly coherent in application and shape. In the full
  snapshot, 172 of 203 groups have one normalized `kind`; 159 are square-only,
  34 rectangular-only, and only 10 mix square and rectangular matrices.
- Size is much less coherent. Fifty-two groups span more than 100× in maximum
  dimension, and those groups contain 2,137 matrices (73.6% of the collection).
  A family can therefore be mathematically very consistent but cover many mesh
  refinements or problem scales.
- Large imported archives and benchmark selections can be diverse on nearly
  every axis. `HB`, `Pajek`, `Bai`, `SNAP`, `MathWorks`, and `Williams` mix
  several kinds, structures, entry types, or definiteness classes.
- Group membership predicts **where a problem came from** more reliably than
  whether it is symmetric, positive definite, real, square, or of a particular
  size. Use `kind`, `RBtype`/structure, dimensions, symmetry statistics, and
  definiteness fields for property-based selection.

The exhaustive answer for every group is in
[GROUP_CATALOGUE.md](GROUP_CATALOGUE.md). It gives all 203 official group links,
source descriptions, kind counts, dimension ranges, square/rectangular counts,
special-structure counts, entry-type counts, known-positive-definite counts,
and a transparent diversity assessment.

## 1. What a “group” means

The collection uses three related but different concepts:

| Concept | Meaning | Example |
|---|---|---|
| `Group` | Source/provenance namespace: person, team, organization, imported collection, or benchmark | `Bai`, `Sandia`, `HB`, `DIMACS10` |
| `Name` | Local problem name inside that group; not guaranteed unique collection-wide | `copter2` |
| `kind` | Curated application/domain label for the primary matrix | `structural problem`, `directed graph`, `linear programming problem` |

Thus `GHS_indef/copter2` is unambiguous; `copter2` alone is not the canonical
identifier. Matrix IDs are also stable once assigned. The official help notes
that kind is often related to group because one submitter may work in one
domain, but explicitly warns that some groups cover many domains.
[Official `ssget` help](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/ssget/sshelp.html)

Group is not identical to `author`. Individual records separately store the
creator/author and editor/curator. A group may name a contributor, but it may
instead name an organization, the earlier archive from which the problem was
imported, or a later benchmark selection.

### Common naming patterns

| Pattern | What it usually represents | Examples |
|---|---|---|
| Contributor or research team | Problems submitted or assembled by a person/team | `Averous`, `Bai`, `Goodwin`, `Grund`, `Rommes` |
| Company, university, or lab | A source institution or toolchain | `Boeing`, `Sandia`, `IBM_EDA`, `Freescale`, `TAMU_SmartGridCenter` |
| Imported archive or benchmark | A pre-existing collection; often intentionally broad | `HB`, `GHS_indef`, `GHS_psdef`, `Oberwolfach`, `LPnetlib`, `Qaplib` |
| Graph/network corpus | Adjacency or weighted-adjacency data from a repository/challenge | `Pajek`, `SNAP`, `LAW`, `DIMACS10`, `Gset`, `GAP` |
| Application, solver, or construction | A simulation, equation family, or generator | `FIDAP`, `DRIVCAV`, `TOKAMAK`, `VLSI`, `FlowIPM22`, `Mycielski` |
| Mathematical subcollection | Related algebraic/combinatorial constructions | the `JGD_*` groups |

These patterns are descriptive, not mutually exclusive. For example,
`Williams` is named for a benchmark study and contains matrices originally
drawn from multiple scientific and graph domains.
[Williams group](https://sparse.tamu.edu/Williams)

## 2. What a collection “matrix” represents

The collection is formally a collection of sparse **problems**. Each problem
has a primary sparse matrix `A` plus metadata and may also include explicit
stored zeros, right-hand sides, solutions, coordinates, node labels, other
matrices, or a matrix sequence in auxiliary data. The published index statistics
and this report describe the primary `A`, not every auxiliary object.
[Davis and Hu paper](https://yifanhu.net/PUB/matrices.pdf)

The primary matrices commonly encode:

1. **Discretized operators or linear systems.** Finite-element,
   finite-volume, and finite-difference models produce stiffness, mass,
   Jacobian, saddle-point, or coupled-physics matrices. Structural mechanics,
   CFD, thermal, electromagnetics, acoustics, semiconductor, and materials
   problems are common.
2. **Optimization systems.** Linear-programming entries often store a
   rectangular constraint matrix. Nonlinear and optimal-control problems often
   store square KKT or Newton systems, frequently symmetric indefinite.
3. **Graphs and networks.** `A(i,j)` represents an edge or weight. Undirected
   graphs normally give symmetric adjacency matrices; directed graphs are
   generally unsymmetric; bipartite data may be rectangular.
4. **Model-reduction and eigenvalue problems.** The visible `A` can be one
   operator in a larger dynamical model whose mass, input, or output matrices
   appear in auxiliary fields.
5. **Algebraic and combinatorial constructions.** Boundary, differential,
   incidence, relation, design, and polynomial-system matrices are often
   integer-valued and rectangular.
6. **Least-squares, inverse, imaging, and data problems.** Examples include
   tomography, bundle adjustment, interpolation, recommendation, documents,
   and statistical designs.

The official overview lists both geometric PDE domains and non-geometric areas
such as optimization, economics, circuits, chemistry, power networks, and
graph data. [Collection overview](https://sparse.tamu.edu/about)

## 3. How to interpret the attributes

The collection’s [statistics documentation](https://sparse.tamu.edu/statistics)
defines the fields used below.

| Attribute | Meaning | Important caution |
|---|---|---|
| `nrows`, `ncols` | Shape of primary `A` | Use both; “size” is ambiguous for rectangular problems. This report uses `max(nrows,ncols)` for group dimension spans. |
| `nnz` | Numerically nonzero entries in `A` | Excludes explicitly stored zeros supplied by the author. `nentries = nnz + nzero`. |
| Entry type | Mutually exclusive RB category: real non-integer/non-binary, complex, integer, or binary | The separate `isReal` flag means non-complex, so integer and binary matrices also count as real under that flag. |
| Structure | Rectangular; square unsymmetric, symmetric, Hermitian, or skew-symmetric | This is the categorical algebraic structure, encoded by the second RB letter. |
| `pattern_symmetry` | Fraction of off-diagonal structural entries having a transposed partner | It is a ratio in `[0,1]`, not a Boolean. Rectangular matrices are assigned 0. |
| `numerical_symmetry` | Fraction of off-diagonal values having a conjugate-equal transposed partner | For complex matrices it measures Hermitian, not plain complex-symmetric, agreement. Rectangular matrices are assigned 0. |
| `posdef` | 1 known positive definite; 0 known not positive definite; historically −1 symmetric/Hermitian but unknown | No −1 cases occur in this snapshot. `cholcand` is only a Cholesky-candidate heuristic, not proof of positive definiteness. |
| `kind` | Curated problem-domain phrase | Semi-controlled vocabulary, not an exhaustive or immutable ontology. Modifiers such as `sequence`, `subsequent`, `duplicate`, and `random` carry useful provenance. |
| `isND`, `isGraph` | Curator judgments for 2D/3D discretization or graph interpretation | These are qualitative problem assessments, not intrinsic numeric properties. |
| `sprank`, `nblocks`, `ncc` | Structural rank, Dulmage–Mendelsohn blocks, and graph components | Graph problems may omit structural-rank data because their diagonal is often irrelevant. |

### Symmetric structure versus symmetry score

Two notions must not be conflated:

- The website/Rutherford–Boeing structure says whether the stored matrix is
  exactly symmetric, Hermitian, skew-symmetric, unsymmetric, or rectangular.
- The two symmetry scores quantify how much of a square matrix’s off-diagonal
  pattern or values match their transpose/conjugate transpose.

For example, a matrix can have a fully symmetric sparsity pattern but unequal
mirrored values. Conversely, complex symmetric and Hermitian are different
classes. This is why the snapshot has 1,201 matrices labelled symmetric but
1,196 with `numerical_symmetry == 1`; those columns answer related, not identical,
questions.

## 4. Snapshot and collection-wide profile

The official site displayed 2,904 problems when checked on 2026-09-02.
The current downloadable `ssstats.csv` and `ss_index.mat` still embed a last
revision of 31-Oct-2023; their hashes and HTTP metadata are recorded in
[data/provenance.json](data/provenance.json). The live total and the download
agree. [Collection index](https://sparse.tamu.edu/)

### Scale

| Measure | Minimum | Median | Maximum |
|---|---:|---:|---:|
| Rows | 1 | 4,942.5 | 226,196,185 |
| Columns | 2 | 6,139.5 | 226,196,185 |
| Numerically nonzero entries | 1 | 57,092 | 11,588,725,964 |
| Stored pattern entries | 1 | 57,235 | 11,588,725,964 |

There are 432 matrices with at least one explicit stored zero. Across those
records, the index reports 259,828,688 explicit-zero entries, which is why
`nentries` should not automatically be substituted for `nnz`.

The largest dimension belongs to `MAWI/mawi_201512020330`
(226,196,185 square; 480,047,894 nonzeros). The largest `nnz` belongs to
`Sybrandt/AGATHA_2015` (183,964,077 square; 11,588,725,964 nonzeros).

### Shape, structure, entry type, and flags

| Axis | Category | Count | Share |
|---|---|---:|---:|
| Shape | Square | 2,226 | 76.65% |
|  | Rectangular | 678 | 23.35% |
| Structure | Symmetric | 1,201 | 41.36% |
|  | Unsymmetric square | 1,020 | 35.12% |
|  | Rectangular | 678 | 23.35% |
|  | Hermitian | 3 | 0.10% |
|  | Skew-symmetric | 2 | 0.07% |
| Entry type | Real, excluding integer/binary | 1,732 | 59.64% |
|  | Binary | 601 | 20.70% |
|  | Integer | 522 | 17.98% |
|  | Complex | 49 | 1.69% |
| Curated flag | Graph | 543 | 18.70% |
|  | 2D/3D discretization | 899 | 30.96% |
| Definiteness | Known positive definite | 240 | 8.26% |

The continuous symmetry scores give another partition: 1,196 exact
numerically symmetric/Hermitian matrices; 222 with symmetric pattern but not
matching values; 768 partly pattern-symmetric; 40 square pattern-unsymmetric;
and 678 rectangular.

### Most common normalized kinds

This table uses the current website kind and collapses only `subsequent`,
`duplicate`, and `sequence` modifiers. It does not merge scientifically
different kinds into a subjective super-category.

| Normalized kind | Matrices | Share |
|---|---:|---:|
| Linear programming problem | 342 | 11.8% |
| Structural problem | 302 | 10.4% |
| Combinatorial problem | 299 | 10.3% |
| Circuit simulation problem | 258 | 8.9% |
| Computational fluid dynamics problem | 174 | 6.0% |
| Optimization problem | 138 | 4.8% |
| 2D/3D problem | 138 | 4.8% |
| Undirected graph | 138 | 4.8% |
| Undirected weighted graph | 122 | 4.2% |
| Optimal control problem | 91 | 3.1% |
| Directed graph | 83 | 2.9% |
| Economic problem | 71 | 2.4% |
| Power network problem | 70 | 2.4% |
| Chemical process simulation problem | 70 | 2.4% |
| Theoretical/quantum chemistry problem | 61 | 2.1% |

The normalized current website vocabulary has 58 values; the unnormalized CSV
has 91 exact strings.

## 5. Group sizes are strongly imbalanced

| Statistic | Result |
|---|---:|
| Groups | 203 |
| Matrices per group, minimum / median / mean / maximum | 1 / 4 / 14.31 / 292 |
| Singleton groups | 51 (25.1%) |
| Groups with fewer than five matrices | 114 (56.2%) |
| Matrices in the ten largest groups | 1,380 (47.5%) |

The largest groups are:

| Group | Matrices | Character |
|---|---:|---|
| [HB](https://sparse.tamu.edu/HB) | 292 | Historical Harwell–Boeing archive; many domains |
| [Sandia](https://sparse.tamu.edu/Sandia) | 192 | Circuit-simulation sequences |
| [Meszaros](https://sparse.tamu.edu/Meszaros) | 166 | Linear-programming problems |
| [DIMACS10](https://sparse.tamu.edu/DIMACS10) | 151 | Graph partitioning/clustering benchmark |
| [LPnetlib](https://sparse.tamu.edu/LPnetlib) | 138 | Netlib LP constraint matrices |
| [JGD_Homology](https://sparse.tamu.edu/JGD_Homology) | 128 | Homology/combinatorial matrices |
| [VDOL](https://sparse.tamu.edu/VDOL) | 91 | Optimal-control KKT systems |
| [Bai](https://sparse.tamu.edu/Bai) | 78 | Multi-domain contributor collection |
| [Pajek](https://sparse.tamu.edu/Pajek) | 76 | Heterogeneous graph/network datasets |
| [SNAP](https://sparse.tamu.edu/SNAP) | 68 | Stanford graph datasets |
| [Gset](https://sparse.tamu.edu/Gset) | 67 | Generated random graph families |
| [GHS_indef](https://sparse.tamu.edu/GHS_indef) | 60 | Indefinite test-matrix collection |
| [Schenk_IBMNA](https://sparse.tamu.edu/Schenk_IBMNA) | 52 | Nonlinear-optimization/KKT matrices |

Consequently, “sample groups uniformly” and “sample matrices uniformly” are
very different experimental designs.

## 6. Are matrices within a group similar or diverse?

### Direct group-level answer

Across all 203 groups, after collapsing only the official sequence/duplicate
modifiers:

| Axis | Group profile | Conclusion |
|---|---:|---|
| Application kind | 172 one-kind; 31 multi-kind | Usually coherent, but the multi-kind groups hold 1,083 matrices (37.3%). |
| Shape | 159 square-only; 34 rectangular-only; 10 mixed | Shape is the most reliable within-group categorical property. |
| Exact numerical/Hermitian symmetry among square members | 61 all exact; 76 none exact; 32 mixed; 34 rectangular-only | Usually consistent, but not guaranteed. Pattern symmetry must still be checked separately. |
| Complex versus non-complex | 184 non-complex-only; 9 complex-only; 10 mixed | Broad scalar domain is usually consistent. Integer/binary/non-integer real may still mix. |
| Maximum-dimension span | 60 one dimension; 31 ≤4×; 60 between 4× and 100×; 52 >100× | Size is the least safe attribute to infer from group. |

The ten shape-mixed groups are `HB`, `MathWorks`, `Bates`, `Pajek`,
`Barabasi`, `Meszaros`, `JGD_BIBD`, `JGD_Homology`, `JGD_Margulies`, and
`Hardesty`.

### More conservative analysis of groups with at least five matrices

Small groups can look homogeneous merely because they contain too few samples.
Among the 89 groups with at least five matrices, the number exactly uniform on
each attribute is:

| Attribute | Exactly uniform groups | Share of 89 |
|---|---:|---:|
| Shape | 82 | 92.1% |
| Website structure | 62 | 69.7% |
| Mutually exclusive entry type | 61 | 68.5% |
| Exact-symmetry class | 55 | 61.8% |
| Positive-definite status | 66 | 74.2% |
| Exact, unnormalized `kind` string | 51 | 57.3% |
| Normalized `kind` used in the catalogue | 64 | 71.9% |

Only 31 of these 89 groups are uniform across all six categorical attributes
used by the reproducible analysis (shape, structure, entry type, exact-symmetry
class, positive-definite status, and unnormalized kind). Even then, scale may
vary substantially.

The report’s optional diversity score is the mean fraction outside the modal
category across those six axes. It is a screening aid, not an official
SuiteSparse statistic. Groups with fewer than five members are not assigned a
strong homogeneous/heterogeneous label. The complete components are in
[output/group_summary.csv](output/group_summary.csv), so readers need not rely
on the score.

### Representative profiles

| Group | What it demonstrates | Measured profile |
|---|---|---|
| [HB](https://sparse.tamu.edu/HB) | Historical archive: very broad diversity | 292 matrices; 16 normalized kinds; dimensions 9–44,609; 276 square + 16 rectangular; 169 symmetric, 105 unsymmetric, 16 rectangular, 2 skew; all four entry types; 60 known PD. |
| [Bai](https://sparse.tamu.edu/Bai) | A contributor name need not mean one domain | 78 square matrices across eight normalized kinds; 66 unsymmetric, 11 symmetric, 1 Hermitian; real, complex, and integer entries; dimensions 62–23,560. |
| [Pajek](https://sparse.tamu.edu/Pajek) | One graph corpus, many graph forms | 76 matrices; 71 square + 5 rectangular; directed, undirected, weighted, multigraph, and bipartite kinds; binary, integer, and real entries; dimensions 10–3,774,768. |
| [Williams](https://sparse.tamu.edu/Williams) | Benchmark selection cuts across sources | Seven square matrices spanning spatial, economic, and graph problems; four symmetric + three unsymmetric; dimensions 36,417–1,000,005. |
| [Sandia](https://sparse.tamu.edu/Sandia) | Same domain/form, very different scale | 192 real square unsymmetric circuit matrices; rows 430–682,862; `nnz` 1,544–2,638,997. |
| [Gset](https://sparse.tamu.edu/Gset) | Related generated graph corpus, mixed value type | 67 square symmetric random graphs across even, 2-D torus, and skew pattern families; 34 binary + 33 integer/weighted; dimensions 800–10,000. |
| [LPnetlib](https://sparse.tamu.edu/LPnetlib) | Coherent algebraic form | 138 rectangular LP constraint matrices; 108 real + 30 integer; dimensions 13–243,246. |
| [VDOL](https://sparse.tamu.edu/VDOL) | Coherent numerical family | 91 real square symmetric, non-PD optimal-control systems; dimensions 99–18,476. |
| [QCD](https://sparse.tamu.edu/QCD) | Coherent complex family | 14 complex square unsymmetric quantum-chromodynamics matrices; dimensions 3,072–49,152. |
| [FlowIPM22](https://sparse.tamu.edu/FlowIPM22) | Structure/domain uniform; scale varies; metadata caveat | 11 real square symmetric graph Laplacians; dimensions 100,000–72,180,402. The index marks four PD and seven not, but the group notes say every matrix has zero row sums, implying singular positive-semidefinite Laplacians rather than PD matrices. |
| [FIDAP](https://sparse.tamu.edu/FIDAP) | Pattern symmetry differs from numerical symmetry | 35 real square CFD matrices; all have symmetric patterns, but only 13 have exact mirrored numerical values. |
| [JGD_Homology](https://sparse.tamu.edu/JGD_Homology) | Same combinatorial subject, shape/scale variation | 128 integer combinatorial matrices; 122 rectangular + 6 square; dimensions 15–564,480. |
| [Mycielski](https://sparse.tamu.edu/Mycielski) | Categorical uniformity does not imply size uniformity | 19 square symmetric binary graph matrices; dimensions 2–786,431 and an `nnz` span exceeding nine orders of magnitude. |

These examples support a continuum rather than a binary rule:

- **Tight families:** same generator, equation, structure, and entry type;
  often different resolution or parameter.
- **Domain-coherent but form-diverse collections:** all graph, circuit, CFD,
  or structural problems, yet different symmetry, weighting, or shape.
- **Provenance-only bundles:** common source/archive or benchmark study, but
  several scientific domains and matrix forms.

## 7. Practical consequences for using the collection

1. **Select by properties, not group name alone.** For a symmetric solver,
   filter categorical structure or exact symmetry. For SPD work, require
   `posdef == 1`. For least squares, require rectangular shape and inspect kind.
2. **Use group for provenance-aware sampling.** Stratifying or holding out
   entire groups can reduce leakage between refinement steps and related
   problem sequences, but group is not a perfect independence boundary.
   Related or duplicate problems can occur across groups, especially when one
   group is an imported collection.
3. **Balance large groups deliberately.** The median group has four matrices,
   while `HB` has 292 and the ten largest groups hold nearly half the collection.
   Report whether results are matrix-weighted or group-weighted.
4. **Separate graphs from linear systems when appropriate.** A graph adjacency
   matrix is not automatically a meaningful `Ax=b` test. Diagonal, rank, and
   positive-definiteness conventions differ.
5. **Do not bucket rectangular matrices as “nonsymmetric.”** Symmetry is not
   applicable; the collection assigns their symmetry ratios zero by definition.
6. **Preserve exact identifiers and metadata.** Use `Group/Name`, retain the
   matrix-specific citation/notes, and state transformations. The collection’s
   matrices are CC BY 4.0. [License and citation guidance](https://sparse.tamu.edu/about)

For the LinearSolverBench repository specifically, a robust sampling scheme
would treat group as a provenance stratum, then balance independently across
kind, shape/structure, entry type, size bands, rank, and definiteness. It should
also audit `duplicate`, `subsequent`, and `sequence` metadata before splitting
training/development/test sets.

## 8. Method and reproducibility

The analysis used:

- The official [`ssstats.csv`](https://sparse.tamu.edu/files/ssstats.csv) and
  [`ss_index.mat`](https://sparse.tamu.edu/files/ss_index.mat) endpoints,
  retrieved 2026-09-02.
- The official website’s public metadata repository at commit
  [`0316f79`](https://github.com/ScottKolo/suitesparse-matrix-collection-website/tree/0316f796684820b319a8601110e1ce51617c9ee0/db/collection_data),
  used for categorical structure, entry type, current kind, and group notes.
- Metadata only; no full matrix payloads were downloaded.

Numeric group summaries deliberately use the downloadable CSV kind so the
snapshot remains reproducible. The current website disagrees only for
`Chevron/Chevron1`–`Chevron/Chevron4`: the CSV says `other problem`, while the
current site says `2D/3D problem`. The readable catalogue uses the current site
label, and [output/kind_source_mismatches.csv](output/kind_source_mismatches.csv)
records the four differences.

There is also a documentation-version nuance around explicit zeros: the
statistics page defines pattern symmetry from the numerically nonzero `A`,
while the newer client help describes a pattern including `A + Zeros`. This
report consumes the published symmetry scores rather than recomputing them, so
the ambiguity does not change the reported results; it matters if independently
reproducing those scores from matrix files.

Run the analysis after checking out the website metadata repository:

```bash
git clone https://github.com/ScottKolo/suitesparse-matrix-collection-website.git \
  /path/to/suitesparse-matrix-collection-website
git -C /path/to/suitesparse-matrix-collection-website \
  checkout 0316f796684820b319a8601110e1ce51617c9ee0

python3 reports/suitesparse-groups/analyze_groups.py \
  --stats reports/suitesparse-groups/data/ssstats.csv \
  --website-repo /path/to/suitesparse-matrix-collection-website \
  --output-dir reports/suitesparse-groups/output

python3 reports/suitesparse-groups/build_catalogue.py \
  --group-summary reports/suitesparse-groups/output/group_summary.csv \
  --matrix-metadata reports/suitesparse-groups/output/matrix_metadata.csv \
  --website-repo /path/to/suitesparse-matrix-collection-website \
  --output-md reports/suitesparse-groups/GROUP_CATALOGUE.md \
  --output-csv reports/suitesparse-groups/output/group_catalogue.csv
```

### Delivered files

| File | Purpose |
|---|---|
| [GROUP_CATALOGUE.md](GROUP_CATALOGUE.md) | Human-readable all-203-group appendix |
| [output/group_catalogue.csv](output/group_catalogue.csv) | Compact, filterable all-group catalogue |
| [output/group_summary.csv](output/group_summary.csv) | Full group-level counts, ranges, medians, symmetry scores, and outliers |
| [output/matrix_metadata.csv](output/matrix_metadata.csv) | Joined 2,904-row matrix metadata table |
| [output/group_diversity_ranking.csv](output/group_diversity_ranking.csv) | Groups sorted by the unofficial categorical-diversity score |
| [output/collection_attribute_distributions.csv](output/collection_attribute_distributions.csv) | Overall kind/shape/structure/type distributions |
| [output/group_size_distribution.csv](output/group_size_distribution.csv) | Exact group-size frequency table |
| [output/largest_matrices.csv](output/largest_matrices.csv) | Largest rows/columns/`nnz`/stored-entry cases |
| [output/collection_summary.json](output/collection_summary.json) | Machine-readable headline statistics and definitions |
| [data/provenance.json](data/provenance.json) | Retrieval date, URLs, hashes, ETags, revision, and metadata commit |
| [analyze_groups.py](analyze_groups.py) | Standard-library analysis generator |
| [build_catalogue.py](build_catalogue.py) | All-group catalogue generator |

## 9. Sources

- Timothy A. Davis and Yifan Hu, *The University of Florida Sparse Matrix
  Collection*, ACM TOMS 38(1), 2011.
  [DOI](https://doi.org/10.1145/2049662.2049663) ·
  [author PDF](https://yifanhu.net/PUB/matrices.pdf)
- Scott P. Kolodziej et al., *The SuiteSparse Matrix Collection Website
  Interface*, JOSS 4(35), 2019.
  [Paper](https://joss.theoj.org/papers/10.21105/joss.01244)
- [Official collection overview](https://sparse.tamu.edu/about)
- [Official statistics definitions](https://sparse.tamu.edu/statistics)
- [Official `ssget`/`ssgui` help](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/ssget/sshelp.html)
- [Official current group pages](https://sparse.tamu.edu/groups?per_page=All)
