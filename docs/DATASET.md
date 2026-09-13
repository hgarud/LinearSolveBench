# Dataset and provenance

## Source snapshot

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

## Numerical qualification

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

## Matrix transformation

Preparation downloads `https://sparse.tamu.edu/MM/{Group}/{Name}.tar.gz`,
verifies the frozen archive digest, safely extracts the Matrix Market member,
expands declared symmetry, converts to float64 CSR, sums duplicate entries,
removes exact zeros, and sorts column indices. It performs no reordering,
scaling, or transposition.

Every prepared case is content-addressed. Ranked right-hand sides are derived
only during trusted preparation from an operator-held key. The key and exact
solution are never copied into the candidate sandbox workspace.

## License and attribution

SuiteSparse matrices are licensed CC BY 4.0. Redistributions must preserve the
original Matrix Market metadata and matrix-specific citation instructions,
identify the canonicalization and manufactured-right-hand-side modifications,
and cite:

- Timothy A. Davis and Yifan Hu, *The University of Florida Sparse Matrix
  Collection*, ACM TOMS 38(1), 2011, DOI 10.1145/2049662.2049663.
- Scott P. Kolodziej et al., *The SuiteSparse Matrix Collection Website
  Interface*, JOSS 4(35), 2019, DOI 10.21105/joss.01244.
