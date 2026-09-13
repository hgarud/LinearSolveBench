# Dataset notices

## SuiteSparse

The catalogue and qualification evidence identify matrices from the
[SuiteSparse Matrix Collection](https://sparse.tamu.edu/). The matrices are
licensed under CC BY 4.0. This repository does not commit matrix payloads.

Prepared systems are modified derivatives: storage is canonicalized to
float64 CSR and the benchmark constructs a right-hand side from an
evaluator-owned manufactured solution. Preserve original matrix metadata and
matrix-specific attribution whenever prepared payloads are distributed.

## FLASH replay

The standalone numerical captures are published in the public
[LinearSolveBench dataset](https://huggingface.co/datasets/hgarud/LinearSolveBench)
under Apache-2.0. The release manifests identify the exact dataset commit and
archive hashes. FLASH software is not redistributed with these captures, and
the dataset license does not grant a license to FLASH itself. Public capture
context, split assignments, and numerical validation are recorded in the dataset
catalogue and card.
