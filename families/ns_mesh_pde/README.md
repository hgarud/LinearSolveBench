# Nonsymmetric mesh/PDE systems

This family uses original matrices from the SuiteSparse Matrix Collection. For
each matrix, the benchmark manufactures a hidden deterministic Rademacher
target and forms `b = A @ target`. Each candidate solver receives only `A`, `b`, the
zero initial guess, and the requested tolerance.

The 19 matrices are real, square, and nonsymmetric. The evaluator
stores them as `float64` CSR matrices. Their dimensions range from 317 × 317 to
1,602,111 × 1,602,111, with 6,858 to 23,487,281 nonzeros.

[`dev.json`](dev.json) is the complete public development set.
Every case in the set is identified by its unique SHA-256.
A candidate's score is the number of development cases it solves.
