# Nonsymmetric mesh/PDE systems

This family uses original matrices from the SuiteSparse Matrix Collection. For
each matrix, the benchmark deterministically manufactures a hidden Rademacher
target and forms `b = A @ target`. Candidate code receives only `A`, `b`, the
zero initial guess, and the requested tolerance.

[`dev.json`](dev.json) is the complete public development set. It pins every
source archive and canonical matrix by SHA-256.
