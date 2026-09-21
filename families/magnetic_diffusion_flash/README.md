# magnetic_diffusion_flash

This family contains the linear systems captured from FLASH magnetic-diffusion solves.

The 96 cases use real, square `float64` CSR matrices at eight
distinct dimensions. They range from 9,984 × 9,984 to 113,664 × 113,664,
with 108,364 to 1,245,184 nonzeros.

[`dev.json`](dev.json) is the complete public development set.
Every case in the set is identified by its unique SHA-256.
A candidate's score is the geometric-mean of speedups over all the cases, provided it solves them completely.
