# Pilot validation

The pilot was exercised on the complete declared NS coverage corpus and the
FLASH replay captures using the fixed public candidate
[`gmres_amg.c`](../submissions/gmres_amg.c). Each split compiled the candidate
once, then evaluated each case once in a fresh native process. The Modal venue
enforced 2 CPUs, 4 GiB RAM, a 90-second case budget, and 5,000 iterations. Creation,
setup, and solve time counted; the accuracy gates and manufactured targets were
unchanged throughout validation.

| Track and split | Cases | Accurate solves |
| --- | ---: | ---: |
| NS coverage, development | 11 | 10 |
| NS coverage, ranked | 32 | 14 |
| FLASH replay, development | 96 | 96 |
| FLASH replay, ranked | 248 | 248 |

All 43 NS executions completed, with no crashes, timeouts, infrastructure
failures, or retries. The 19 unsuccessful solves returned nonzero solver
statuses and count as coverage failures. Both atmospheric operators with more
than one million rows passed under the same 4 GiB limit: 1,270,432 rows in
3.47 seconds and 1,489,752 rows in 4.58 seconds of native elapsed time.
These results demonstrate a working coverage evaluation; this candidate is not
a universal NS reference. The NS performance track remains unsupported.

The FLASH reference passed all 344 cases: 96 development cases and 248 ranked
cases, including all 88 correctness controls. Each case ran once, with one
compilation per split.

The fixed identities used for these runs are:

| Item | SHA-256 |
| --- | --- |
| Candidate source | `89f8ff13a1c21448692c1df9cfc363e180dc0838f031d531ca51573b43ea16b0` |
| Modal CPU runtime manifest | `4b09e68ce2fc17261c1d776bb01d63ec40f8f92388b9aff95c046f7c1f8bb076` |

The frozen runtime manifest is
[`data/releases/cpu-runtime-v2.json`](../data/releases/cpu-runtime-v2.json).
The FLASH reference artifacts are
[`flash-replay-dev-reference.json`](../data/releases/flash-replay-dev-reference.json)
and
[`flash-replay-ranked-reference.json`](../data/releases/flash-replay-ranked-reference.json).
Each reference artifact embeds its complete verified report, including every
scored case and correctness control, source and executable identities, exact
prepared inputs, resource limits, and per-case timings. A replay candidate must
match the registered reference's release, runtime, and venue and pass all cases
before receiving an aggregate speedup.

The data release pins Hugging Face commit
[`3da5eda0ce3e85da1808d4b62d28d9111127e354`](https://huggingface.co/datasets/hgarud/LinearSolveBench/tree/3da5eda0ce3e85da1808d4b62d28d9111127e354).
All 344 published archive digests were checked against the release inventory;
anonymous downloads of sample cases also passed checksum and numerical-input
validation. NS preparation uses the original public SuiteSparse archives. A
freshly installed wheel successfully prepared NS and FLASH data without
depending on a source checkout.

The reference source, runtime manifest, and frozen calibrations are published
at Hugging Face commit
[`cbb06d2735b38d4078182d9db92fbc63790fe87b`](https://huggingface.co/datasets/hgarud/LinearSolveBench/tree/cbb06d2735b38d4078182d9db92fbc63790fe87b/references).
Anonymous downloads of all eight files added or updated in that publication
matched their local release bytes exactly.

Two independent clean Linux runtime builds produced identical complete
manifests, including the HYPRE library, trusted driver, and header tree. Two
clean macOS builds also matched each other. This is reproducibility within each
platform and toolchain; it does not require Linux and macOS binaries to match.
Preparing all 43 NS cases with the minimum supported NumPy 2.0.2 and SciPy
1.14.1 reproduced the exact prepared manifest identities obtained with the
newer validation environment.

All 122 automated tests pass. They include loading and scoring both published
references from the built wheel outside the checkout, with each registered
reference scoring exactly 1.0 against itself. Ruff checks pass, and regenerating
the original v1 split manifests leaves their bytes unchanged.

To repeat the public development evaluation from a checkout:

```bash
python -m pytest
linear-solver-bench dataset prepare \
  --release ns-mesh-pilot-dev --output data/prepared/ns-dev
modal run modal_app.py --source submissions/gmres_amg.c \
  --cases data/prepared/ns-dev --output results/ns-dev.json \
  --score-output results/ns-dev-score.json
linear-solver-bench dataset prepare \
  --release flash-replay-dev-pilot --output data/prepared/flash-dev
modal run modal_app.py --source submissions/gmres_amg.c \
  --cases data/prepared/flash-dev --output results/flash-dev.json \
  --calibration data/releases/flash-replay-dev-reference.json \
  --score-output results/flash-dev-score.json
```

These are single measurements, not estimates of timing variance or stable
hardware-level precision. Rebuilt runtimes must match the frozen identity before
using its timing reference. Hashes bind artifact contents and detect changes;
they do not authenticate who performed an evaluation. Official results depend
on the trusted operator running the declared venue and publishing its reports.
