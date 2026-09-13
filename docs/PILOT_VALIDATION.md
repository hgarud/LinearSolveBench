# Pilot validation

The coverage and replay evaluations use the fixed public candidate
[`gmres_amg.c`](../submissions/gmres_amg.c). Each split compiled the candidate
once, then evaluated each case once in a fresh native process. The Modal venue
enforced 2 CPUs, 4 GiB RAM, a 90-second case budget, and 5,000 iterations. Creation,
setup, and solve time counted; the accuracy gates and manufactured targets were
unchanged throughout validation.

| Track and split | Cases | Accurate solves |
| --- | ---: | ---: |
| NS cohort v2 coverage, development | 19 | 11 |
| NS cohort v2 coverage, ranked | 30 | 14 |
| FLASH replay, development | 96 | 96 |
| FLASH replay, ranked | 248 | 248 |

The [NS cohort review](NS_COHORT_V2.md) records the new 49-case inventory,
source evidence, selection policy, and remaining representation gaps. All
49 cases received one execution, with no crashes or infrastructure failures.
Janna/CoupCons3D in development and Goodwin/Goodwin_095 in ranking reached the
90-second limit and count as unsuccessful solves; neither was retried or
removed. The other 22 unsuccessful solver outcomes also remain in the coverage
count. The 1,602,111-row Transport case passed in 5.81 seconds of native elapsed
time under the same 4 GiB limit.
The full scalar reports preserve every outcome, resource limit, release
identity, and prepared-system identity:
[`development`](../data/releases/ns-cohort-v2-dev-validation.json) and
[`ranked`](../data/releases/ns-cohort-v2-ranked-validation.json).

All 49 fresh workloads independently passed the unchanged tenfold
qualification margins: 46 refined sparse-LU witnesses, one refined PyAMG/GMRES
witness, and two exact row-dominance certificates. The
[offline attempt history](../data/releases/ns-cohort-v2-offline-reference-attempts.json)
preserves earlier unsuccessful reference attempts and their separate resource
budgets. Those offline timings do not define a candidate performance baseline.

These are examples of coverage evaluation. This candidate is not a universal
NS reference, and the NS performance track remains unsupported. The older
`ns-mesh-pilot` release remains frozen with its historical 10/11 development
and 14/32 ranked results; those runs had no crashes, timeouts, infrastructure
failures, or retries. The replacement release uses fresh manufactured targets,
so results from the two releases should not be treated as a paired solver
comparison.

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
Preparing all 49 fresh NS workloads with the minimum supported NumPy 2.0.2 and
SciPy 1.14.1 reproduced every canonical matrix and complete numerical-system
identity obtained with the newer validation environment. The
[reproduction receipt](../data/releases/ns-cohort-v2-preparation-reproduction.json)
records all 49 matches. This repeats source decoding and input preparation;
ordinary public preparation reuses the frozen qualification evidence and does
not rerun expensive reference factorizations.

All 167 automated tests pass. They cover the selector against independent
exhaustive and distance calculations, release-to-selection bindings,
qualification, complete venue reports, and loading public resources from a
built wheel outside the checkout.
Both registered FLASH references score exactly 1.0 against themselves. The
original v1 split manifests and earlier frozen family releases retain their
identities. Ruff checks pass.

To repeat the public development evaluation from a checkout:

```bash
uv pip install -e '.[dev,qualification]'
python -m pytest
linear-solver-bench dataset prepare \
  --release ns-mesh-cohort-v2-dev --output data/prepared/ns-dev
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
