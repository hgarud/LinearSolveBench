# Frozen validation and reference artifacts

## NS cohort v2 validation

`ns-cohort-v2-dev-validation.json` and
`ns-cohort-v2-ranked-validation.json` contain complete coverage evaluation
reports for the fixed public GMRES+AMG example. They bind the selected releases,
fresh prepared inputs, runtime, resource limits, and every individual outcome.
They are coverage examples; NS scoring does not use their elapsed times as a
speedup reference. Candidate failures and timeouts remain in each report.

`ns-cohort-v2-preparation-reproduction.json` records exact matrix and complete
system identity agreement for all 49 fresh workloads under the minimum
supported NumPy 2.0.2 and SciPy 1.14.1 versions. Only source acquisition and
manufactured-input preparation were repeated for that check.

`ns-cohort-v2-offline-reference-attempts.json` records the large-case offline
reference attempts, including failed or cancelled attempts on unchanged inputs.
These use separate qualification budgets and are not candidate timing results.

See the [cohort review](../../docs/NS_COHORT_V2.md) for selection, qualification
and remaining representation gaps, and
[pilot validation](../../docs/PILOT_VALIDATION.md) for the measured results.

## FLASH replay reference

`flash-replay-dev-reference.json` and `flash-replay-ranked-reference.json`
contain the complete qualification reports and frozen timings for the pilot.
The fixed public solver, `submissions/gmres_amg.c`, passed all 96 development
cases and all 248 ranked cases, including 88 required correctness controls.
Each split compiled the same source once and executed each case exactly once.

Use the artifact matching the candidate's split when running or scoring FLASH
replay. These timings apply to `modal-sandbox-pilot-cpu-v2`, with two CPUs,
4 GiB RAM, a 90-second case deadline, and at most 5,000 iterations. Creation,
setup, and solve time count toward the score. Transfer and archive loading do
not. Local timings cannot be compared against this reference.

`cpu-runtime-v2.json` records the exact Linux compiler, HYPRE commit, native
artifacts, and runtime identity. Two independent clean Linux builds reproduced
this manifest. A changed runtime needs a newly qualified reference.

| Identity | SHA-256 |
| --- | --- |
| Reference C source | `89f8ff13a1c21448692c1df9cfc363e180dc0838f031d531ca51573b43ea16b0` |
| Runtime manifest | `4b09e68ce2fc17261c1d776bb01d63ec40f8f92388b9aff95c046f7c1f8bb076` |
| Compiled reference executable | `4d6d411793e5d33d4fd24aa08301867a872473f8056a328ae7394a265c3a237f` |

Each reference file carries its own canonical `calibration_sha256`; the scorer
registers that exact value for its release, runtime, and venue. The embedded
report also binds every prepared numerical input, accuracy gate, resource
limit, case role, weight, and timing. Hashes establish artifact identity;
official evaluation still requires trusted operator execution.

The dataset and reference files are also available from the public
[Hugging Face dataset](https://huggingface.co/datasets/hgarud/LinearSolveBench).
Reference artifacts are pinned at commit
[`cbb06d2735b38d4078182d9db92fbc63790fe87b`](https://huggingface.co/datasets/hgarud/LinearSolveBench/tree/cbb06d2735b38d4078182d9db92fbc63790fe87b/references).
Numerical release manifests retain their original data commit.
See [operations](../../docs/OPERATIONS.md) for commands and
[pilot validation](../../docs/PILOT_VALIDATION.md) for the release evidence.
