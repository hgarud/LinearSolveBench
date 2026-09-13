# Evaluator operations

Preparation and verification run on the trusted operator side. Manufactured
targets, optional numerical reference vectors, and ranked RHS keys stay there.
The candidate receives only the current numerical input in a network-disabled
Modal sandbox. Pilot cases are loaded and streamed one at a time, compiled
candidate code is reused, and each case runs once in a fresh native process.

## Prepare and run NS coverage

The initial development release has two cases. From a checkout:

```bash
linear-solver-bench dataset prepare \
  --release ns-mesh-dev-pilot --output data/prepared/ns-dev
modal run modal_app.py \
  --source submissions/candidate.c --cases data/prepared/ns-dev \
  --output results/ns-coverage.json
linear-solver-bench score results/ns-coverage.json --output results/ns-score.json
```

Coverage requires no timing reference or calibration. Partial solver coverage
is a valid result: every expected case remains in the report and solved counts
determine ranking. All expected cases must be evaluated, even when some fail.

For a future published ranked NS release, supply its committed operator key
using `dataset prepare --rhs-key-file`. The key must match the release's key
commitment. Changing it changes numerical inputs and requires a new release
and qualification. Store it outside candidate-visible paths and images.

## Qualify and score FLASH replay

The public FLASH manifests are `flash-replay-dev-pilot` and
`flash-replay-ranked-pilot`. The fixed reference implementation is
`submissions/gmres_amg.c`. Dataset publication and qualification of a timing
reference on the chosen execution environment are separate steps.

```bash
linear-solver-bench dataset prepare \
  --release flash-replay-ranked-pilot --output data/prepared/flash-ranked
modal run modal_app.py \
  --source submissions/gmres_amg.c --cases data/prepared/flash-ranked \
  --output results/flash-reference.json
linear-solver-bench calibrate results/flash-reference.json \
  --output operator/flash-calibration.json
modal run modal_app.py \
  --source submissions/candidate.c --cases data/prepared/flash-ranked \
  --calibration operator/flash-calibration.json \
  --output results/flash-candidate.json
linear-solver-bench score results/flash-candidate.json \
  operator/flash-calibration.json --output results/flash-score.json
```

`calibrate` accepts only a complete replay report with every scored case and
correctness control passing. It freezes the one measured execution per case,
including the reference source identity and complete report. It does not adjust
pilot deadlines or run the reference again. Publish the reference source and
configuration with its report and calibration artifact.

Register the exact published calibration digest for its release, runtime, and
venue before accepting official replay scores. An unregistered calibration can
produce a development score, but reports `reference_status: unregistered` and
`official: false` even when its solver passes every case.

Candidate runs reuse those timings. Scoring rejects a different numerical
release, case order, prepared corpus, accuracy/execution/scoring contract,
runtime, venue, roles, or weights. A failed candidate case or control leaves
aggregate speedup null. Without a reference, a replay report remains useful for
accuracy diagnostics but scores as `awaiting_reference`.

Local runs can provide development references, but their `local-uncontrolled`
timings cannot be mixed with Modal results or treated as official venue timings.

## Frozen settings and official evaluation

Pilot execution takes deadlines, maximum iteration requests, CPU cores, and
memory from the release manifest. Modal applies matching CPU and memory
requests and hard limits. Networking is disabled. Local evaluation reports the
same requested settings but does not claim those resource limits are enforced.
The candidate factory receives the frozen iteration request and tolerance.

Add `--official` to `modal run modal_app.py` when evaluating a trusted published
release. This checks its exact digest against the packaged release registry
and requires the complete split. A draft manifest may be evaluated without
that flag, but hashing a user-authored manifest does not make it an official
release. The resulting score records whether release trust and venue conditions
qualify it as official. The development pilot is not a claim that a full ranked
NS corpus or a public replay timing reference has been qualified.

Candidate failures are recorded without retrying or skipping other cases.
Malformed inputs, missing expected cases, and infrastructure failures invalidate
the evaluation; they do not reduce its denominator. Preserve the full report
and original source bytes, whose SHA-256 is bound into the report.

## Runtime and installed packages

```bash
linear-solver-bench runtime build --output build/runtime
linear-solver-bench candidate validate submissions/candidate.c --runtime build/runtime
```

The runtime pins HYPRE, compiler/build information, native driver, and the
`solver_create` entry point. After changing the candidate interface or runtime,
build a new runtime, recompile submissions, and requalify references before
using new timing artifacts. Calibration from the previous runtime is rejected.

The Python CLI includes packaged native sources, public headers, benchmark
configuration, and dataset manifests. Installed release IDs work outside a
checkout. A writable cache is separate from installed resources; use
`LINEAR_SOLVER_BENCH_CACHE` or `dataset prepare --cache` when needed. Modal
examples use the checkout's `modal_app.py` and the optional Modal dependency.

## Existing v1 operations

V1 remains separate and retains 203 ranked cases, three repetitions, its own
condition-dependent tolerances, and calibration-based timing tie-breaks.
Commands without a pilot `--release` retain the v1 preparation route:

```bash
mkdir -p secrets
openssl rand 32 > secrets/ranked-rhs-v1.bin
linear-solver-bench dataset prepare --split ranked \
  --rhs-key-file secrets/ranked-rhs-v1.bin --output data/prepared/ranked-v1
modal run modal_app.py \
  --source operator/reference.c --cases data/prepared/ranked-v1 \
  --deadline 60 --output results/reference-modal-ranked-v1.json
linear-solver-bench calibrate results/reference-modal-ranked-v1.json \
  --output operator/calibration-cpu-v1.json
modal run modal_app.py \
  --source submissions/candidate.c --cases data/prepared/ranked-v1 \
  --calibration operator/calibration-cpu-v1.json \
  --output results/candidate-modal-ranked-v1.json
linear-solver-bench score results/candidate-modal-ranked-v1.json \
  operator/calibration-cpu-v1.json
```

Keep the persistent ranked key and all prepared reference vectors on the trusted
operator side. Freeze the benchmark commit, runtime, dataset identities, venue,
reference source, and calibration before accepting official results. Changes to
scientific requirements or venue require a new version.
