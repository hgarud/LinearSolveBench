# Evaluator operations

Official evaluation separates trusted data preparation and verification from
candidate execution. Prepared archives contain exact solutions and must stay on
the operator machine. Only the public binary case payload is copied into the
Modal Sandbox.

Runtime manifests bind the required `solver_create` candidate entry point.
After an entry-point change, rebuild the runtime into a new output directory
and recompile submissions. Reference timing artifacts must match the rebuilt
runtime identity before they can be used for scoring.

## Prepare ranked systems

Create one persistent 32-byte binary key, store it in secret storage, and back
it up. Changing this key changes every ranked right-hand side and therefore
requires a new prepared artifact and calibration.

```bash
mkdir -p secrets
openssl rand 32 > secrets/ranked-rhs-v1.bin
linear-solver-bench dataset prepare \
  --split ranked \
  --rhs-key-file secrets/ranked-rhs-v1.bin \
  --output data/prepared/ranked-v1
```

The `secrets/` and `data/prepared/` directories are ignored by Git. Confirm the
prepared manifest has 203 cases and preserve its manifest digest with the
operator records.

## Bootstrap calibration

Use a trusted reference submission that solves every ranked case. The first
Modal run has no calibration artifact, so provide a bootstrap deadline whose
aggregate fits within Modal's 24-hour Sandbox lifetime. A 60-second per-case
deadline fits v1.

```bash
modal run modal_app.py \
  --source operator/reference.c \
  --cases data/prepared/ranked-v1 \
  --deadline 60 \
  --output results/reference-modal-ranked.json

linear-solver-bench calibrate results/reference-modal-ranked.json \
  --output operator/calibration-cpu-v1.json
```

Review the report: all 203 cases must pass, `venue_id` must be
`modal-sandbox-cpu-v1`, and the runtime, dataset, and prepared-manifest
identities must match the intended release. Freeze the resulting calibration
artifact; never regenerate it within the same benchmark version.

## Evaluate and score

```bash
modal run modal_app.py \
  --source submissions/candidate.c \
  --cases data/prepared/ranked-v1 \
  --calibration operator/calibration-cpu-v1.json \
  --output results/candidate-modal-ranked.json

linear-solver-bench score \
  results/candidate-modal-ranked.json \
  operator/calibration-cpu-v1.json
```

The evaluator rejects calibration artifacts from a different venue, HYPRE
runtime, dataset manifest, prepared RHS corpus, or case order. Keep the original
submitted source and verify its SHA-256 against the report before publishing a
result.

## Release checklist

- Pin and publish the repository commit, `benchmark.toml`, ranked manifest,
  HYPRE commit/tree, Modal base-image digest, and calibration artifact.
- Preserve the ranked RHS key privately and ensure it never enters an image,
  volume, log, result, environment variable, or candidate path.
- Publish per-case outcomes and timings, not prepared archives or exact
  solutions.
- Run the CI suite and native build/audit smoke check on the release commit.
- Start a new benchmark version for any scientific-contract or venue change.
