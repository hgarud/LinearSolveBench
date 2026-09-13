# Contributing

Changes should preserve the benchmark's scientific identity and keep the
trusted path small. Bug fixes, portability improvements, new public tests, and
proposals for future tracks are welcome.

## Development checks

```bash
uv venv --python 3.12
uv pip install -e '.[dev]'
ruff format --check .
ruff check .
pytest
```

To exercise the native boundary, build the pinned runtime and audit the starter
submission:

```bash
linear-solver-bench runtime build --output build/runtime
linear-solver-bench candidate validate submissions/starter.c --runtime build/runtime
```

Run `python tools/freeze_manifests.py` after intentionally changing selection
logic. Both generated manifest digests and the reason for a dataset-version
change must be reviewed. Do not rewrite a published manifest in place; create a
new benchmark version.

## Contract changes

A pull request that changes the dataset, accuracy gates, timed boundary,
submission ABI, resource limits, HYPRE identity, or ranking rule must call that
out explicitly. Such changes normally belong in a new track or major benchmark
version so existing results remain comparable.
