# Contributing

Keep the participant workflow small and readable. New abstractions need more than
one real use before they are introduced.

```bash
python -m pip install -e '.[dev]'
ruff format --check src tests
ruff check src tests
pytest
```

Changes to numerical inputs, accuracy, the native ABI, the fixed reference,
timing, or Modal resources change benchmark results and require an explicit
version decision. Keep private test data, credentials, full private reports, and
dataset-construction artifacts outside this repository.
