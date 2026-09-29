# Discovered FLASH solvers

These four solvers were discovered by the named models using the **frontier0**
harness, with high effort, in ten iterations of four groups of sixteen samples
per model. Each file is the fully passing candidate with the greatest raw
runtime speedup in its 640-candidate run, as selected for the
[FLASH leaderboard](https://www.autodidakt.ai/linear-solve-bench).

| Model | Solver | Historical website speedup |
| --- | --- | ---: |
| Fable 5.1 | [fable_5_1.c](fable_5_1.c) | 1.8504555314557656× |
| Opus 5.5 | [opus_5_5.c](opus_5_5.c) | 1.71372430749397× |
| GPT-6 Sol | [gpt_6_sol.c](gpt_6_sol.c) | 1.524146994732567× |
| GPT-6 Astra | [gpt_6_astra.c](gpt_6_astra.c) | 1.380943946963153× |

The selection metric is `sum(reference_seconds) / sum(candidate_seconds)` over
all 96 cases, with successful reference and candidate verification on every
case. Starting seeds are excluded; ties use the earliest iteration/group/rollout.
These are development-set selection results. The historical discovery reward
used a different aggregation, and selects a different GPT-6 Sol candidate.
The files here correspond to the website's raw-runtime winners.

## Run

From an installed source checkout, after configuring Modal:

```bash
for model in fable_5_1 opus_5_5 gpt_6_sol gpt_6_astra; do
  linear-solver-bench run "discovered_solvers/magnetic_diffusion_flash/${model}.c" \
    --family magnetic_diffusion_flash --venue modal \
    --output "results/${model}.json"
done
```

Use `--venue local` for local validation. Omit `--case` to run all 96 cases;
partial runs have no aggregate score. Each solver is an ordinary candidate and
uses the same compiler audit, driver, and accuracy checks as other submissions.

The public reference is GMRES(50)+BoomerAMG, and the score is the ratio of total
runtimes. The public reference removes discovery's 1,000-iteration cutoff and
disables residual-history logging. See
[benchmark details](../../BENCHMARK.md).
Fresh timings depend on hardware, compiler, environment, and repeated execution;
the historical values above are not new public-run measurements. The
[published timing evidence](https://www.autodidakt.ai/benchmarks/flash-runtime-speedups-20260928.json)
contains the original per-case times.

## Source provenance

All four original sources passed 96/96 cases in discovery. Original sources
are identified below relative to the discovery repository; those paths are
provenance records, not dependencies of the public benchmark.

| Solver | Provider model ID | Original source under `frontier0_log/` |
| --- | --- | --- |
| `fable_5_1.c` | `claude-fable-5-1` | `flash-fable51-high-full-20260912/sources/iter_000008_g0000_r0010.c` |
| `opus_5_5.c` | `claude-opus-5-5` | `flash-opus55-high-full-20260926/sources/iter_000004_g0001_r0010.c` |
| `gpt_6_sol.c` | `gpt-6-sol` | `flash-gpt6-sol-high-full-20260927/sources/iter_000008_g0000_r0007.c` |
| `gpt_6_astra.c` | `gpt-6-astra` | `flash-gpt6Astra-high-full-20260912/sources/iter_000002_g0000_r0015.c` |

| Solver | Original C SHA-256 | Released C SHA-256 |
| --- | --- | --- |
| `fable_5_1.c` | `6baa0d92b73c1c79ca702b931355f5aa0ebe6867b180d99560a67e80f0446c2d` | `1b7252b2746d47c9b3b427f833d51fd2c3eb22db62300563f959c36820ad6dbe` |
| `opus_5_5.c` | `f67d38403827804d7a9da62bdbb7555ca8438485eeaa3f86f8f201d7d13809f0` | `80842e00494c8e782655273f3c8a41dfdb522707226e5c7c7fb6b600986054cc` |
| `gpt_6_sol.c` | `68e845e64a1141b1937e546a08d9ab0e1c089e8cebd55f512a073cf0a7d342c1` | `99b6899241dfe99ee3db75274bc3c3075340f7f1483593a774ec9aab0d4b8418` |
| `gpt_6_astra.c` | `2ceeb743f5c44781b2ab9ea07e493e46b4bcc85037efe08fd582e221b2e2ded5` | `e559f8b902a7ea9e64eb1e26018ba7e752c1713b2931c1d895f406efcf171288` |

The direct factory conversion renames `nsl_create` to `solver_create`, removes
its fourth parameter, and defines that parameter as the local constant
`maximum_iterations = 16000`, preserving the value supplied during discovery.
The numerical code is otherwise byte-for-byte unchanged. This is each imported
solver's internal budget; the benchmark imposes no shared candidate iteration
limit. The hashes above distinguish the original discovery sources from the
released files with the public factory interface.

Model names identify discovery provenance. The repository maintainers performed
the interface conversion. The collection uses the repository's
[Apache-2.0 license](../../LICENSE).

## Public validation

All four imported files compiled, passed the public symbol/export audits, and
passed **96/96 public cases each** through the ordinary local evaluator on macOS
(Apple Silicon). These local runs establish correctness;
a fresh Linux/Modal performance comparison has not been run for this import.
