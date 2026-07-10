# cudf-bench

Benchmark NVIDIA's [cuDF](https://github.com/rapidsai/cudf) GPU dataframe library against pandas and Polars, find a case where it is slower than it should be, diagnose why with Nsight Systems, and fix it (upstream PR or standalone proof).

Every number in this repo is reproducible: the harness generates its own data deterministically, logs every run with backend versions and device info, and the Colab notebook reruns the whole thing on a free T4 GPU.

## Why trust these benchmarks

GPU timing has three classic traps, and the harness handles all of them (`benchmark/timer.py`):

1. **Warm-up** — the first GPU run pays one-time setup (kernel compilation, memory pool growth). Each op runs once untimed first.
2. **Async execution** — cuDF can return before the GPU finishes. The clock only stops after a device synchronize *and* touching the result.
3. **Noise** — every op runs 5+ times; the median is reported and all rep times are logged.

Also: all backends benchmark byte-identical input (data is generated once in pandas, then converted — conversion cost, including host→GPU copy, is excluded from op timing), and a test suite proves pandas and Polars agree on every op's *result* before any speed is measured.

## Quickstart (local, CPU backends)

```bash
pip install -e ".[dev]"
pytest                                            # correctness first
python -m benchmark.runner --backend pandas --smoke
python -m benchmark.runner --backend polars --smoke
```

## Quickstart (Colab, GPU)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alexislowys/cudf-bench/blob/main/notebooks/colab_run.ipynb)

Open [`notebooks/colab_run.ipynb`](notebooks/colab_run.ipynb) in Google Colab (badge above), set **Runtime → Change runtime type → T4 GPU**, and Run All. It verifies the GPU, runs the same benchmarks under pandas and cuDF, and plots the speedups.

## Running a real benchmark

```bash
python -m benchmark.runner --backend cudf \
    --ops groupby_agg,inner_join,sort,str_contains \
    --rows 1e6,1e7 --skew 0,1.1 \
    --out results/results.csv
```

Knobs: `--rows` (table size), `--skew` (key distribution — 0 is uniform, 1.1+ means a few keys dominate, which stresses hash joins/groupbys), `--n-keys`, `--str-len`, `--reps`, `--warmup`, `--seed`. Results append to a CSV; `benchmark/report.py` computes speedup tables and charts from it.

## Repo layout

```
benchmark/
  backends.py   # pandas / polars / cudf adapters (convert, sync, device info)
  datagen.py    # deterministic synthetic tables: size, skew, strings
  ops.py        # groupby_agg, inner_join, sort, filter, str_contains, str_replace, rolling_mean
  timer.py      # the fair stopwatch
  runner.py     # CLI: grid runner, appends to results CSV
  report.py     # speedup tables and plots
notebooks/
  colab_run.ipynb  # end-to-end GPU run on Colab
results/           # committed benchmark logs (CSV)
tests/             # datagen properties, pandas↔polars result equivalence, timer behavior
```

## Project roadmap

See [ROADMAP.md](ROADMAP.md) — six phases from "get a GPU" to "ship a fix upstream".
