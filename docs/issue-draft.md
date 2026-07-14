# DRAFT — GitHub issue for rapidsai/cudf (not yet filed)

**Title:** [BUG] Hash groupby ~1.5x slower on skewed keys: per-block cardinality gate disables shared-memory path exactly when hot-key atomic contention is worst

---

## Describe the bug

`groupby.agg` on a table with **Zipf-skewed keys** is ~1.48x slower than on
**uniform keys** of the same nominal cardinality — measured at locked GPU clocks
with a preallocated RMM pool, so the gap is pure kernel time. CPU dataframe
libraries (pandas, Polars) get ~2x *faster* on the same skewed input, so the
gap versus CPU baselines narrows sharply exactly on the data shape (heavy-hitter
keys) that's common in real workloads.

| condition (10M rows, 100k nominal keys, Tesla T4 @ locked 1590 MHz, pool preallocated) | `groupby('key0').agg({'val0':'mean','val1':'sum'})` |
|---|---|
| uniform keys | 16.0 ms |
| Zipf(1.5) keys | 23.7 ms (**1.48x**) |

Medians over 11 back-to-back in-process calls, CUDA-event timing == wall time.
`sort_values` on identical data shows no skew sensitivity, and 10-uniform-keys
data (extreme concentration, tiny cardinality) is the *fastest* configuration
we measured.

## Steps/Code to reproduce bug

```python
import cudf, cupy, numpy as np, time

rows, n_keys = 10_000_000, 100_000
rng = np.random.default_rng(0)

def table(skewed):
    if skewed:
        ranks = np.arange(1, n_keys + 1, dtype=np.float64)
        p = ranks ** -1.5
        keys = rng.choice(n_keys, size=rows, p=p / p.sum())
    else:
        keys = rng.integers(0, n_keys, size=rows)
    return cudf.DataFrame({"key0": keys.astype("int64"),
                           "val0": rng.random(rows), "val1": rng.random(rows)})

for label, skewed in [("uniform", False), ("zipf1.5", True)]:
    df = table(skewed)
    times = []
    for _ in range(12):
        cupy.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        r = df.groupby("key0").agg({"val0": "mean", "val1": "sum"})
        len(r)
        cupy.cuda.runtime.deviceSynchronize()
        times.append(time.perf_counter() - t0)
    print(label, [f"{t*1000:.1f}" for t in times])
```

Best measured with locked clocks (`nvidia-smi -lgc 1590,1590` on T4) — otherwise
DVFS adds a large call-order artifact (details below). Full harness, per-call
CSVs, and charts: https://github.com/alexislowys/cudf-bench

## Expected behavior

Skewed keys have *lower* effective cardinality (smaller output, better cache
behavior — CPU libraries speed up), so at worst parity with uniform keys.

## Environment

- cudf 26.02.01 (`pip install cudf-cu12`), Google Colab, Tesla T4, CUDA 13.0
- also reproduced with default allocator and with 8 GiB preallocated RMM pool

## Additional context — a reading of the code (happy to be corrected)

#15262 added shared-memory aggregation to avoid "serializing atomic operations
over a small range of global memory". The gate is per-block distinct-key count vs
`GROUPBY_CARDINALITY_THRESHOLD = 128` (`cpp/src/groupby/hash/helpers.cuh`,
`compute_single_pass_aggs.cuh`).

Zipf-skewed data seems to be the worst case for this gate: the long tail pushes
each block's distinct count past 128, disabling the shared-memory path, while the
hot head concentrates most *updates* on a few global addresses — the exact
contention the path was built to avoid. The gate measures how many keys a block
sees, not how concentrated its updates are; only the latter drives atomic
serialization. If that reading is right, a frequency-aware gate (or the
originally-projected shared-memory-with-spill design) would cover the skewed case.

Observations consistent with this reading:
- 10 uniform keys (all blocks under threshold): fastest config measured
- 100k uniform keys (over threshold, updates spread wide): fast
- Zipf 1.5 over 100k keys (over threshold, updates concentrated): 1.48x slow
- skew sensitivity grows with the skew exponent (sweep data in the repo)

## Side note for other benchmarkers

On T4, DVFS (585 MHz idle → ~1590 MHz after ~6 sustained calls) produces a 1.5–2x
call-order artifact that can masquerade as library behavior — including a
convincing false dose–response we chased for a day. Per-call SM-clock logs and
locked-clock runs are in the repo if useful.
