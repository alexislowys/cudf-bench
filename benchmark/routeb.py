"""Route B prototype: heavy-hitter-aware groupby aggregation.

The diagnosis (docs/FINDINGS.md, rapidsai/cudf#23256): on skewed keys, cuDF's
hash groupby aggregates hot keys through contended global-memory atomics because
the shared-memory fast path is gated on per-block distinct-key count, which the
long tail blows past.

This prototype does what a frequency-aware gate would do:
1. sample the key column, pick up to 128 heavy hitters,
2. one pass: heavy keys accumulate in per-block SHARED memory (one global
   atomic per block per hot key at the end), tail keys go straight to a dense
   global table (updates spread wide, so atomics don't serialize),
3. combine, compute mean/sum, compare against cudf for correctness and speed.

Scope (a prototype, not a library): int keys in [0, n_keys), mean+sum aggs —
the exact benchmarked case. GPU-only:
    python -m benchmark.routeb --skew 1.5 --iters 8
"""

from __future__ import annotations

import argparse
import csv
import datetime
import platform
import time
from pathlib import Path

MAX_HOT = 128
TPB = 256


def main() -> None:
    p = argparse.ArgumentParser(prog="benchmark.routeb")
    p.add_argument("--skew", type=float, default=1.5)
    p.add_argument("--rows", type=float, default=1e7)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--out", default="results/routeb.csv")
    args = p.parse_args()

    import cudf
    import cupy as cp
    import numpy as np
    from numba import cuda, float64, int64

    from benchmark.datagen import make_table

    @cuda.jit
    def agg_kernel(keys, v0, v1, hot_map, n_hot, hot_s0, hot_s1, hot_cnt,
                   tail_s0, tail_s1, tail_cnt):
        sh_s0 = cuda.shared.array(MAX_HOT, float64)
        sh_s1 = cuda.shared.array(MAX_HOT, float64)
        sh_c = cuda.shared.array(MAX_HOT, int64)
        tid = cuda.threadIdx.x
        i = tid
        while i < n_hot:
            sh_s0[i] = 0.0
            sh_s1[i] = 0.0
            sh_c[i] = 0
            i += TPB
        cuda.syncthreads()
        j = cuda.grid(1)
        stride = cuda.gridsize(1)
        while j < keys.size:
            k = keys[j]
            slot = hot_map[k]
            if slot >= 0:
                cuda.atomic.add(sh_s0, slot, v0[j])
                cuda.atomic.add(sh_s1, slot, v1[j])
                cuda.atomic.add(sh_c, slot, 1)
            else:
                cuda.atomic.add(tail_s0, k, v0[j])
                cuda.atomic.add(tail_s1, k, v1[j])
                cuda.atomic.add(tail_cnt, k, 1)
            j += stride
        cuda.syncthreads()
        i = tid
        while i < n_hot:
            if sh_c[i] > 0:
                cuda.atomic.add(hot_s0, i, sh_s0[i])
                cuda.atomic.add(hot_s1, i, sh_s1[i])
                cuda.atomic.add(hot_cnt, i, sh_c[i])
            i += TPB

    def hha_groupby(keys: cp.ndarray, v0: cp.ndarray, v1: cp.ndarray, n_keys: int):
        """Heavy-hitter-aware groupby: returns (present_keys, mean_v0, sum_v1)."""
        # 1. find heavy hitters from a strided sample
        sample = keys[:: max(1, keys.size // 100_000)]
        counts = cp.bincount(sample, minlength=n_keys)
        order = cp.argsort(counts)[::-1][:MAX_HOT]
        heavy = order[counts[order] > sample.size // 256]
        hot_map = cp.full(n_keys, -1, dtype=cp.int32)
        hot_map[heavy] = cp.arange(heavy.size, dtype=cp.int32)

        # 2. one aggregation pass
        hot_s0 = cp.zeros(MAX_HOT); hot_s1 = cp.zeros(MAX_HOT)
        hot_cnt = cp.zeros(MAX_HOT, dtype=cp.int64)
        tail_s0 = cp.zeros(n_keys); tail_s1 = cp.zeros(n_keys)
        tail_cnt = cp.zeros(n_keys, dtype=cp.int64)
        blocks = 640
        agg_kernel[blocks, TPB](keys, v0, v1, hot_map, int(heavy.size),
                                hot_s0, hot_s1, hot_cnt, tail_s0, tail_s1, tail_cnt)

        # 3. scatter hot results into the dense table, then finalize
        if heavy.size:
            tail_s0[heavy] = hot_s0[: heavy.size]
            tail_s1[heavy] = hot_s1[: heavy.size]
            tail_cnt[heavy] = hot_cnt[: heavy.size]
        present = cp.nonzero(tail_cnt)[0]
        return present, tail_s0[present] / tail_cnt[present], tail_s1[present]

    pdf = make_table(int(args.rows), skew=args.skew, seed=0)
    n_keys = int(pdf["key0"].max()) + 1
    gdf = cudf.from_pandas(pdf)
    keys = cp.asarray(gdf["key0"].values)
    v0 = cp.asarray(gdf["val0"].values)
    v1 = cp.asarray(gdf["val1"].values)

    # correctness first: prototype must match cudf exactly (sorted by key)
    ref = gdf.groupby("key0").agg({"val0": "mean", "val1": "sum"}).sort_index()
    pk, pm, ps = hha_groupby(keys, v0, v1, n_keys)
    assert cp.allclose(cp.asarray(ref.index.values), pk)
    assert cp.allclose(cp.asarray(ref["val0"].values), pm, atol=1e-9)
    assert cp.allclose(cp.asarray(ref["val1"].values), ps, atol=1e-6)
    print(f"correctness: OK ({int(pk.size)} groups match cudf)")

    def bench(fn):
        times = []
        for _ in range(args.iters):
            cp.cuda.runtime.deviceSynchronize()
            t0 = time.perf_counter()
            r = fn()
            cp.cuda.runtime.deviceSynchronize()
            times.append(time.perf_counter() - t0)
        return times

    results = {
        "cudf": bench(lambda: gdf.groupby("key0").agg({"val0": "mean", "val1": "sum"})),
        "hha_prototype": bench(lambda: hha_groupby(keys, v0, v1, n_keys)),
    }
    med = {m: sorted(t)[len(t) // 2] for m, t in results.items()}
    print(f"skew={args.skew} rows={int(args.rows):,}")
    for m, t in results.items():
        print(f"  {m:14s} median {med[m]*1000:7.2f} ms   " + " ".join(f"{x*1000:.1f}" for x in t))
    print(f"  speedup vs cudf: {med['cudf'] / med['hha_prototype']:.2f}x")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    is_new = not out.exists()
    with out.open("a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "host", "device", "rows", "skew", "method", "iter", "time_s"])
        stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        device = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        for method, times in results.items():
            for i, t in enumerate(times):
                writer.writerow([stamp, platform.node(), device, int(args.rows), args.skew, method, i, round(t, 6)])


if __name__ == "__main__":
    main()
