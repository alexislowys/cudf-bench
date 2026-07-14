"""Per-iteration timing of repeated cuDF groupby calls, including the cold first call.

Probes the warm-up transient found in the string-column probe (results/probe.csv):
skewed groupby is slow for the first few calls in a process, then settles.
Hypothesis: the GPU memory pool growing on demand. --rmm-pool prealloc
pre-grows the pool before any work; if the transient disappears, the
allocator is the culprit.

GPU-only — run on Colab: python -m benchmark.transient --skew 1.5 --str-cols 2
"""

from __future__ import annotations

import argparse
import csv
import datetime
import platform
import time
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(prog="benchmark.transient")
    p.add_argument("--skew", type=float, default=0.0)
    p.add_argument("--str-cols", type=int, default=0)
    p.add_argument("--rows", type=float, default=1e7)
    p.add_argument("--iters", type=int, default=12)
    p.add_argument("--rmm-pool", choices=["default", "prealloc"], default="default")
    p.add_argument("--pool-gib", type=int, default=8)
    p.add_argument("--out", default="results/transient.csv")
    args = p.parse_args()

    import cudf  # noqa: F401  (GPU only)
    import cupy

    if args.rmm_pool == "prealloc":
        import rmm

        rmm.reinitialize(pool_allocator=True, initial_pool_size=args.pool_gib << 30)

    from benchmark.datagen import make_table

    pdf = make_table(int(args.rows), str_cols=args.str_cols, skew=args.skew, seed=0)
    gdf = cudf.from_pandas(pdf)
    cupy.cuda.runtime.deviceSynchronize()

    times: list[float] = []
    for _ in range(args.iters):
        cupy.cuda.runtime.deviceSynchronize()
        start = time.perf_counter()
        res = gdf.groupby("key0").agg({"val0": "mean", "val1": "sum"})
        len(res)
        cupy.cuda.runtime.deviceSynchronize()
        times.append(time.perf_counter() - start)

    print(f"skew={args.skew} str_cols={args.str_cols} pool={args.rmm_pool}")
    print("  " + " ".join(f"{t:.4f}" for t in times))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    is_new = not out.exists()
    with out.open("a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(
                ["timestamp", "host", "device", "rows", "skew", "str_cols", "rmm_pool", "iter", "time_s"]
            )
        stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        device = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
        for i, t in enumerate(times):
            writer.writerow(
                [stamp, platform.node(), device, int(args.rows), args.skew, args.str_cols, args.rmm_pool, i, round(t, 6)]
            )


if __name__ == "__main__":
    main()
