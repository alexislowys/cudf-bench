"""Per-call timing of repeated cuDF ops, split into GPU busy-time vs wall time.

Round 1 (results/transient.csv) found: skewed groupby runs ~1.6x slow for the
first ~5 calls in a process, then snaps to fast. A preallocated RMM pool does
NOT remove the transient, so the allocator is not the cause.

Round 2 adds three discriminators:
- GPU event timers per call: is the extra time GPU work or CPU-side overhead?
- --gap N: sleep between calls — does idle time reset the transient?
- --op sort: is the transient groupby-specific or general under skew?

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
    p.add_argument("--op", choices=["groupby", "sort"], default="groupby")
    p.add_argument("--skew", type=float, default=0.0)
    p.add_argument("--str-cols", type=int, default=0)
    p.add_argument("--rows", type=float, default=1e7)
    p.add_argument("--iters", type=int, default=12)
    p.add_argument("--gap", type=float, default=0.0, help="seconds to sleep between calls")
    p.add_argument("--rmm-pool", choices=["default", "prealloc"], default="default")
    p.add_argument("--pool-gib", type=int, default=8)
    p.add_argument("--out", default="results/transient3.csv")
    args = p.parse_args()

    import cudf
    import cupy

    # SM clock per call — the DVFS suspect needs direct evidence
    try:
        import pynvml

        pynvml.nvmlInit()
        _handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        def sm_clock() -> int:
            return pynvml.nvmlDeviceGetClockInfo(_handle, pynvml.NVML_CLOCK_SM)
    except Exception:

        def sm_clock() -> int:
            return -1

    if args.rmm_pool == "prealloc":
        import rmm

        rmm.reinitialize(pool_allocator=True, initial_pool_size=args.pool_gib << 30)

    from benchmark.datagen import make_table

    pdf = make_table(int(args.rows), str_cols=args.str_cols, skew=args.skew, seed=0)
    gdf = cudf.from_pandas(pdf)
    cupy.cuda.runtime.deviceSynchronize()

    if args.op == "groupby":
        run = lambda: gdf.groupby("key0").agg({"val0": "mean", "val1": "sum"})
    else:
        run = lambda: gdf.sort_values("val0")

    rows_out: list[tuple[int, float, float, int]] = []
    for i in range(args.iters):
        if args.gap:
            time.sleep(args.gap)
        cupy.cuda.runtime.deviceSynchronize()
        clock = sm_clock()
        ev_start, ev_stop = cupy.cuda.Event(), cupy.cuda.Event()
        wall_start = time.perf_counter()
        ev_start.record()
        res = run()
        len(res)
        ev_stop.record()
        ev_stop.synchronize()
        wall = time.perf_counter() - wall_start
        gpu_ms = cupy.cuda.get_elapsed_time(ev_start, ev_stop)
        rows_out.append((i, wall * 1000, gpu_ms, clock))

    print(f"op={args.op} skew={args.skew} str_cols={args.str_cols} pool={args.rmm_pool} gap={args.gap}")
    print("  wall(ms): " + " ".join(f"{w:5.1f}" for _, w, _, _ in rows_out))
    print("  gpu (ms): " + " ".join(f"{g:5.1f}" for _, _, g, _ in rows_out))
    print("  sm (MHz): " + " ".join(f"{c:5d}" for _, _, _, c in rows_out))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    is_new = not out.exists()
    with out.open("a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(
                ["timestamp", "host", "device", "op", "rows", "skew", "str_cols",
                 "rmm_pool", "gap", "iter", "wall_ms", "gpu_ms", "sm_mhz"]
            )
        stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        device = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
        for i, wall_ms, gpu_ms, clock in rows_out:
            writer.writerow(
                [stamp, platform.node(), device, args.op, int(args.rows), args.skew,
                 args.str_cols, args.rmm_pool, args.gap, i, round(wall_ms, 3), round(gpu_ms, 3), clock]
            )


if __name__ == "__main__":
    main()
