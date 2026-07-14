"""Benchmark runner CLI.

Example:
    python -m benchmark.runner --backend pandas --ops groupby_agg,inner_join \
        --rows 1e6,1e7 --skew 0,1.1 --out results/results.csv

Every run appends one CSV row per (op x rows x skew) cell, tagged with
backend version and device, so results from different machines (laptop,
Colab T4) accumulate in one comparable log.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import platform
import sys
from pathlib import Path

from .backends import get_backend
from .datagen import make_join_tables
from .ops import OPS, run_op
from .timer import bench

CSV_FIELDS = [
    "timestamp",
    "host",
    "backend",
    "backend_version",
    "device",
    "op",
    "rows",
    "n_keys",
    "skew",
    "str_cols",
    "str_len",
    "warmup",
    "reps",
    "median_s",
    "times_s",
    "seed",
]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="benchmark.runner", description="Run dataframe benchmarks."
    )
    p.add_argument("--backend", required=True, choices=["pandas", "polars", "cudf"])
    p.add_argument(
        "--ops",
        default=",".join(OPS),
        help=f"comma-separated ops (default: all). Available: {', '.join(OPS)}",
    )
    p.add_argument("--rows", default="1e6", help="comma-separated row counts, e.g. 1e6,1e7")
    p.add_argument("--skew", default="0", help="comma-separated key-skew exponents, e.g. 0,1.1")
    p.add_argument("--n-keys", type=int, default=None, help="distinct keys (default rows/10, capped 100k)")
    p.add_argument("--str-len", type=int, default=12)
    p.add_argument(
        "--str-cols",
        type=int,
        default=None,
        help="string columns in the table (default: 1 if any string op selected, else 0)",
    )
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/results.csv")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="tiny fast sanity run (1e4 rows, 2 reps); overrides --rows/--reps",
    )
    args = p.parse_args(argv)

    args.ops = [s.strip() for s in args.ops.split(",") if s.strip()]
    unknown = [o for o in args.ops if o not in OPS]
    if unknown:
        p.error(f"unknown ops: {unknown}. Available: {sorted(OPS)}")
    args.rows = [int(float(s)) for s in args.rows.split(",")]
    args.skew = [float(s) for s in args.skew.split(",")]
    if args.smoke:
        args.rows = [10_000]
        args.reps = 2
    return args


def append_row(out_path: Path, row: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not out_path.exists()
    if not is_new:
        with out_path.open() as f:
            existing = f.readline().strip().split(",")
        if existing != CSV_FIELDS:
            raise SystemExit(
                f"{out_path} has an older column layout; write to a new file instead"
            )
    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main(argv=None) -> None:
    args = parse_args(argv)
    backend = get_backend(args.backend)
    out_path = Path(args.out)
    device = backend.device()
    print(f"backend={backend.name} {backend.version()} | device={device}")

    for rows in args.rows:
        for skew in args.skew:
            need_strings = any(OPS[o].needs_strings for o in args.ops)
            str_cols = args.str_cols if args.str_cols is not None else (1 if need_strings else 0)
            if need_strings and str_cols < 1:
                raise SystemExit("selected string ops need --str-cols >= 1")
            left_pd, right_pd = make_join_tables(
                rows=rows,
                n_keys=args.n_keys,
                skew=skew,
                str_cols=str_cols,
                str_len=args.str_len,
                seed=args.seed,
            )
            # conversion (incl. any host->GPU copy) happens outside the timed region
            left = backend.from_pandas(left_pd)
            right = backend.from_pandas(right_pd)
            backend.sync()

            for op_name in args.ops:
                op = OPS[op_name]
                result = bench(
                    lambda: run_op(op, backend.name, left, right),
                    sync=backend.sync,
                    warmup=args.warmup,
                    reps=args.reps,
                )
                print(
                    f"  {op_name:<14} rows={rows:<12,} skew={skew:<5} "
                    f"median={result.median_s:.4f}s"
                )
                append_row(
                    out_path,
                    {
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "host": platform.node(),
                        "backend": backend.name,
                        "backend_version": backend.version(),
                        "device": device,
                        "op": op_name,
                        "rows": rows,
                        "n_keys": args.n_keys or "",
                        "skew": skew,
                        "str_cols": str_cols,
                        "str_len": args.str_len,
                        "warmup": args.warmup,
                        "reps": args.reps,
                        "median_s": f"{result.median_s:.6f}",
                        "times_s": json.dumps([round(t, 6) for t in result.times_s]),
                        "seed": args.seed,
                    },
                )

    print(f"results appended to {out_path}")


if __name__ == "__main__":
    sys.exit(main())
