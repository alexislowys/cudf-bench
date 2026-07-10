"""Load benchmark results and compute/plot speedups.

Minimal for now — Phase 3 (the stress-test matrix) will grow this into
proper grid heatmaps.
"""

from __future__ import annotations

import pandas as pd


def load_results(path: str = "results/results.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["median_s"] = df["median_s"].astype(float)
    return df


def speedup_table(df: pd.DataFrame, baseline: str = "pandas") -> pd.DataFrame:
    """Median seconds per backend, plus speedup of each backend vs the baseline.

    If the same cell was benchmarked more than once, the latest run wins.
    """
    latest = (
        df.sort_values("timestamp")
        .groupby(["op", "rows", "skew", "backend"], as_index=False)
        .last()
    )
    pivot = latest.pivot_table(
        index=["op", "rows", "skew"], columns="backend", values="median_s"
    )
    if baseline not in pivot.columns:
        raise ValueError(f"no {baseline!r} runs in results — run that backend first")
    for backend in pivot.columns:
        if backend != baseline:
            pivot[f"speedup_{backend}_vs_{baseline}"] = pivot[baseline] / pivot[backend]
    return pivot


def plot_speedups(df: pd.DataFrame, backend: str = "cudf", baseline: str = "pandas", ax=None):
    import matplotlib.pyplot as plt

    table = speedup_table(df, baseline=baseline)
    col = f"speedup_{backend}_vs_{baseline}"
    if col not in table.columns:
        raise ValueError(f"no {backend!r} runs in results")
    labels = [f"{op}\n{rows:,} rows, skew={skew}" for op, rows, skew in table.index]
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))
    ax.bar(labels, table[col])
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    ax.set_ylabel(f"{backend} speedup vs {baseline} (x)")
    ax.set_title(f"{backend} vs {baseline} (above dashed line = {backend} wins)")
    ax.tick_params(axis="x", labelsize=8)
    return ax
