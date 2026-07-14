"""Generate the two headline figures from committed results.

Usage: python scripts/make_figs.py   (writes results/figs/*.png)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "results" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

# validated categorical palette (fixed slot order)
BLUE, AQUA, YELLOW, GREEN = "#2a78d6", "#1baf7a", "#eda100", "#008300"
INK, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11,
})


def fig_dvfs() -> None:
    """Per-call groupby time: the DVFS artifact and the locked-clock truth."""
    df = pd.read_csv(ROOT / "results" / "transient3.csv")
    sel = {
        ("free clocks, skewed keys", BLUE): (1.5, 0.0, "default"),
        ("free clocks, skewed, 2s gaps", YELLOW): (1.5, 2.0, "default"),
        ("locked 1590 MHz, skewed keys", AQUA): (1.5, 0.0, "prealloc"),
        ("locked 1590 MHz, uniform keys", GREEN): (0.0, 0.0, "prealloc"),
    }
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for (label, color), (skew, gap, pool) in sel.items():
        g = df[(df["skew"] == skew) & (df["gap"] == gap) & (df["rmm_pool"] == pool)].sort_values("iter")
        g = g[g["iter"] > 0]  # call 0 pays one-off JIT/pool costs; separate story
        ax.plot(g["iter"], g["wall_ms"], marker="o", ms=5, lw=2, color=color, label=label)
        ax.annotate(label, (g["iter"].iloc[-1] + 0.15, g["wall_ms"].iloc[-1]),
                    color=color, fontsize=9, va="center")
    ax.set_xlim(0.5, 16.5)
    ax.set_xlabel("consecutive call # (same process, 10M rows)")
    ax.set_ylabel("groupby time (ms)")
    ax.set_title("GPU clock boosting masquerades as cuDF behavior — locked clocks tell the truth")
    ax.grid(axis="y", lw=0.5, alpha=0.35)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGS / "dvfs_transient.png", dpi=160)
    print("wrote", FIGS / "dvfs_transient.png")


def fig_skew_penalty() -> None:
    """Skew penalty ratio per backend — the finding in one image."""
    # cuDF at locked clocks (transient3, steady-state medians of calls 1..11)
    t3 = pd.read_csv(ROOT / "results" / "transient3.csv")
    locked = t3[(t3["rmm_pool"] == "prealloc") & (t3["gap"] == 0.0) & (t3["iter"] > 0)]
    cudf_ratio = (
        locked[locked["skew"] == 1.5]["wall_ms"].median()
        / locked[locked["skew"] == 0.0]["wall_ms"].median()
    )
    # CPU baselines from the string-column probe (str_cols matched, free CPU clocks irrelevant)
    probe = pd.read_csv(ROOT / "results" / "probe.csv")
    pol = probe[(probe["backend"] == "polars") & (probe["str_cols"] == 1)]
    polars_ratio = (
        pol[pol["skew"] == 1.5]["median_s"].iloc[0] / pol[pol["skew"] == 0.0]["median_s"].iloc[0]
    )
    grid = pd.read_csv(ROOT / "results" / "results.csv")
    pan = grid[(grid["backend"] == "pandas") & (grid["op"] == "groupby_agg") & (grid["rows"] == 10_000_000)]
    pandas_ratio = (
        pan[pan["skew"] == 1.5]["median_s"].iloc[0] / pan[pan["skew"] == 0.0]["median_s"].iloc[0]
    )

    labels = ["cuDF (T4, locked clocks)", "Polars (CPU)", "pandas (CPU)"]
    ratios = [cudf_ratio, polars_ratio, pandas_ratio]
    colors = [BLUE, AQUA, YELLOW]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(labels, ratios, color=colors, width=0.55)
    for b, r in zip(bars, ratios):
        ax.annotate(f"{r:.2f}x", (b.get_x() + b.get_width() / 2, r),
                    ha="center", va="bottom", fontsize=12, color=INK, fontweight="bold")
    ax.axhline(1.0, color=MUTED, lw=1.2, ls="--")
    ax.annotate("1.0 = skew makes no difference", (2.35, 1.02), color=MUTED,
                fontsize=9, ha="right")
    ax.set_ylabel("time on skewed keys ÷ time on uniform keys")
    ax.set_title("Same groupby, same 10M rows: skewed keys slow cuDF down,\nspeed CPU libraries up")
    ax.grid(axis="y", lw=0.5, alpha=0.35)
    fig.tight_layout()
    fig.savefig(FIGS / "skew_penalty.png", dpi=160)
    print("wrote", FIGS / "skew_penalty.png")


if __name__ == "__main__":
    fig_dvfs()
    fig_skew_penalty()
