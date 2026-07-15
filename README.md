# cudf-bench

**Found, diagnosed, and reported a performance defect in NVIDIA's cuDF GPU dataframe
library — reproduced by a RAPIDS maintainer on GH200 hardware, with a kernel-level fix
direction now under investigation upstream.**

> Upstream report + maintainer discussion: [rapidsai/cudf#23256](https://github.com/rapidsai/cudf/issues/23256) ·
> Full analysis: [docs/FINDINGS.md](docs/FINDINGS.md)

![skew penalty](results/figs/skew_penalty.png)

## The finding

cuDF's hash groupby is **~1.5x slower on skewed (Zipf) keys** than on uniform keys —
measured at locked GPU clocks so the gap is pure kernel time — while pandas and Polars
get ~2x *faster* on the same input. Real-world keys (customers, products, URLs) are
almost always skewed.

**Mechanism** (confirmed and refined upstream): the Zipf input's hottest key owns
38.4% of all rows — 3.84M atomic adds serializing on a single output slot in the
global-atomic aggregation path. libcudf's shared-memory rescue path is deliberately
reserved for far more extreme concentration (1–2 keys), so mid-skew data takes the
contended path. A RAPIDS maintainer reproduced the penalty on a GH200 (1.63x),
corrected my original code-path reading (kept for the record in FINDINGS), and is
looking into warp-aggregating same-key updates before the atomic as the fix.

**Prototype** ([benchmark/routeb.py](benchmark/routeb.py)): route sampled heavy
hitters through per-block shared memory, tail keys to storage spread wide. Identical
results to cuDF (asserted), **2.47x faster on the skewed case** (caveats in FINDINGS —
it exploits a dense-int-key specialization a general library cannot).

**Bonus methodology finding:** the T4 idles at 585 MHz and boosts to 1590 MHz only
after ~6 back-to-back calls. Naive benchmark loops measure boost clocks; real
single-shot users run at idle clocks up to 2x slower — and this artifact produced a
convincing false lead (see the dead-hypotheses table in FINDINGS) until per-call clock
logging and `nvidia-smi -lgc` exposed it.

![dvfs](results/figs/dvfs_transient.png)

## Reproduce everything

**CPU parts (any machine):**
```bash
pip install -e ".[dev]"
pytest                       # incl. pandas↔polars result-equivalence guards
python scripts/make_figs.py  # regenerate figures from committed CSVs
```

**GPU parts (free Colab T4):**
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alexislowys/cudf-bench/blob/main/notebooks/colab_run.ipynb)
— Runtime → T4 GPU → Run all. Every experiment in `results/*.csv` is a committed
notebook or `benchmark/*.py` script; the minimal repro is in the issue itself.

## Why trust these numbers

The harness (`benchmark/`) controls the classic GPU timing traps: warm-up runs,
device synchronization before stopping the clock, medians over repeated calls with
**every per-call time logged** (that habit is what caught the DVFS confound), byte-identical
input across backends, correctness tests before speed tests, and — for final numbers —
locked GPU clocks and preallocated memory pools. CSV logs carry device, library
versions, and full data-shape parameters for every row.

## Repo layout

```
benchmark/        harness: backends, datagen (skew dial), fair timer, runner CLI,
                  transient.py (per-call GPU/clock probes), routeb.py (prototype fix)
notebooks/        colab_run.ipynb (current experiment) + archived steps
results/          every measurement ever taken (CSV) + figures
docs/             FINDINGS.md (the analysis), issue-draft.md (as filed)
tests/            datagen properties, cross-backend result equivalence, timer behavior
```

## Timeline

Built and diagnosed 2026-07-10 → 2026-07-14 (harness → 112-cell stress grid → anomaly
→ five falsified hypotheses → source-level mechanism → upstream issue → prototype).
[ROADMAP.md](ROADMAP.md) tracks the remaining polish.
