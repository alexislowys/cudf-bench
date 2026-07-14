# Roadmap (sprint)

Goal: find, diagnose, and fix a real performance anomaly in NVIDIA's cuDF — as fast as possible.
Each step is one focused session. Started 2026-07-10.

## Done

- [x] Repo + benchmark harness (fair timer: warmup/sync/median, deterministic datagen with skew, CSV logging)
- [x] Colab T4 verified, notebook runs end-to-end
- [x] First results: 10M rows, cuDF 18–40x vs pandas on numeric ops, 245–405x on string ops

## Step 1 — Stress grid (1 Colab session, ~1 hr)

- [x] Run pandas + polars + cudf across: sizes (1e6, 1e7, 3e7), skew (0, 1.5), string ops
- [ ] Push toward T4 memory limit (3e7+ rows) — OOM behavior is itself a finding
- [x] Commit results CSV

## Step 2 — Pick the anomaly (same day, ~30 min)

- [x] Compute full speedup grid; find cells where cuDF's win collapses, loses to Polars, or scales worse than linear
- [x] Pick the single worst one (groupby_agg on skewed keys)

## Step 3 — Diagnose (1–2 sessions)

- [x] Break the slow op into stages with targeted timings (skew sweep, str-col probe, GPU-event split, clock lock — Nsight not needed)
- [x] Find the slow path in cuDF source (GROUPBY_CARDINALITY_THRESHOLD gate); searched issues — no prior skew report, #15262 is the machinery
- [x] Hypothesis written with numbers — docs/FINDINGS.md

## Step 4 — Fix or prove (1–3 sessions, the real work)

- [ ] File evidence-backed issue on rapidsai/cudf (this alone is a contribution — do it early, maintainers respond while you keep working)
- [ ] Route A: PR the fix upstream (merge timing is theirs, not yours — don't block on it)
- [ ] Route B: faster custom implementation (Numba/CuPy kernel or smarter algorithm) beating cuDF on the problem case

## Step 5 — Write-up (1 session)

- [ ] README: a stranger reproduces every number in < 30 min
- [ ] One write-up: anomaly → profile → hypothesis → fix → before/after chart
- [ ] Post it (HN, r/CUDA, LinkedIn); resume line with measured numbers

## Rules of thumb when stuck

- Stuck > half a day on setup → switch environments, don't fight it.
- Can't read a profile → RAPIDS Slack (rapids.ai); maintainers answer beginners.
- No anomalies → push harder on skew, strings, and data bigger than GPU memory.
- "Not qualified to fix NVIDIA's code" → Route B exists; a well-documented issue report already has real value.
