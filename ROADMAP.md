# Roadmap

Goal: find, diagnose, and fix a real performance anomaly in NVIDIA's cuDF.
Started 2026-07-10.

## Phase 0 — Get a GPU (week 1, by Jul 17)

- [x] Repo + benchmark harness scaffolded
- [ ] Colab account, T4 runtime verified with `nvidia-smi`
- [ ] `notebooks/colab_run.ipynb` runs end-to-end on T4

## Phase 1 — Learn what "normal" looks like (weeks 1–2, by Jul 24)

- [ ] Redo a familiar pandas workflow in cuDF (API is nearly identical)
- [ ] First timings: cuDF vs pandas on 10M+ rows — expect 10–50x wins
- [ ] Read: "What is a CUDA kernel" + "GPU vs CPU memory (why copies are slow)"

## Phase 2 — Fair stopwatch (weeks 3–4, by Aug 7)

- [x] Data generator with size / columns / skew / string-length knobs
- [x] Timer handling warm-up, async execution, repetition (median of 5+)
- [x] CSV results logger with backend version + device metadata
- [ ] Sanity check on GPU: groupby at 3 sizes — bigger takes longer, cuDF beats pandas

## Phase 3 — Stress-test matrix (weeks 4–5, by Aug 14)

- [ ] Grid: 7 ops × sizes × skew × string lengths × (data > GPU memory)
- [ ] Overnight full-grid run, all results committed
- [ ] Speedup plots per grid cell; circle the anomalies
- [ ] **Blog post #1:** "I benchmarked cuDF across N scenarios — where it shines and where it struggles"

## Phase 4 — Find the jammed gear (weeks 5–7, by Aug 28)

- [ ] Pick the single worst anomaly
- [ ] Profile with Nsight Systems (`nsys profile python ...`), read the timeline
- [ ] Locate the slow path in cuDF's source on GitHub
- [ ] Search cuDF issues for prior reports
- [ ] One-paragraph hypothesis: "X is slow on Y data because Z"

## Phase 5 — Fix it or prove it (weeks 8–9, by Sep 11)

- [ ] Route A: evidence-backed GitHub issue → guided fix → PR to rapidsai/cudf
- [ ] Route B (fallback): custom kernel / smarter algorithm beating cuDF on the problem case

## Phase 6 — Tell the story (weeks 10–12, by Oct 2)

- [ ] README lets a stranger reproduce every number in < 30 min
- [ ] **Blog post #2:** anomaly → Nsight timeline → hypothesis → fix → before/after
- [ ] Share: Hacker News, r/CUDA, r/dataengineering, LinkedIn
- [ ] Resume line with the measured improvement

## Rules of thumb when stuck

- Stuck > 2 days on setup → switch environments (Colab ↔ cloud VM), don't fight it.
- Can't read a profiler timeline → post it on the RAPIDS Slack (rapids.ai); maintainers answer beginners.
- No anomalies anywhere → push harder on skew, strings, and data bigger than GPU memory. Anomalies live at the edges.
- Feeling unqualified to fix NVIDIA's code → Route B exists, and a well-documented issue report alone has real value.
