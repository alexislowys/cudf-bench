# I benchmarked NVIDIA's GPU dataframe library and found a bug in four days

*Draft — Alexis's voice, edit before posting. Target: personal blog / LinkedIn article; condensed version for HN/r/CUDA comment.*

---

I'm a data science student. Two weeks ago my plan was modest: learn GPU computing by
benchmarking [cuDF](https://github.com/rapidsai/cudf) — NVIDIA's drop-in GPU
replacement for pandas — against pandas and Polars, and write up what I found.

Instead I ended up filing a performance bug against NVIDIA's flagship data library
([rapidsai/cudf#23256](https://github.com/rapidsai/cudf/issues/23256)), with the
mechanism traced to a specific constant in their CUDA source, and a prototype showing
the fix direction runs 2.47x faster. Here's the whole chase, including the three wrong
theories I believed on the way.

## The setup: build a stopwatch you can trust

GPU benchmarks lie by default. The first run pays one-time compilation costs. The GPU
returns control before it's finished computing. Single measurements are noise. So
before measuring anything I built a small harness that does warm-up runs, synchronizes
the device before stopping the clock, repeats every measurement and — this turned out
to matter most — **logs every individual repetition, not just the median**.

Then I ran a grid: 7 operations × 3 table sizes × 3 libraries × two key
distributions — uniform, and Zipf-skewed, where a few keys own most of the rows, which
is what real data (customers, products, URLs) actually looks like.

## The anomaly

One cell of the grid misbehaved. On skewed keys, pandas got 2x faster — fewer distinct
groups, smaller hash table, happier cache. Polars: 3.3x faster. cuDF got **1.65x
slower**. Same computation, easier data, opposite direction. cuDF's lead over Polars
collapsed from 33x to 6x.

An easier job with a worse result is an engine problem, not a workload problem. I had
my anomaly.

## Four theories, four funerals

**Theory 1: atomic contention.** Thousands of GPU threads all updating the same hot
key's accumulator must serialize. Classic. Testable: uniform data with only 10
distinct keys maximizes that collision pattern. Result: *fastest configuration I
measured*. Theory dead.

**Theory 2: memory allocator.** Per-repetition logs showed the skewed groupby wasn't
uniformly slow — the first ~5 calls in a process ran 1.6x slow, then it snapped to
fast. Looks like a memory pool growing on demand. Testable: preallocate an 8 GiB pool
up front. Result: transient completely unchanged. Theory dead.

**Theory 3: unused string columns.** The slowdown seemed to appear only when the table
carried string columns the groupby never touched, and it scaled with how many — a
beautiful dose–response. I believed this one hardest. It died the strangest death.

## The twist: my benchmark was measuring the power manager

CUDA event timers proved the "extra" time was genuine GPU work. Then one control
changed everything: put 2 seconds of idle between calls, and the groupby **never**
speeds up. The "warm-up" wasn't the software warming up — it was the *silicon*.

A Tesla T4 idles at 585 MHz and only boosts toward 1590 MHz after several back-to-back
kernels. Logging the SM clock at every call made it undeniable: the moment the clock
column jumped, the time column dropped, same call number. Lock the clocks with
`nvidia-smi -lgc` and every transient vanishes — including the string-column
dose–response, which was just longer data transfers changing the clock state entering
each measurement.

Two uncomfortable implications. Most GPU benchmark loops measure boost-clock
performance, but real users calling groupby once run at idle clocks — up to 2x slower
than the number in the README. And if I hadn't logged every repetition, I'd have
published the string-column theory with a straight face and a convincing chart.

## What survived: the real bug

At locked clocks, preallocated pool, everything controlled: skewed groupby 23.7 ms,
uniform 16.0 ms. **1.48x, pure kernel time, only groupby** (sort doesn't care), only
cuDF (CPU libraries speed up).

I read libcudf's source and formed a theory about their shared-memory fast path's
cardinality gate (`GROUPBY_CARDINALITY_THRESHOLD = 128`) having a blind spot for
skew. I put that reading in the issue, flagged as "happy to be corrected."

**Theory 4 died too — killed by NVIDIA.** A RAPIDS maintainer reproduced my numbers
on a GH200 within hours (1.63x there — so it's not a T4 quirk) and corrected the
mechanism: my Zipf data has ~42k distinct keys with the hottest key owning **38.4% of
all rows**. That's 3.84 million atomic adds serializing on *one output slot* in the
plain global-atomic path. The shared-memory rescue path was never in play — it's
deliberately reserved for far more extreme concentration (rows collapsing onto 1–2
keys, where contention costs 30–100x, not 1.6x). My contention instinct was right;
my story about *which code path* was wrong. Their proposed real fix —
warp-aggregating same-key updates before the atomic — is now something they're
looking into.

## Proving the fix direction

I wrote a Numba CUDA prototype that does what a frequency-aware gate would do: hot keys
accumulate in per-block shared memory (one global write per block at the end), tail
keys go to storage spread wide enough that atomics don't collide. Identical output to
cuDF, asserted before timing. **2.47x faster on the skewed case.**

Honesty box: my prototype exploits dense integer keys and skips hashing entirely — a
general-purpose library can't assume that, which is also why it wins on uniform data.
It's evidence for the structure, not a drop-in patch.

## Filed upstream

Everything went into [rapidsai/cudf#23256](https://github.com/rapidsai/cudf/issues/23256):
minimal repro, locked-clock numbers, the code reading, the fix direction. Every number
in this post reproduces from committed CSVs at
[github.com/alexislowys/cudf-bench](https://github.com/alexislowys/cudf-bench) — the
GPU parts on a free Colab T4.

## What I'd tell you if you try this

1. **Log every repetition.** Medians hide the shape of the data, and the shape is
   where the clues live. Every breakthrough in this project came from looking at
   rep-by-rep times.
2. **Your GPU benchmark is measuring DVFS until proven otherwise.** Log clocks, lock
   clocks.
3. **Dead hypotheses are progress.** I was wrong four times — three killed by my own
   experiments, one by an NVIDIA engineer. Each death narrowed the truth; publishing
   the graveyard is what makes the surviving claim credible.
4. **You don't need permission or hardware.** Free Colab, an afternoon of Python, and
   stubbornness got a student's name onto NVIDIA's issue tracker.
