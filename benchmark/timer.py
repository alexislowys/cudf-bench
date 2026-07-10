"""Fair stopwatch for (possibly asynchronous) dataframe operations.

Handles the three classic GPU timing traps:
1. Warm-up: the first GPU run pays one-time setup (kernel compilation,
   memory pool growth) — run untimed first.
2. Async execution: cuDF can return before the GPU finishes — sync and
   touch the result before stopping the clock.
3. Noise: single runs lie — repeat and take the median.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class BenchResult:
    median_s: float
    times_s: list[float] = field(default_factory=list)


def touch(result) -> None:
    """Force lazy/async results to materialize by observing them."""
    if result is None:
        return
    try:
        len(result)
    except TypeError:
        pass


def bench(
    fn: Callable[[], object],
    sync: Callable[[], None] = lambda: None,
    warmup: int = 1,
    reps: int = 5,
) -> BenchResult:
    if reps < 1:
        raise ValueError("reps must be >= 1")

    for _ in range(warmup):
        result = fn()
        touch(result)
        sync()

    times: list[float] = []
    for _ in range(reps):
        sync()  # nothing pending when the clock starts
        start = time.perf_counter()
        result = fn()
        touch(result)
        sync()  # everything finished when the clock stops
        times.append(time.perf_counter() - start)

    return BenchResult(median_s=statistics.median(times), times_s=times)
