import statistics

import pytest

from benchmark.timer import BenchResult, bench, touch


def test_warmup_and_reps_counted():
    calls = {"fn": 0, "sync": 0}

    def fn():
        calls["fn"] += 1
        return [1, 2, 3]

    def sync():
        calls["sync"] += 1

    result = bench(fn, sync=sync, warmup=2, reps=3)
    assert calls["fn"] == 5  # 2 warmup + 3 timed
    assert len(result.times_s) == 3  # warmup runs never appear in timings
    # each timed rep syncs before start and after touch; warmups sync once each
    assert calls["sync"] == 2 + 3 * 2


def test_median_is_median():
    result = bench(lambda: None, reps=5)
    assert result.median_s == statistics.median(result.times_s)
    assert isinstance(result, BenchResult)


def test_reps_must_be_positive():
    with pytest.raises(ValueError):
        bench(lambda: None, reps=0)


def test_touch_handles_unsized_results():
    touch(None)
    touch(42)  # no __len__ — must not raise
    touch([1, 2])
