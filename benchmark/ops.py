"""Operation registry.

pandas and cudf share (nearly) the same API, so one implementation covers
both; polars gets its own. Every op takes backend-native tables and returns
a backend-native result — conversion cost is deliberately excluded from
timing (the runner converts before the stopwatch starts).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

ROLLING_WINDOW = 100
NEEDLE = "ab"  # substring for str ops; random 12-char lowercase strings contain it ~1.6% of the time


@dataclass(frozen=True)
class Op:
    name: str
    fn_pandas_like: Callable  # pandas and cudf
    fn_polars: Callable
    needs_join_table: bool = False
    needs_strings: bool = False


def _pl():
    import polars

    return polars


# --- pandas / cudf implementations -----------------------------------------


def _groupby_agg_pd(df):
    return df.groupby("key0").agg({"val0": "mean", "val1": "sum"})


def _inner_join_pd(df, right):
    return df.merge(right, on="key0", how="inner")


def _sort_pd(df):
    return df.sort_values("val0")


def _filter_pd(df):
    return df[df["val0"] > 0.5]


def _str_contains_pd(df):
    return df[df["str0"].str.contains(NEEDLE, regex=False)]


def _str_replace_pd(df):
    return df["str0"].str.replace("a", "z", regex=False)


def _rolling_mean_pd(df):
    return df["val0"].rolling(ROLLING_WINDOW).mean()


# --- polars implementations -------------------------------------------------


def _groupby_agg_pl(df):
    pl = _pl()
    return df.group_by("key0").agg(pl.col("val0").mean(), pl.col("val1").sum())


def _inner_join_pl(df, right):
    return df.join(right, on="key0", how="inner")


def _sort_pl(df):
    return df.sort("val0")


def _filter_pl(df):
    pl = _pl()
    return df.filter(pl.col("val0") > 0.5)


def _str_contains_pl(df):
    pl = _pl()
    return df.filter(pl.col("str0").str.contains(NEEDLE, literal=True))


def _str_replace_pl(df):
    pl = _pl()
    return df.select(pl.col("str0").str.replace_all("a", "z", literal=True))


def _rolling_mean_pl(df):
    pl = _pl()
    return df.select(pl.col("val0").rolling_mean(window_size=ROLLING_WINDOW))


OPS: dict[str, Op] = {
    op.name: op
    for op in [
        Op("groupby_agg", _groupby_agg_pd, _groupby_agg_pl),
        Op("inner_join", _inner_join_pd, _inner_join_pl, needs_join_table=True),
        Op("sort", _sort_pd, _sort_pl),
        Op("filter", _filter_pd, _filter_pl),
        Op("str_contains", _str_contains_pd, _str_contains_pl, needs_strings=True),
        Op("str_replace", _str_replace_pd, _str_replace_pl, needs_strings=True),
        Op("rolling_mean", _rolling_mean_pd, _rolling_mean_pl),
    ]
}


def run_op(op: Op, backend_name: str, df, right=None):
    fn = op.fn_polars if backend_name == "polars" else op.fn_pandas_like
    if op.needs_join_table:
        return fn(df, right)
    return fn(df)
