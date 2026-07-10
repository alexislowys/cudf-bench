"""Correctness guard: pandas and polars must agree on every op.

A backend that returns fast-but-wrong answers would poison every benchmark
number downstream, so equivalence is tested before speed is ever measured.
"""

import pandas as pd
import polars as pl
import pytest

from benchmark.datagen import make_join_tables
from benchmark.ops import OPS, run_op


@pytest.fixture(scope="module")
def tables():
    left, right = make_join_tables(rows=3_000, n_keys=50, skew=1.1, str_cols=1, seed=3)
    return left, right


def to_pandas(result):
    if isinstance(result, (pl.DataFrame, pl.Series)):
        return result.to_pandas()
    return result


def normalize(result) -> pd.DataFrame:
    df = to_pandas(result)
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.reset_index() if df.index.name else df.reset_index(drop=True)
    df = df.sort_values(list(df.columns)).reset_index(drop=True)
    return df


@pytest.mark.parametrize("op_name", sorted(OPS))
def test_pandas_polars_equivalent(op_name, tables):
    left_pd, right_pd = tables
    op = OPS[op_name]

    res_pandas = run_op(op, "pandas", left_pd, right_pd)
    res_polars = run_op(
        op, "polars", pl.from_pandas(left_pd), pl.from_pandas(right_pd)
    )

    a = normalize(res_pandas)
    b = normalize(res_polars)
    assert len(a) == len(b), f"{op_name}: row counts differ ({len(a)} vs {len(b)})"
    # align column names positionally (polars names aggregates differently)
    b.columns = a.columns
    pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=False, atol=1e-9)
