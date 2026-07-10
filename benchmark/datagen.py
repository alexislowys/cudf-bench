"""Synthetic table generator with controllable size, shape, and skew.

All data is generated as pandas DataFrames (the single source of truth);
backend adapters convert to their native format before timing, so every
backend benchmarks byte-identical input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _key_column(rng: np.random.Generator, rows: int, n_keys: int, skew: float) -> np.ndarray:
    """Integer keys in [0, n_keys).

    skew=0 is uniform. skew>0 draws from a Zipf-like distribution where
    P(key=k) is proportional to 1/(k+1)^skew — at skew ~1.1+ a handful of
    keys dominate the column, which is the shape that stresses hash-based
    joins and groupbys.
    """
    if skew <= 0:
        return rng.integers(0, n_keys, size=rows, dtype=np.int64)
    ranks = np.arange(1, n_keys + 1, dtype=np.float64)
    p = ranks**-skew
    p /= p.sum()
    return rng.choice(n_keys, size=rows, p=p).astype(np.int64)


def _string_column(rng: np.random.Generator, rows: int, str_len: int) -> np.ndarray:
    """Random lowercase ASCII strings of fixed length, generated vectorized
    (building millions of strings in a Python loop is far too slow)."""
    codes = rng.integers(ord("a"), ord("z") + 1, size=(rows, str_len), dtype=np.uint8)
    return codes.view(f"S{str_len}").ravel().astype(str)


def make_table(
    rows: int,
    num_key_cols: int = 1,
    num_val_cols: int = 3,
    str_cols: int = 0,
    n_keys: int | None = None,
    skew: float = 0.0,
    str_len: int = 12,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a synthetic table: key0..keyN int64, val0..valN float64, str0..strN str."""
    rng = np.random.default_rng(seed)
    if n_keys is None:
        n_keys = max(1, min(rows // 10, 100_000))

    data: dict[str, np.ndarray] = {}
    for i in range(num_key_cols):
        data[f"key{i}"] = _key_column(rng, rows, n_keys, skew)
    for i in range(num_val_cols):
        data[f"val{i}"] = rng.random(rows)
    for i in range(str_cols):
        data[f"str{i}"] = _string_column(rng, rows, str_len)
    return pd.DataFrame(data)


def make_join_tables(
    rows: int,
    n_keys: int | None = None,
    skew: float = 0.0,
    str_cols: int = 0,
    str_len: int = 12,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A large fact table plus a small dimension table keyed on key0.

    The dimension table has one row per distinct key, so an inner join
    preserves the fact table's row count — including under heavy skew,
    which is exactly the case that stresses hash joins.
    """
    left = make_table(
        rows, str_cols=str_cols, n_keys=n_keys, skew=skew, str_len=str_len, seed=seed
    )
    rng = np.random.default_rng(seed + 1)
    keys = np.unique(left["key0"].to_numpy())
    right = pd.DataFrame({"key0": keys, "dim_val": rng.random(len(keys))})
    return left, right
