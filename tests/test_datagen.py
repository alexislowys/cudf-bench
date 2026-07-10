import numpy as np

from benchmark.datagen import make_join_tables, make_table


def test_shape_and_columns():
    df = make_table(rows=1_000, num_key_cols=2, num_val_cols=3, str_cols=1)
    assert len(df) == 1_000
    assert list(df.columns) == ["key0", "key1", "val0", "val1", "val2", "str0"]
    assert df["key0"].dtype == np.int64
    assert df["val0"].dtype == np.float64
    assert df["str0"].map(len).eq(12).all()


def test_seed_reproducible():
    a = make_table(rows=500, str_cols=1, seed=42)
    b = make_table(rows=500, str_cols=1, seed=42)
    assert a.equals(b)
    c = make_table(rows=500, str_cols=1, seed=43)
    assert not a.equals(c)


def test_skew_concentrates_keys():
    uniform = make_table(rows=50_000, n_keys=1_000, skew=0.0, seed=0)
    skewed = make_table(rows=50_000, n_keys=1_000, skew=1.5, seed=0)
    top_uniform = uniform["key0"].value_counts().iloc[0]
    top_skewed = skewed["key0"].value_counts().iloc[0]
    # under skew the hottest key should dominate; uniform tops out near rows/n_keys
    assert top_skewed > 5 * top_uniform


def test_join_tables_align():
    left, right = make_join_tables(rows=2_000, n_keys=100, skew=1.1, seed=7)
    assert set(left["key0"]).issubset(set(right["key0"]))
    assert right["key0"].is_unique
    # dimension table covers exactly the keys that appear
    assert set(right["key0"]) == set(left["key0"])
