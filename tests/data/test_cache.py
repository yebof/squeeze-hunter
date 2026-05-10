from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from squeeze_hunter.data.cache import ParquetCache


def test_parquet_cache_dedup_keys_passed_per_call(tmp_path: Path) -> None:
    """C1_minor: dedup_keys should be a per-call parameter, not state. The
    cache instance can be reused across domains without dedup leakage.
    """
    cache = ParquetCache(root=tmp_path)
    df1 = pd.DataFrame({"ticker": ["A"], "ts": [datetime(2024, 5, 13, tzinfo=UTC)], "v": [1]})
    df2 = pd.DataFrame({"ticker": ["A"], "ts": [datetime(2024, 5, 13, tzinfo=UTC)], "v": [2]})
    # First write uses (ticker, ts) dedup
    cache.write_partition("d1", "k", df1, dedup_keys=["ticker", "ts"])
    cache.append_partition("d1", "k", df2, dedup_keys=["ticker", "ts"])
    out = cache.read_partition("d1", "k")
    assert len(out) == 1
    assert out["v"].iloc[0] == 2

    # Second write to a different domain WITHOUT dedup — both rows kept
    cache.write_partition("d2", "k", df1, dedup_keys=None)
    cache.append_partition("d2", "k", df2, dedup_keys=None)
    out2 = cache.read_partition("d2", "k")
    assert len(out2) == 2


def test_parquet_cache_roundtrip(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path)
    df = pd.DataFrame(
        {
            "ticker": ["GME", "AMC"],
            "ts": [datetime(2024, 5, 13, tzinfo=UTC), datetime(2024, 5, 13, tzinfo=UTC)],
            "close": [18.0, 4.0],
        }
    )
    cache.write_partition(domain="bars", partition_key="2024-05-13", df=df)
    out = cache.read_partition(domain="bars", partition_key="2024-05-13")
    assert len(out) == 2
    assert set(out["ticker"]) == {"GME", "AMC"}


def test_parquet_cache_dedup_on_append(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path, dedup_keys=["ticker", "ts"])
    df1 = pd.DataFrame(
        {"ticker": ["GME"], "ts": [datetime(2024, 5, 13, tzinfo=UTC)], "close": [18.0]}
    )
    df2 = pd.DataFrame(
        {"ticker": ["GME"], "ts": [datetime(2024, 5, 13, tzinfo=UTC)], "close": [18.5]}
    )
    cache.write_partition("bars", "2024-05-13", df1)
    cache.append_partition("bars", "2024-05-13", df2)
    out = cache.read_partition("bars", "2024-05-13")
    assert len(out) == 1
    assert out["close"].iloc[0] == 18.5  # latest wins
