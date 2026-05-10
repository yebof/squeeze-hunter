from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from squeeze_hunter.data.cache import ParquetCache


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
