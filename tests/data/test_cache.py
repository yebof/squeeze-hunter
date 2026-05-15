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


def test_parquet_cache_rejects_path_traversal_partition_key(tmp_path: Path) -> None:
    """CDX-P2-6 regression: _path joined root/domain/f"{key}.parquet" with no
    validation. A ticker/partition_key containing '../' or '/' could read or
    WRITE parquet outside the cache root (the universe file is operator-edited
    and ticker strings flow straight into partition keys). Domain and
    partition_key must be restricted to a safe charset and reject traversal.
    """
    import pytest

    cache = ParquetCache(root=tmp_path)
    df = pd.DataFrame({"ticker": ["A"], "v": [1]})

    bad_keys = [
        "../escape",
        "../../etc/passwd",
        "a/b",
        "x/../../y",
        "/abs/path",
        "..",
    ]
    for bad in bad_keys:
        with pytest.raises(ValueError, match=r"unsafe|partition|domain"):
            cache.write_partition("bars", bad, df)
        with pytest.raises(ValueError, match=r"unsafe|partition|domain"):
            cache.read_partition("bars", bad)
    # Bad DOMAIN is rejected too.
    with pytest.raises(ValueError, match=r"unsafe|partition|domain"):
        cache.write_partition("../sneaky", "k", df)

    # Legit keys still work: ISO dates (hyphens), ticker__date composites
    # (double underscore), and dotted tickers like BRK.B.
    cache.write_partition("bars", "2024-05-13", df)
    cache.write_partition("options", "AAPL__2024-05-13", df)
    cache.write_partition("bars", "BRK.B", df)
    assert len(cache.read_partition("bars", "2024-05-13")) == 1
    assert len(cache.read_partition("options", "AAPL__2024-05-13")) == 1
    assert len(cache.read_partition("bars", "BRK.B")) == 1


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
