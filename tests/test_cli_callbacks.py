"""Tests that verify the CLI wires all critical scheduler callbacks.

R10 wiring regression: paper/live commands only wired intraday_loop;
nightly_scan, premarket_verify, and eod_close had no callbacks.
"""

from __future__ import annotations

from datetime import UTC, timedelta
from pathlib import Path

import pandas as pd

from squeeze_hunter.cli import _build_runtime_callbacks
from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.runtime import RuntimeContext


def _seed(cache: ParquetCache) -> None:
    from datetime import datetime

    base = datetime(2026, 5, 14, tzinfo=UTC)
    rows = [
        {
            "ticker": "GME",
            "ts": base + timedelta(days=i),
            "open": 18.0,
            "high": 18.5,
            "low": 17.5,
            "close": 18.0,
            "volume": 1_000_000,
        }
        for i in range(30)
    ]
    cache.write_partition("bars", "GME", pd.DataFrame(rows))
    cache.write_partition(
        "short_interest",
        "all",
        pd.DataFrame(
            columns=[
                "ticker",
                "settlement_date",
                "si_shares",
                "si_pct_float",
                "avg_daily_volume_20d",
            ]
        ),
    )
    cache.write_partition(
        "earnings",
        "all",
        pd.DataFrame(columns=["ticker", "report_at", "actual_eps", "estimate_eps"]),
    )


def test_runtime_callbacks_wires_critical_jobs(tmp_path: Path) -> None:
    """_build_runtime_callbacks must provide callables for the four Phase 3 jobs."""
    cache = ParquetCache(root=tmp_path)
    _seed(cache)
    rc = RuntimeContext(
        cache=cache,
        settings=Settings(),
        tickers=["GME"],
        mode="sim",
    )
    cbs = _build_runtime_callbacks(rc)

    # Phase 3: critical callbacks must be present and callable
    assert "intraday_loop" in cbs
    assert callable(cbs["intraday_loop"])

    assert "nightly_scan" in cbs
    assert callable(cbs["nightly_scan"])

    assert "eod_close" in cbs
    assert callable(cbs["eod_close"])

    assert "premarket_verify" in cbs
    assert callable(cbs["premarket_verify"])

    # Phase 4 follow-ups: explicitly None with comments in source
    assert cbs.get("ingest_eod") is None
    assert cbs.get("premarket_data") is None
    assert cbs.get("moc_decision") is None


def test_runtime_callbacks_covers_all_scheduler_job_ids(tmp_path: Path) -> None:
    """_build_runtime_callbacks must return a key for every job the scheduler knows about."""
    from squeeze_hunter.scheduler import list_job_specs

    cache = ParquetCache(root=tmp_path)
    _seed(cache)
    rc = RuntimeContext(
        cache=cache,
        settings=Settings(),
        tickers=["GME"],
        mode="sim",
    )
    cbs = _build_runtime_callbacks(rc)
    job_ids = {spec["id"] for spec in list_job_specs()}
    # Every known scheduler job must have an explicit entry (even if the value is None)
    assert job_ids.issubset(set(cbs.keys()))
