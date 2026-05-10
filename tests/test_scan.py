from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.data.providers.backtest import BacktestProvider, Clock
from squeeze_hunter.scan import run_scan


def _seed_cache(cache: ParquetCache) -> None:
    base = datetime(2024, 3, 13, tzinfo=UTC)
    rows_gme = []
    rows_aapl = []
    for i in range(62):
        ts = base + pd.Timedelta(days=i)
        rows_gme.append(
            {
                "ticker": "GME",
                "ts": ts,
                "open": 18.0,
                "high": 18.1,
                "low": 17.9,
                "close": 18.0,
                "volume": 1_000_000,
            }
        )
        rows_aapl.append(
            {
                "ticker": "AAPL",
                "ts": ts,
                "open": 200.0,
                "high": 200.5,
                "low": 199.5,
                "close": 200.0,
                "volume": 50_000_000,
            }
        )
    rows_gme[-1]["close"] = 22.0
    rows_gme[-1]["high"] = 22.5
    rows_gme[-1]["volume"] = 8_000_000
    cache.write_partition("bars", "GME", pd.DataFrame(rows_gme))
    cache.write_partition("bars", "AAPL", pd.DataFrame(rows_aapl))
    cache.write_partition(
        "short_interest",
        "all",
        pd.DataFrame(
            [
                {
                    "ticker": "GME",
                    "settlement_date": date(2024, 4, 30),
                    "si_shares": 5_000_000,
                    "si_pct_float": 0.30,
                    "avg_daily_volume_20d": 1_000_000,
                },
                {
                    "ticker": "AAPL",
                    "settlement_date": date(2024, 4, 30),
                    "si_shares": 50_000_000,
                    "si_pct_float": 0.005,
                    "avg_daily_volume_20d": 80_000_000,
                },
            ]
        ),
    )
    cache.write_partition(
        "earnings",
        "all",
        pd.DataFrame(columns=["ticker", "report_at", "actual_eps", "estimate_eps"]),
    )
    cache.write_partition(
        "sentiment",
        "2024-05-13",
        pd.DataFrame(
            [
                {
                    "ticker": "GME",
                    "subreddit": "wallstreetbets",
                    "count_24h": 400,
                    "baseline_30d_mean": 10.0,
                    "baseline_30d_std": 20.0,
                },
                {
                    "ticker": "AAPL",
                    "subreddit": "wallstreetbets",
                    "count_24h": 5,
                    "baseline_30d_mean": 10.0,
                    "baseline_30d_std": 20.0,
                },
            ]
        ),
    )


@pytest.mark.asyncio
async def test_scan_ranks_gme_above_aapl(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path)
    _seed_cache(cache)
    clock = Clock(now=datetime(2024, 5, 13, 23, 59, tzinfo=UTC))
    provider = BacktestProvider(cache=cache, clock=clock)
    settings = Settings()
    settings.score.weights = {
        "f1_si_pct": 2.0,
        "f2_days_to_cover": 1.0,
        "f3_earnings_reaction": 2.0,
        "f4_wsb_mention": 1.5,
        "f5_call_oi_velocity": 1.5,
        "f6_bollinger_breakout": 1.0,
        "f7_volume_spike": 1.0,
    }
    ranked = await run_scan(["GME", "AAPL"], provider, clock.now, settings)
    by_t = ranked.set_index("ticker")
    assert by_t.loc["GME", "score"] > by_t.loc["AAPL", "score"]
    assert "setup_type" in ranked.columns
