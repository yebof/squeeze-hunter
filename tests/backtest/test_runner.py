from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from squeeze_hunter.backtest.runner import BacktestConfig, run_backtest
from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache

# Background tickers with negligible signals make GME's cross-sectional z-scores
# reach the ±3.0 clip, pushing car_strength and gme_strength past the Mixed (≥3)
# classifier threshold so the gate lets trades through.
_BG = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"]
_ALL_TICKERS = ["GME", "AAPL", *_BG]


def _seed(cache: ParquetCache) -> None:
    base = datetime(2024, 5, 1, tzinfo=UTC)
    # GME: flat for 14 days, big spike on day 14, then drifts down
    gme_bars = []
    for i in range(30):
        ts = base + timedelta(days=i)
        close = 18.0 if i < 14 else (22.0 if i == 14 else 21.0 - (i - 14) * 0.3)
        vol = 1_000_000 if i < 14 else (8_000_000 if i == 14 else 1_500_000)
        gme_bars.append(
            {
                "ticker": "GME",
                "ts": ts,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": vol,
            }
        )
    cache.write_partition("bars", "GME", pd.DataFrame(gme_bars))

    # AAPL: steady, high volume
    aapl_bars = [
        {
            "ticker": "AAPL",
            "ts": base + timedelta(days=i),
            "open": 200.0,
            "high": 200.5,
            "low": 199.5,
            "close": 200.0,
            "volume": 50_000_000,
        }
        for i in range(30)
    ]
    cache.write_partition("bars", "AAPL", pd.DataFrame(aapl_bars))

    # Background tickers: minimal volume / flat price — zero SI, zero sentiment
    for t in _BG:
        bg_bars = [
            {
                "ticker": t,
                "ts": base + timedelta(days=i),
                "open": 50.0,
                "high": 50.1,
                "low": 49.9,
                "close": 50.0,
                "volume": 100_000,
            }
            for i in range(30)
        ]
        cache.write_partition("bars", t, pd.DataFrame(bg_bars))

    # Short interest: only GME has significant SI; rest are zero to maximise its z-score
    si_rows = [
        {
            "ticker": "GME",
            "settlement_date": date(2024, 4, 30),
            "si_shares": 5_000_000,
            "si_pct_float": 0.30,
            "avg_daily_volume_20d": 1_000_000,
        }
    ]
    for t in [*_BG, "AAPL"]:
        si_rows.append(
            {
                "ticker": t,
                "settlement_date": date(2024, 4, 30),
                "si_shares": 5_000,
                "si_pct_float": 0.01,  # small but non-zero so f1 z-score has meaningful spread
                "avg_daily_volume_20d": 1_000,
            }
        )
    cache.write_partition("short_interest", "all", pd.DataFrame(si_rows))

    cache.write_partition(
        "earnings",
        "all",
        pd.DataFrame(columns=["ticker", "report_at", "actual_eps", "estimate_eps"]),
    )

    for i in range(30):
        d = (base + timedelta(days=i)).date().isoformat()
        sent = [
            {
                "ticker": "GME",
                "subreddit": "wallstreetbets",
                "count_24h": 400 if i == 14 else 10,
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
        for t in _BG:
            sent.append(
                {
                    "ticker": t,
                    "subreddit": "wallstreetbets",
                    "count_24h": 0,
                    "baseline_30d_mean": 10.0,
                    "baseline_30d_std": 20.0,
                }
            )
        cache.write_partition("sentiment", d, pd.DataFrame(sent))


@pytest.mark.asyncio
async def test_runner_takes_position_and_records_pnl(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path)
    _seed(cache)
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
    cfg = BacktestConfig(
        tickers=_ALL_TICKERS,
        start=datetime(2024, 5, 14, tzinfo=UTC),
        end=datetime(2024, 5, 28, tzinfo=UTC),
        initial_cash=100_000.0,
        score_threshold=3.0,  # synthetic universe produces lower scores than live; production keeps 8.0
    )
    result = await run_backtest(cfg, cache=cache, settings=settings)
    assert result.equity_curve.iloc[-1] != pytest.approx(100_000.0)  # something happened
    assert (result.trade_log["ticker"] == "GME").any()
