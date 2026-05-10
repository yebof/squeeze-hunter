from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from squeeze_hunter.backtest.walk_forward import WalkForwardConfig, run_walk_forward
from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache


def _cfg_kwargs(**overrides):
    """Helper: minimal valid WalkForwardConfig kwargs that the validator accepts."""
    base = {
        "tickers": ["GME"],
        "train_start": datetime(2024, 1, 1, tzinfo=UTC),
        "train_end": datetime(2024, 4, 1, tzinfo=UTC),
        "test_windows": [
            (datetime(2024, 4, 2, tzinfo=UTC), datetime(2024, 5, 1, tzinfo=UTC)),
        ],
        "holdout": (datetime(2024, 5, 2, tzinfo=UTC), datetime(2024, 6, 1, tzinfo=UTC)),
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_walk_forward_rejects_train_end_after_test_start(tmp_path: Path) -> None:
    cfg = WalkForwardConfig(
        **_cfg_kwargs(
            train_end=datetime(2024, 5, 15, tzinfo=UTC),  # after test_windows[0][0]
        )
    )
    cache = ParquetCache(root=tmp_path)

    with pytest.raises(ValueError, match="train_end"):
        await run_walk_forward(cfg, cache=cache, settings=Settings())


@pytest.mark.asyncio
async def test_walk_forward_rejects_overlapping_test_windows(tmp_path: Path) -> None:
    cfg = WalkForwardConfig(
        **_cfg_kwargs(
            test_windows=[
                (datetime(2024, 4, 2, tzinfo=UTC), datetime(2024, 5, 1, tzinfo=UTC)),
                (datetime(2024, 4, 20, tzinfo=UTC), datetime(2024, 5, 30, tzinfo=UTC)),  # overlaps
            ],
        )
    )
    cache = ParquetCache(root=tmp_path)

    with pytest.raises(ValueError, match="test_windows"):
        await run_walk_forward(cfg, cache=cache, settings=Settings())


@pytest.mark.asyncio
async def test_walk_forward_rejects_holdout_before_test_end(tmp_path: Path) -> None:
    cfg = WalkForwardConfig(
        **_cfg_kwargs(
            holdout=(datetime(2024, 4, 15, tzinfo=UTC), datetime(2024, 5, 1, tzinfo=UTC)),
        )
    )
    cache = ParquetCache(root=tmp_path)

    with pytest.raises(ValueError, match="holdout"):
        await run_walk_forward(cfg, cache=cache, settings=Settings())


@pytest.mark.asyncio
async def test_walk_forward_produces_per_window_metrics(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    rows = []
    for i in range(200):
        rows.append(
            {
                "ticker": "GME",
                "ts": base + timedelta(days=i),
                "open": 18.0,
                "high": 18.5,
                "low": 17.8,
                "close": 18.0 + 0.01 * i,
                "volume": 1_000_000,
            }
        )
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
    settings = Settings()
    settings.score.weights = {
        "f1_si_pct": 1.0,
        "f2_days_to_cover": 1.0,
        "f3_earnings_reaction": 1.0,
        "f4_wsb_mention": 1.0,
        "f5_call_oi_velocity": 1.0,
        "f6_bollinger_breakout": 1.0,
        "f7_volume_spike": 1.0,
    }
    cfg = WalkForwardConfig(
        tickers=["GME"],
        train_start=datetime(2024, 1, 1, tzinfo=UTC),
        train_end=datetime(2024, 4, 1, tzinfo=UTC),
        test_windows=[
            (datetime(2024, 4, 2, tzinfo=UTC), datetime(2024, 5, 1, tzinfo=UTC)),
            (datetime(2024, 5, 2, tzinfo=UTC), datetime(2024, 6, 1, tzinfo=UTC)),
        ],
        holdout=(datetime(2024, 6, 2, tzinfo=UTC), datetime(2024, 7, 19, tzinfo=UTC)),
    )
    report = await run_walk_forward(cfg, cache=cache, settings=settings)
    assert "train" in report
    assert len(report["test_windows"]) == 2
    assert "holdout" in report
