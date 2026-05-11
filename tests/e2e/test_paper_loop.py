from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from squeeze_hunter.backtest.cost_model import StockCostModel
from squeeze_hunter.broker.simulator import SimulatorBroker
from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.execution.lifecycle import LifecycleState
from squeeze_hunter.runtime import RuntimeContext


@pytest.mark.asyncio
async def test_runtime_three_ticks_no_crash(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path)
    base = datetime(2026, 5, 14, tzinfo=UTC)
    bars = [
        {
            "ticker": "GME",
            "ts": base + timedelta(days=i),
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.0,
            "volume": 1_000_000,
        }
        for i in range(30)
    ]
    cache.write_partition("bars", "GME", pd.DataFrame(bars))
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

    sim_broker = SimulatorBroker(initial_cash=100_000.0, cost_model=StockCostModel())
    # Establish a real broker-side position so lifecycle's submit_sell can succeed
    await sim_broker.submit_buy(
        ticker="GME",
        qty=100,
        limit_price=100.0,
        ts=base,
    )
    rc = RuntimeContext(
        cache=cache,
        settings=Settings(),
        tickers=["GME"],
        mode="sim",
        broker=sim_broker,  # type: ignore[arg-type]
    )
    rc.lifecycle_state = LifecycleState(
        positions={
            "GME": {
                "qty": 100,
                "entry_price": 100.0,
                "peak_price": 100.0,
                "entry_score": 10.0,
                "current_score": 10.0,
                "bars_held": 1,
                "setup_type": "CAR",
            }
        }
    )
    await rc.setup()
    # Three ticks at increasing time during the US regular session
    # (Thu 2026-06-11 10:00 ET = 14:00 UTC during EDT). R4.2 fix: previous
    # timestamps were `base + 29 days = 2026-06-12 00:00 UTC = 2026-06-11
    # 20:00 ET`, outside the session — the R3.2 guard early-returned and
    # the test never actually exercised manage_positions.
    in_session_base = datetime(2026, 6, 11, 14, 0, tzinfo=UTC)  # Thu 10:00 ET
    for offset in (0, 60, 120):
        await rc.tick(now=in_session_base + timedelta(seconds=offset))
    await rc.shutdown()
    # Position is still held (price didn't move adversely from entry)
    assert "GME" in rc.lifecycle_state.positions
