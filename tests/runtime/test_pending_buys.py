"""P3 — entries that do not fill on the submitting tick are tracked, settled
on later ticks, persisted, and cancelled after the entry window."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from squeeze_hunter.backtest.cost_model import StockCostModel
from squeeze_hunter.broker.base import BrokerOrder
from squeeze_hunter.broker.simulator import SimulatorBroker
from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.execution.orders import OrderState
from squeeze_hunter.runtime import RuntimeContext
from squeeze_hunter.store.state import JsonStateStore
from tests.runtime.test_session_clamp import _seed

_PREMARKET = datetime(2026, 6, 10, 12, 0, tzinfo=UTC)  # 08:00 ET
_T1 = datetime(2026, 6, 10, 13, 36, tzinfo=UTC)  # 09:36 ET
_T2 = datetime(2026, 6, 10, 13, 37, tzinfo=UTC)
_LATE = datetime(2026, 6, 10, 14, 30, tzinfo=UTC)  # 10:30 ET, past the entry window


class _SlowFillSimulator(SimulatorBroker):
    """Buys go pending; `settle(order_id, qty)` fills them later."""

    def __init__(self) -> None:
        super().__init__(initial_cash=100_000.0, cost_model=StockCostModel())
        self.pending: dict[str, BrokerOrder] = {}

    async def submit_buy(self, ticker, qty, limit_price, ts, is_open_5min=False):  # type: ignore[override]
        oid = f"buy-{len(self.pending) + 1}"
        order = BrokerOrder(
            broker_order_id=oid,
            ticker=ticker,
            side="buy",
            qty=qty,
            limit_price=limit_price,
            status="pending",
        )
        self.pending[oid] = order
        return order

    async def settle(self, oid: str, qty: int | None = None) -> None:
        o = self.pending[oid]
        fill_qty = qty if qty is not None else o.qty
        real = await SimulatorBroker.submit_buy(self, o.ticker, fill_qty, o.limit_price, _T1)
        status = "filled" if fill_qty == o.qty else "partial"
        self.pending[oid] = BrokerOrder(
            broker_order_id=oid,
            ticker=o.ticker,
            side="buy",
            qty=o.qty,
            limit_price=o.limit_price,
            status=status,
            filled_qty=fill_qty,
            avg_fill_price=real.avg_fill_price,
            commission_usd=real.commission_usd,
        )

    def reject(self, oid: str) -> None:
        o = self.pending[oid]
        self.pending[oid] = BrokerOrder(
            broker_order_id=oid,
            ticker=o.ticker,
            side="buy",
            qty=o.qty,
            limit_price=o.limit_price,
            status="rejected",
        )

    async def get_order(self, order_id: str) -> BrokerOrder | None:  # type: ignore[override]
        return self.pending.get(order_id) or await SimulatorBroker.get_order(self, order_id)

    async def cancel_order(self, order_id: str) -> bool:  # type: ignore[override]
        o = self.pending.get(order_id)
        if o is None:
            return False
        self.pending[order_id] = BrokerOrder(
            broker_order_id=order_id,
            ticker=o.ticker,
            side="buy",
            qty=o.qty,
            limit_price=o.limit_price,
            status="cancelled",
            filled_qty=o.filled_qty,
            avg_fill_price=o.avg_fill_price,
        )
        return True

    async def get_open_orders(self) -> list[BrokerOrder]:  # type: ignore[override]
        return [o for o in self.pending.values() if o.status in {"pending", "routed", "partial"}]


async def _rc(
    tmp_path: Path, broker: SimulatorBroker, store: JsonStateStore | None = None
) -> RuntimeContext:
    cache = ParquetCache(root=tmp_path / "parquet")
    _seed(cache)
    settings = Settings()
    settings.score.weights = {"f6_bollinger_breakout": 1.0, "f7_volume_spike": 1.0}
    settings.execution.auto_enter = True
    settings.data.require_fresh_for_entries = False  # P6 gate has its own tests
    rc = RuntimeContext(
        cache=cache,
        settings=settings,
        tickers=["GME"],
        mode="sim",
        broker=broker,
        state_store=store,
    )
    await rc.setup()
    rc.last_candidates = pd.DataFrame(
        [{"ticker": "GME", "score": 99.0, "setup_type": "CAR", "rank": 1, "as_of": _PREMARKET}]
    )
    await rc.premarket_verify(now=_PREMARKET)
    assert rc.planned_entries
    return rc


@pytest.mark.asyncio
async def test_pending_buy_is_tracked_then_settled_on_a_later_tick(tmp_path: Path) -> None:
    broker = _SlowFillSimulator()
    rc = await _rc(tmp_path, broker)
    await rc.tick(now=_T1)
    assert rc.lifecycle_state.positions == {}
    assert [r.state for r in rc.pending_buys.open()] == [OrderState.PENDING]
    await broker.settle("buy-1")
    await rc.tick(now=_T2)
    meta = rc.lifecycle_state.positions["GME"]
    assert meta["qty"] == broker.pending["buy-1"].qty
    assert meta["entry_price"] == pytest.approx(broker.pending["buy-1"].avg_fill_price)
    assert meta["entry_score"] == 99.0
    assert meta["setup_type"] == "CAR"
    assert rc.pending_buys.open() == []
    assert "GME" in rc.telemetry.position_marks


@pytest.mark.asyncio
async def test_partial_fill_after_the_entry_window_is_cancelled_and_kept(tmp_path: Path) -> None:
    broker = _SlowFillSimulator()
    rc = await _rc(tmp_path, broker)
    await rc.tick(now=_T1)
    full = broker.pending["buy-1"].qty
    await broker.settle("buy-1", qty=max(1, full // 2))
    await rc.tick(now=_LATE)
    assert broker.pending["buy-1"].status == "cancelled"
    meta = rc.lifecycle_state.positions["GME"]
    assert meta["qty"] == max(1, full // 2)
    assert rc.pending_buys.open() == []


@pytest.mark.asyncio
async def test_rejected_buy_is_dropped_without_a_position(tmp_path: Path) -> None:
    broker = _SlowFillSimulator()
    rc = await _rc(tmp_path, broker)
    await rc.tick(now=_T1)
    broker.reject("buy-1")
    await rc.tick(now=_T2)
    assert rc.lifecycle_state.positions == {}
    assert rc.pending_buys.open() == []


@pytest.mark.asyncio
async def test_pending_buys_survive_a_restart(tmp_path: Path) -> None:
    broker = _SlowFillSimulator()
    store = JsonStateStore(tmp_path / "state.json")
    rc1 = await _rc(tmp_path, broker, store)
    await rc1.tick(now=_T1)
    assert rc1.pending_buys.open()

    rc2 = RuntimeContext(
        cache=ParquetCache(root=tmp_path / "parquet"),
        settings=rc1.settings,
        tickers=["GME"],
        mode="sim",
        broker=broker,
        state_store=store,
    )
    await rc2.setup(now=_T1)  # restarted within the entry window
    assert [r.order_id for r in rc2.pending_buys.open()] == ["buy-1"]
    await broker.settle("buy-1")
    await rc2.tick(now=_T2)
    assert "GME" in rc2.lifecycle_state.positions
