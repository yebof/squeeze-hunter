"""SimulatorBroker must satisfy IBroker Protocol semantically.

These tests exist because Phase 3 review (C4) found that SimulatorBroker was
missing several IBroker methods, and its submit_buy/submit_sell signatures
diverged from the Protocol. The result: in sim-mode runtime, manage_positions
silently swallowed the AttributeError, leaving stops un-evaluated.
"""

from datetime import UTC, datetime

import pytest

from squeeze_hunter.backtest.cost_model import StockCostModel
from squeeze_hunter.broker.base import BrokerHealth, BrokerOrder, IBroker, Quote
from squeeze_hunter.broker.simulator import SimulatorBroker


def _broker() -> SimulatorBroker:
    return SimulatorBroker(initial_cash=100_000.0, cost_model=StockCostModel())


def test_simulator_satisfies_ibroker_protocol() -> None:
    b = _broker()
    # runtime_checkable Protocol: isinstance returns True if duck typing matches
    assert isinstance(b, IBroker)


def test_simulator_has_name_attribute() -> None:
    b = _broker()
    assert hasattr(b, "name")
    assert isinstance(b.name, str)
    assert len(b.name) > 0


@pytest.mark.asyncio
async def test_simulator_connect_disconnect_noop() -> None:
    b = _broker()
    await b.connect()
    await b.disconnect()


@pytest.mark.asyncio
async def test_simulator_health_reports_connected() -> None:
    b = _broker()
    h = await b.health()
    assert isinstance(h, BrokerHealth)
    assert h.connected is True
    assert h.account == "simulator" or h.account.startswith("sim")


@pytest.mark.asyncio
async def test_simulator_fetch_quote_uses_last_known_or_avg_price() -> None:
    b = _broker()
    # Open position, then fetch quote — should reflect entered price
    await b.submit_buy(
        ticker="GME",
        qty=100,
        limit_price=18.0,
        ts=datetime(2026, 5, 14, 14, 0, tzinfo=UTC),
    )
    q = await b.fetch_quote("GME")
    assert isinstance(q, Quote)
    assert q.ticker == "GME"
    assert q.last > 0  # not zero — sim should use the position's avg price
    assert q.bid > 0
    assert q.ask > 0


@pytest.mark.asyncio
async def test_simulator_fetch_quote_unknown_ticker_returns_zero_quote() -> None:
    """If we have no position and no marked price, return a zero quote (caller can detect)."""
    b = _broker()
    q = await b.fetch_quote("UNKNOWN")
    # Either zero (caller checks) or raise — both acceptable. But it must NOT
    # AttributeError. The original C4 bug was AttributeError silently swallowed.
    assert isinstance(q, Quote)
    assert q.ticker == "UNKNOWN"


@pytest.mark.asyncio
async def test_simulator_submit_buy_uses_limit_price_keyword() -> None:
    """OMS calls broker.submit_buy(ticker=..., qty=..., limit_price=..., ts=...).

    The original SimulatorBroker took `reference_price` not `limit_price`,
    causing TypeError when called via the Protocol contract.
    """
    b = _broker()
    order = await b.submit_buy(
        ticker="GME",
        qty=100,
        limit_price=18.5,
        ts=datetime(2026, 5, 14, 14, 0, tzinfo=UTC),
    )
    assert isinstance(order, BrokerOrder)
    assert order.broker_order_id  # non-empty
    assert order.ticker == "GME"
    assert order.side == "buy"
    assert order.qty == 100
    assert order.status == "filled"  # simulator fills immediately
    assert order.filled_qty == 100
    assert order.avg_fill_price is not None
    # Slippage applied: actual fill > limit (buy)
    assert order.avg_fill_price >= 18.5


@pytest.mark.asyncio
async def test_simulator_submit_sell_uses_limit_price_keyword() -> None:
    b = _broker()
    await b.submit_buy(
        ticker="GME",
        qty=100,
        limit_price=18.0,
        ts=datetime(2026, 5, 14, 14, 0, tzinfo=UTC),
    )
    order = await b.submit_sell(
        ticker="GME",
        qty=100,
        limit_price=20.0,
        ts=datetime(2026, 5, 17, 14, 0, tzinfo=UTC),
    )
    assert isinstance(order, BrokerOrder)
    assert order.side == "sell"
    assert order.qty == 100
    assert order.status == "filled"


@pytest.mark.asyncio
async def test_simulator_submit_with_none_limit_price_uses_market_semantics() -> None:
    """limit_price=None should mean market order. Use a reasonable reference for fill price."""
    b = _broker()
    # First establish a position so avg_price is known for the fallback
    await b.submit_buy(
        ticker="GME",
        qty=100,
        limit_price=18.0,
        ts=datetime(2026, 5, 14, 14, 0, tzinfo=UTC),
    )
    # Now sell with limit_price=None (market semantics)
    order = await b.submit_sell(
        ticker="GME",
        qty=100,
        limit_price=None,
        ts=datetime(2026, 5, 14, 14, 1, tzinfo=UTC),
    )
    # Even with None limit, fill price should be valid (>0) — not crash
    assert order.avg_fill_price is not None
    assert order.avg_fill_price > 0


@pytest.mark.asyncio
async def test_simulator_get_open_orders_returns_empty_list() -> None:
    """Sim broker fills immediately, so no open orders."""
    b = _broker()
    await b.submit_buy(
        ticker="GME",
        qty=100,
        limit_price=18.0,
        ts=datetime(2026, 5, 14, 14, 0, tzinfo=UTC),
    )
    opens = await b.get_open_orders()
    assert opens == []


@pytest.mark.asyncio
async def test_simulator_cancel_unknown_order_returns_false() -> None:
    b = _broker()
    cancelled = await b.cancel_order("nonexistent-id")
    assert cancelled is False


@pytest.mark.asyncio
async def test_simulator_lifecycle_can_evaluate_stops_without_attribute_error() -> None:
    """Regression test for the original C4 symptom: lifecycle's manage_positions
    silently swallowed the AttributeError raised when fetch_quote was missing.
    """
    from squeeze_hunter.execution.lifecycle import LifecycleState, manage_positions

    b = _broker()
    # Open position so lifecycle has something to manage
    await b.submit_buy(
        ticker="GME",
        qty=100,
        limit_price=100.0,
        ts=datetime(2026, 5, 14, 14, 0, tzinfo=UTC),
    )
    state = LifecycleState(
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
    # Should NOT raise; lifecycle now evaluates stops normally.
    out = await manage_positions(state, b, datetime(2026, 5, 14, 14, 1, tzinfo=UTC))
    # Position still held (no adverse movement)
    assert "GME" in out.positions
