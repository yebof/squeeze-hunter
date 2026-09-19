"""P3 — the order state machine (design spec §6)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from squeeze_hunter.broker.base import BrokerOrder
from squeeze_hunter.execution.orders import (
    InvalidTransitionError,
    OrderRecord,
    OrderState,
    OrderTracker,
    state_from_broker_order,
)

_T0 = datetime(2026, 5, 14, 14, 0, tzinfo=UTC)


def _bo(status: str, filled: int = 0, price: float | None = None, qty: int = 100) -> BrokerOrder:
    return BrokerOrder(
        broker_order_id="o-1",
        ticker="GME",
        side="buy",
        qty=qty,
        limit_price=20.0,
        status=status,
        filled_qty=filled,
        avg_fill_price=price,
    )


def test_status_mapping_from_broker_orders() -> None:
    assert state_from_broker_order(_bo("pending")) is OrderState.PENDING
    assert state_from_broker_order(_bo("routed")) is OrderState.ROUTED
    assert state_from_broker_order(_bo("routed", filled=10)) is OrderState.PARTIAL
    assert state_from_broker_order(_bo("partial", filled=10)) is OrderState.PARTIAL
    assert state_from_broker_order(_bo("filled", filled=100, price=20.0)) is OrderState.FILLED
    assert state_from_broker_order(_bo("cancelled")) is OrderState.CANCELLED
    assert state_from_broker_order(_bo("rejected")) is OrderState.REJECTED
    assert state_from_broker_order(_bo("expired")) is OrderState.EXPIRED


def test_lifecycle_pending_routed_partial_filled() -> None:
    rec = OrderRecord.from_broker_order(_bo("pending"), purpose="entry", now=_T0)
    assert rec.state is OrderState.PENDING
    assert not rec.is_terminal
    rec = rec.apply(_bo("routed"), now=_T0)
    assert rec.state is OrderState.ROUTED
    rec = rec.apply(_bo("routed", filled=40, price=20.01), now=_T0)
    assert rec.state is OrderState.PARTIAL
    assert rec.filled_qty == 40
    rec = rec.apply(_bo("filled", filled=100, price=20.02), now=_T0)
    assert rec.state is OrderState.FILLED
    assert rec.is_terminal
    assert rec.avg_fill_price == 20.02


def test_terminal_states_are_final() -> None:
    rec = OrderRecord.from_broker_order(_bo("filled", filled=100, price=20.0), "entry", _T0)
    with pytest.raises(InvalidTransitionError):
        rec.apply(_bo("routed"), now=_T0)
    cancelled = OrderRecord.from_broker_order(_bo("cancelled", filled=30, price=20.0), "exit", _T0)
    assert cancelled.state is OrderState.CANCELLED
    assert cancelled.filled_qty == 30  # a partial fill survives the cancel


def test_tracker_roundtrips_through_a_snapshot() -> None:
    tracker = OrderTracker()
    tracker.add(OrderRecord.from_broker_order(_bo("pending"), "entry", _T0, meta={"score": 9.0}))
    tracker.update(_bo("routed", filled=10, price=20.0), now=_T0)
    assert [r.order_id for r in tracker.open()] == ["o-1"]
    restored = OrderTracker.from_snapshot(tracker.to_snapshot())
    rec = restored.get("o-1")
    assert rec is not None
    assert rec.state is OrderState.PARTIAL
    assert rec.meta == {"score": 9.0}
    assert rec.submitted_at == _T0
    restored.update(_bo("filled", filled=100, price=20.0), now=_T0)
    assert restored.open() == []
    assert restored.get("o-1").state is OrderState.FILLED  # type: ignore[union-attr]
