"""Order state machine (design spec §6) — P3 of the architecture-hardening plan.

    PENDING → ROUTED → PARTIAL → FILLED
            ↘ REJECTED  ↘ CANCELLED  ↘ EXPIRED

`OrderRecord` is the durable view of one broker order; `OrderTracker` holds
the open ones (persisted in the runtime snapshot) so a buy that did not fill
on the submitting tick is settled on a later tick — or after a restart —
instead of being forgotten.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from typing import Any

from squeeze_hunter.broker.base import BrokerOrder


class OrderState(StrEnum):
    PENDING = "pending"  # accepted by the API, not yet at the exchange
    ROUTED = "routed"  # live at the exchange, nothing filled
    PARTIAL = "partial"  # live, some quantity filled
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


TERMINAL_STATES = frozenset(
    {OrderState.FILLED, OrderState.REJECTED, OrderState.CANCELLED, OrderState.EXPIRED}
)

_ALLOWED: dict[OrderState, frozenset[OrderState]] = {
    OrderState.PENDING: frozenset(
        {
            OrderState.PENDING,
            OrderState.ROUTED,
            OrderState.PARTIAL,
            OrderState.FILLED,
            OrderState.REJECTED,
            OrderState.CANCELLED,
            OrderState.EXPIRED,
        }
    ),
    # IBKR can bounce Submitted ↔ PreSubmitted; treat ROUTED → PENDING as a
    # no-op rather than an error.
    OrderState.ROUTED: frozenset(
        {
            OrderState.ROUTED,
            OrderState.PENDING,
            OrderState.PARTIAL,
            OrderState.FILLED,
            OrderState.REJECTED,
            OrderState.CANCELLED,
            OrderState.EXPIRED,
        }
    ),
    OrderState.PARTIAL: frozenset(
        {OrderState.PARTIAL, OrderState.FILLED, OrderState.CANCELLED, OrderState.EXPIRED}
    ),
    OrderState.FILLED: frozenset(),
    OrderState.REJECTED: frozenset(),
    OrderState.CANCELLED: frozenset(),
    OrderState.EXPIRED: frozenset(),
}


class InvalidTransitionError(RuntimeError):
    pass


def state_from_broker_order(bo: BrokerOrder) -> OrderState:
    """Map the broker's status vocabulary onto the state machine."""
    status = bo.status
    filled = int(bo.filled_qty or 0)
    if status == "filled":
        return OrderState.FILLED
    if status == "rejected":
        return OrderState.REJECTED
    if status == "cancelled":
        return OrderState.CANCELLED
    if status == "expired":
        return OrderState.EXPIRED
    if status == "partial" or (status in {"routed", "pending"} and filled > 0):
        return OrderState.PARTIAL
    if status == "routed":
        return OrderState.ROUTED
    return OrderState.PENDING


@dataclass(frozen=True, slots=True)
class OrderRecord:
    order_id: str
    ticker: str
    side: str
    qty: int
    limit_price: float | None
    purpose: str  # "entry" | "exit" | "halve"
    state: OrderState
    filled_qty: int
    avg_fill_price: float | None
    submitted_at: datetime
    updated_at: datetime
    commission_usd: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @classmethod
    def from_broker_order(
        cls,
        bo: BrokerOrder,
        purpose: str,
        now: datetime,
        *,
        meta: dict[str, Any] | None = None,
    ) -> OrderRecord:
        return cls(
            order_id=bo.broker_order_id,
            ticker=bo.ticker,
            side=bo.side,
            qty=int(bo.qty),
            limit_price=bo.limit_price,
            purpose=purpose,
            state=state_from_broker_order(bo),
            filled_qty=int(bo.filled_qty or 0),
            avg_fill_price=bo.avg_fill_price,
            submitted_at=now,
            updated_at=now,
            commission_usd=float(bo.commission_usd or 0.0),
            meta=dict(meta or {}),
        )

    def apply(self, bo: BrokerOrder, *, now: datetime) -> OrderRecord:
        nxt = state_from_broker_order(bo)
        if nxt not in _ALLOWED[self.state]:
            raise InvalidTransitionError(f"{self.order_id}: {self.state.value} -> {nxt.value}")
        if nxt is OrderState.PENDING and self.state is OrderState.ROUTED:
            nxt = OrderState.ROUTED  # bounce: stay routed
        filled = max(self.filled_qty, int(bo.filled_qty or 0))
        return replace(
            self,
            state=nxt,
            filled_qty=filled,
            avg_fill_price=bo.avg_fill_price if bo.avg_fill_price else self.avg_fill_price,
            commission_usd=float(bo.commission_usd or self.commission_usd),
            updated_at=now,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "ticker": self.ticker,
            "side": self.side,
            "qty": self.qty,
            "limit_price": self.limit_price,
            "purpose": self.purpose,
            "state": self.state.value,
            "filled_qty": self.filled_qty,
            "avg_fill_price": self.avg_fill_price,
            "submitted_at": self.submitted_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "commission_usd": self.commission_usd,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OrderRecord:
        return cls(
            order_id=str(d["order_id"]),
            ticker=str(d["ticker"]),
            side=str(d["side"]),
            qty=int(d["qty"]),
            limit_price=d.get("limit_price"),
            purpose=str(d.get("purpose", "entry")),
            state=OrderState(d.get("state", "pending")),
            filled_qty=int(d.get("filled_qty", 0)),
            avg_fill_price=d.get("avg_fill_price"),
            submitted_at=datetime.fromisoformat(d["submitted_at"]),
            updated_at=datetime.fromisoformat(d["updated_at"]),
            commission_usd=float(d.get("commission_usd", 0.0)),
            meta=dict(d.get("meta") or {}),
        )


@dataclass
class OrderTracker:
    """Open orders by id. Terminal records are kept until `forget()` so the
    caller can act on the final fill exactly once."""

    _records: dict[str, OrderRecord] = field(default_factory=dict)

    def add(self, rec: OrderRecord) -> None:
        self._records[rec.order_id] = rec

    def get(self, order_id: str) -> OrderRecord | None:
        return self._records.get(order_id)

    def update(self, bo: BrokerOrder, *, now: datetime) -> OrderRecord | None:
        rec = self._records.get(bo.broker_order_id)
        if rec is None:
            return None
        rec = rec.apply(bo, now=now)
        self._records[rec.order_id] = rec
        return rec

    def forget(self, order_id: str) -> None:
        self._records.pop(order_id, None)

    def open(self) -> list[OrderRecord]:
        return [r for r in self._records.values() if not r.is_terminal]

    def all(self) -> list[OrderRecord]:
        return list(self._records.values())

    def to_snapshot(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self._records.values()]

    @classmethod
    def from_snapshot(cls, rows: list[dict[str, Any]] | None) -> OrderTracker:
        tracker = cls()
        for row in rows or []:
            if isinstance(row, dict):
                tracker.add(OrderRecord.from_dict(row))
        return tracker
