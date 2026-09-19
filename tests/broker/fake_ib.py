# ruff: noqa: N802, N803  (names mirror the ib_async API)
"""A fake `ib_async.IB` that reproduces the semantics the MagicMock-based
tests hid for twelve review rounds (P3):

- `reqAccountUpdates()` is BLOCKING and raises inside a running loop; only
  `reqAccountUpdatesAsync()` works.
- `placeOrder()` returns a Trade in `PendingSubmit`; fills arrive later
  (`advance()` / `fill()`), never on the placing call.
- `cancelOrder()` only moves the order to `PendingCancel` — still open and
  fillable until `advance()` acknowledges it as `Cancelled`.
- `reqMktData()` returns ONE cached Ticker per contract; stale values stay
  finite, so freshness must come from `Ticker.time`.
- `openTrades()` is "not in DoneStates" (PendingCancel and ValidationError
  are open); `trades()` includes done ones.
- `MarketOrder.lmtPrice` is `UNSET_DOUBLE`, not None.

Only the surface `IBKRBroker` touches is implemented.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

UNSET_DOUBLE = 1.7976931348623157e308
DONE_STATES = frozenset({"Filled", "Cancelled", "ApiCancelled", "Inactive"})


@dataclass
class FakeTicker:
    contract: Any
    bid: float = math.nan
    ask: float = math.nan
    last: float = math.nan
    close: float = math.nan
    time: datetime | None = None


@dataclass
class FakeIB:
    accounts: list[str] = field(default_factory=lambda: ["DU111"])
    net_liquidation: float = 100_000.0
    cancel_latency: int = 1  # advance() calls before PendingCancel → Cancelled
    connected: bool = False
    _tickers: dict[int, FakeTicker] = field(default_factory=dict)
    _trades: list[Any] = field(default_factory=list)
    _positions: list[Any] = field(default_factory=list)
    _next_order_id: int = 1
    _pending_cancels: dict[int, int] = field(default_factory=dict)
    _account_updates_subscribed: bool = False
    placed: list[Any] = field(default_factory=list)

    # ---------------------------------------------------------------- session
    def __post_init__(self) -> None:
        self.client = SimpleNamespace(serverVersion=lambda: 176)

    async def connectAsync(self, host: str, port: int, clientId: int = 1, **_: Any) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def isConnected(self) -> bool:
        return self.connected

    def managedAccounts(self) -> list[str]:
        return list(self.accounts)

    def reqAccountUpdates(self, account: str = "") -> None:
        # ib_async: util.run(...) → loop.run_until_complete() inside a running loop.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self._account_updates_subscribed = True
            return
        raise RuntimeError("This event loop is already running")

    async def reqAccountUpdatesAsync(self, account: str = "") -> None:
        self._account_updates_subscribed = True

    def accountValues(self, account: str = "") -> list[Any]:
        if not self._account_updates_subscribed:
            return []
        return [
            SimpleNamespace(
                tag="NetLiquidation",
                value=str(self.net_liquidation),
                currency="USD",
                account=self.accounts[0],
            )
        ]

    # ---------------------------------------------------------------- market data
    async def qualifyContractsAsync(self, *contracts: Any) -> list[Any]:
        for c in contracts:
            c.conId = abs(hash(c.symbol)) % 1_000_000
        return list(contracts)

    def reqMktData(self, contract: Any, *_: Any, **__: Any) -> FakeTicker:
        key = int(getattr(contract, "conId", 0) or abs(hash(contract.symbol)) % 1_000_000)
        return self._tickers.setdefault(key, FakeTicker(contract=contract))

    def set_quote(
        self,
        symbol: str,
        *,
        bid: float,
        ask: float,
        last: float,
        at: datetime | None = None,
    ) -> None:
        key = abs(hash(symbol)) % 1_000_000
        t = self._tickers.setdefault(key, FakeTicker(contract=SimpleNamespace(symbol=symbol)))
        t.bid, t.ask, t.last, t.close = bid, ask, last, last
        t.time = at or datetime.now(UTC)

    # ---------------------------------------------------------------- orders
    def placeOrder(self, contract: Any, order: Any) -> Any:
        order.orderId = self._next_order_id
        self._next_order_id += 1
        if not hasattr(order, "lmtPrice") or order.lmtPrice is None:
            order.lmtPrice = UNSET_DOUBLE
        trade = SimpleNamespace(
            contract=contract,
            order=order,
            orderStatus=SimpleNamespace(
                status="PendingSubmit",
                filled=0,
                remaining=int(order.totalQuantity),
                avgFillPrice=0.0,
            ),
        )
        self._trades.append(trade)
        self.placed.append(trade)
        return trade

    def cancelOrder(self, order: Any, *_: Any) -> Any:
        for trade in self._trades:
            if trade.order.orderId == order.orderId:
                if trade.orderStatus.status not in DONE_STATES:
                    trade.orderStatus.status = "PendingCancel"
                    self._pending_cancels[order.orderId] = self.cancel_latency
                return trade
        return None

    def openTrades(self) -> list[Any]:
        return [t for t in self._trades if t.orderStatus.status not in DONE_STATES]

    def trades(self) -> list[Any]:
        return list(self._trades)

    def advance(self) -> None:
        """One TWS round-trip: PendingSubmit → Submitted; acknowledged cancels."""
        for trade in self._trades:
            st = trade.orderStatus
            if st.status == "PendingSubmit":
                st.status = "Submitted"
            elif st.status == "PendingCancel":
                left = self._pending_cancels.get(trade.order.orderId, 0) - 1
                if left <= 0:
                    st.status = "Cancelled"
                    self._pending_cancels.pop(trade.order.orderId, None)
                else:
                    self._pending_cancels[trade.order.orderId] = left

    def fill(self, order_id: int, qty: int, price: float) -> None:
        for trade in self._trades:
            if trade.order.orderId != order_id:
                continue
            st = trade.orderStatus
            if st.status in DONE_STATES:
                raise AssertionError(f"order {order_id} already done: {st.status}")
            prev_qty, prev_px = st.filled, st.avgFillPrice
            st.filled = prev_qty + qty
            st.avgFillPrice = (prev_qty * prev_px + qty * price) / st.filled
            st.remaining = int(trade.order.totalQuantity) - st.filled
            st.status = "Filled" if st.remaining <= 0 else "Submitted"
            self._sync_position(trade.contract.symbol, trade.order.action, qty, price)
            return
        raise AssertionError(f"unknown order {order_id}")

    def reject(self, order_id: int, *, validation: bool = False) -> None:
        for trade in self._trades:
            if trade.order.orderId == order_id:
                trade.orderStatus.status = "ValidationError" if validation else "Inactive"
                return

    # ---------------------------------------------------------------- positions
    def set_position(
        self, symbol: str, qty: int, avg_cost: float, account: str | None = None
    ) -> None:
        self._positions = [p for p in self._positions if p.contract.symbol != symbol]
        if qty:
            self._positions.append(
                SimpleNamespace(
                    contract=SimpleNamespace(symbol=symbol),
                    account=account or self.accounts[0],
                    position=qty,
                    avgCost=avg_cost,
                )
            )

    def _sync_position(self, symbol: str, action: str, qty: int, price: float) -> None:
        current = next((p for p in self._positions if p.contract.symbol == symbol), None)
        held = int(current.position) if current else 0
        cost = float(current.avgCost) if current else 0.0
        if action == "BUY":
            new_qty = held + qty
            new_cost = (held * cost + qty * price) / new_qty if new_qty else 0.0
        else:
            new_qty = held - qty
            new_cost = cost
        self.set_position(symbol, new_qty, new_cost)

    async def reqPositionsAsync(self) -> list[Any]:
        return list(self._positions)

    def positions(self, account: str = "") -> list[Any]:
        return list(self._positions)
