"""IBKRBroker — ib-async wrapper. Phase 0 only implements connect / quote / health."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime

from ib_async import IB, LimitOrder, MarketOrder, Stock

from squeeze_hunter.broker.base import BrokerHealth, BrokerOrder, Quote
from squeeze_hunter.logging_setup import get_logger

log = get_logger("broker.ibkr")

_STATUS_MAP = {
    "PendingSubmit": "pending",
    "PendingCancel": "pending",
    "PreSubmitted": "pending",
    "Submitted": "pending",
    "ApiPending": "pending",
    "Filled": "filled",
    "Cancelled": "cancelled",
    "ApiCancelled": "cancelled",
    "Inactive": "rejected",
}


def _translate_status(ibkr_status: str) -> str:
    return _STATUS_MAP.get(ibkr_status, "pending")


def _ibkr_default_host() -> str:
    return os.environ.get("IBKR_HOST", "127.0.0.1")


def _ibkr_default_port() -> int:
    raw = os.environ.get("IBKR_PORT", "7497")
    try:
        return int(raw)
    except ValueError:
        log.warning("ibkr_port_invalid_using_default", value=raw)
        return 7497


def _ibkr_default_client_id() -> int:
    raw = os.environ.get("IBKR_CLIENT_ID", "42")
    try:
        return int(raw)
    except ValueError:
        log.warning("ibkr_client_id_invalid_using_default", value=raw)
        return 42


def _ibkr_default_account() -> str:
    return os.environ.get("IBKR_ACCOUNT", "")


@dataclass
class IBKRBroker:
    name: str = "ibkr"
    host: str = field(default_factory=_ibkr_default_host)
    port: int = field(default_factory=_ibkr_default_port)
    client_id: int = field(default_factory=_ibkr_default_client_id)
    account: str = field(default_factory=_ibkr_default_account)

    def __post_init__(self: IBKRBroker) -> None:
        self._ib = IB()

    async def connect(self: IBKRBroker) -> None:
        log.info("connecting", host=self.host, port=self.port)
        await self._ib.connectAsync(self.host, self.port, clientId=self.client_id)
        # R5.C1: subscribe to account-value push updates so accountValues()
        # is populated. Without this, get_equity_usd always returns None and
        # the drawdown/3-day-loss killswitch arms are dead. In ib-async,
        # calling reqAccountUpdates(account) IS the subscribe call (no
        # explicit subscribe kwarg; "" means default account).
        self._ib.reqAccountUpdates(self.account or "")
        log.info(
            "connected",
            server_version=self._ib.client.serverVersion(),
            account_subscribed=True,
        )

    async def disconnect(self: IBKRBroker) -> None:
        # R7.C7: ib-async is event-loop native. R6.C4 already removed the
        # to_thread wrap from accountValues() for the same reason — disconnect()
        # should match. Calling it from a worker thread can race the loop's
        # pending push tasks. Direct call is the documented usage.
        self._ib.disconnect()

    async def fetch_quote(self: IBKRBroker, ticker: str) -> Quote:
        contract = Stock(ticker, "SMART", "USD")
        await self._ib.qualifyContractsAsync(contract)
        ticker_data = self._ib.reqMktData(contract, "", snapshot=True, regulatorySnapshot=False)
        # Wait for snapshot — ib-async populates fields as updates arrive
        for _ in range(40):
            await asyncio.sleep(0.25)
            if ticker_data.last is not None or ticker_data.bid is not None:
                break
        return Quote(
            ticker=ticker,
            bid=float(ticker_data.bid or 0.0),
            ask=float(ticker_data.ask or 0.0),
            last=float(ticker_data.last or ticker_data.close or 0.0),
            timestamp_ns=time.time_ns(),
        )

    async def health(self: IBKRBroker) -> BrokerHealth:
        return BrokerHealth(
            connected=self._ib.isConnected(),
            last_ping_ms=0,
            account=self.account,
        )

    async def submit_buy(
        self: IBKRBroker,
        ticker: str,
        qty: int,
        limit_price: float | None,
        ts: datetime,
    ) -> BrokerOrder:
        contract = Stock(ticker, "SMART", "USD")
        await self._ib.qualifyContractsAsync(contract)
        order = LimitOrder("BUY", qty, limit_price) if limit_price else MarketOrder("BUY", qty)
        trade = self._ib.placeOrder(contract, order)
        log.info(
            "order_submitted",
            ticker=ticker,
            side="buy",
            qty=qty,
            limit=limit_price,
            broker_order_id=trade.order.orderId,
        )
        return BrokerOrder(
            broker_order_id=str(trade.order.orderId),
            ticker=ticker,
            side="buy",
            qty=qty,
            limit_price=limit_price,
            status=_translate_status(trade.orderStatus.status),
        )

    async def submit_sell(
        self: IBKRBroker,
        ticker: str,
        qty: int,
        limit_price: float | None,
        ts: datetime,
    ) -> BrokerOrder:
        contract = Stock(ticker, "SMART", "USD")
        await self._ib.qualifyContractsAsync(contract)
        order = LimitOrder("SELL", qty, limit_price) if limit_price else MarketOrder("SELL", qty)
        trade = self._ib.placeOrder(contract, order)
        log.info(
            "order_submitted",
            ticker=ticker,
            side="sell",
            qty=qty,
            limit=limit_price,
            broker_order_id=trade.order.orderId,
        )
        return BrokerOrder(
            broker_order_id=str(trade.order.orderId),
            ticker=ticker,
            side="sell",
            qty=qty,
            limit_price=limit_price,
            status=_translate_status(trade.orderStatus.status),
        )

    async def cancel_order(self: IBKRBroker, broker_order_id: str) -> bool:
        for trade in self._ib.openTrades():
            if str(trade.order.orderId) == broker_order_id:
                self._ib.cancelOrder(trade.order)
                return True
        return False

    async def get_open_orders(self: IBKRBroker) -> list[BrokerOrder]:
        out = []
        for trade in self._ib.openTrades():
            contract = trade.contract
            order = trade.order
            st = trade.orderStatus
            out.append(
                BrokerOrder(
                    broker_order_id=str(order.orderId),
                    ticker=getattr(contract, "symbol", ""),
                    side="buy" if order.action == "BUY" else "sell",
                    qty=int(order.totalQuantity),
                    limit_price=getattr(order, "lmtPrice", None) or None,
                    status=_translate_status(st.status),
                    filled_qty=int(st.filled or 0),
                    avg_fill_price=float(st.avgFillPrice) if st.avgFillPrice else None,
                )
            )
        return out

    async def get_equity_usd(self: IBKRBroker) -> float | None:
        """R4.1: pull NetLiquidation (NAV) from IB account values.

        R5.C1: depends on reqAccountUpdates(subscribe=True) being called in
        connect(). Without that subscription the cached accountValues() list
        is empty forever.

        Returns None if the value isn't available yet (just-connected, account
        snapshot not received). Caller treats None as 'skip equity recording
        this tick' so the killswitch doesn't trip on a zero.
        """
        try:
            # Idempotent re-subscribe in case connect() was bypassed (e.g.,
            # the broker was passed pre-constructed). Calling
            # reqAccountUpdates is the subscribe call in ib-async.
            self._ib.reqAccountUpdates(self.account or "")
            # R6.C4: ib-async maintains the accountValues list via push events
            # on the asyncio event loop. Reading it from a thread (via
            # to_thread) races with those updates and can produce an
            # inconsistent partial view. accountValues() is a cheap in-memory
            # snapshot read, so call it directly on the event loop.
            values = self._ib.accountValues()
        except Exception as e:
            log.warning("ibkr_account_values_failed", err=str(e))
            return None
        for v in values:
            # NetLiquidation in USD is the broker-side NAV. Some accounts also
            # report it in BASE currency; we want USD specifically for our
            # USD-equity universe.
            if (
                getattr(v, "tag", None) == "NetLiquidation"
                and getattr(v, "currency", None) == "USD"
            ):
                try:
                    return float(v.value)
                except (ValueError, TypeError):
                    return None
        return None
