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
        log.info("connected", server_version=self._ib.client.serverVersion())

    async def disconnect(self: IBKRBroker) -> None:
        await asyncio.to_thread(self._ib.disconnect)

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
