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


# R9.6 / R10.1: per-ticker primaryExchange disambiguator. Stock("X","SMART","USD")
# without primaryExchange is ambiguous when more than one contract on TWS shares
# the symbol (e.g., a delisted-then-reused ticker like BBBY). When we KNOW the
# listing exchange, supplying primaryExchange forces the right contract.
#
# IMPORTANT (R10.1): supplying the WRONG primaryExchange (e.g., "NASDAQ" for a
# NYSE-listed name) makes qualifyContractsAsync return zero matches — orders
# fail to submit. So we DO NOT set a hardcoded module-wide default. Tickers
# not in the defaults map (or per-instance override map) fall back to plain
# Stock(t,"SMART","USD") and rely on SMART's auto-disambiguation. That matches
# the pre-R9 behavior for the vast majority of unambiguous symbols and only
# adds explicit hints where we know the listing.
#
# Operators trading names with known ambiguity should populate
# `primary_exchange_overrides` at IBKRBroker construction (or extend the
# module-level map below) — including the explicit value "" to force SMART
# auto-disambiguation when overriding a default would otherwise mis-route.
_PRIMARY_EXCHANGE_DEFAULTS: dict[str, str] = {
    # Known NYSE-listed squeeze candidates (per R9.6 universe survey).
    "GME": "NYSE",
    "AMC": "NYSE",
    "RBLX": "NYSE",
    # Known NASDAQ-listed names with reused/recycled-symbol ambiguity history
    # — explicit hint avoids matching a delisted shadow contract.
    "BBBY": "NASDAQ",
    "BYND": "NASDAQ",
    "HTZ": "NASDAQ",
    "KOSS": "NASDAQ",
    "COIN": "NASDAQ",
    "DJT": "NASDAQ",
    "HOOD": "NASDAQ",
}


@dataclass
class IBKRBroker:
    name: str = "ibkr"
    host: str = field(default_factory=_ibkr_default_host)
    port: int = field(default_factory=_ibkr_default_port)
    client_id: int = field(default_factory=_ibkr_default_client_id)
    account: str = field(default_factory=_ibkr_default_account)
    # R9.6 / R10.1: optional per-instance override map. Takes precedence over
    # _PRIMARY_EXCHANGE_DEFAULTS. Set to "" to FORCE SMART auto-disambiguation
    # for a ticker that would otherwise pick up a default value.
    primary_exchange_overrides: dict[str, str] = field(default_factory=dict)

    def __post_init__(self: IBKRBroker) -> None:
        self._ib = IB()

    def _make_stock(self: IBKRBroker, ticker: str) -> Stock:
        """R9.6 / R10.1: build a Stock contract, attaching primaryExchange ONLY
        when we have an explicit value for this ticker. Tickers without an
        entry get plain `Stock(t,"SMART","USD")` so SMART can pick the right
        listing — a wrong primaryExchange makes qualifyContractsAsync fail.

        Lookup order: per-instance overrides > module-level defaults > none.
        An explicit empty-string override (e.g., `{"GME": ""}`) forces no
        primaryExchange even if the ticker is in the defaults map.
        """
        if ticker in self.primary_exchange_overrides:
            primary = self.primary_exchange_overrides[ticker]
        else:
            primary = _PRIMARY_EXCHANGE_DEFAULTS.get(ticker, "")
        if primary:
            return Stock(ticker, "SMART", "USD", primaryExchange=primary)
        return Stock(ticker, "SMART", "USD")

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
        contract = self._make_stock(ticker)
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
        contract = self._make_stock(ticker)
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
        contract = self._make_stock(ticker)
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

    async def get_position_qty(
        self: IBKRBroker, ticker: str, *, refresh_timeout_s: float = 5.0
    ) -> int:
        """CDX-P1-3 / CDX2-P1: authoritative broker-side held quantity.

        ib-async's IB.positions() is a CACHED snapshot fed by TWS push
        events; immediately after a fill it can still show the pre-fill
        position. The lifecycle pending-exit reconcile depends on this value
        to decide whether a stale exit already filled — a stale read there
        re-introduces the double-sell/short bug CDX-P1-3 set out to kill. So
        we force a fresh sync with reqPositionsAsync() BEFORE reading
        positions(), bounded by `refresh_timeout_s` so a wedged TWS can't
        block the reconcile forever. On timeout we raise TimeoutError, which
        the lifecycle treats as a transient error and conservatively skips
        the resubmit (no shorting).

        We sum any position rows whose contract symbol matches (a US-equity
        ticker normally has exactly one position row).
        """
        await asyncio.wait_for(self._ib.reqPositionsAsync(), timeout=refresh_timeout_s)
        total = 0
        for pos in self._ib.positions():
            if getattr(pos.contract, "symbol", None) == ticker:
                total += int(pos.position)
        return total

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
        except (ConnectionError, TimeoutError, OSError, RuntimeError) as e:
            # R8.Q-I3: narrow per CLAUDE.md. RuntimeError covers ib-async
            # "not connected" raises. AttributeError must propagate so a
            # broker-impl typo (renamed attr) surfaces as a real bug.
            log.warning(
                "ibkr_account_values_failed",
                err=str(e),
                err_type=type(e).__name__,
            )
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
