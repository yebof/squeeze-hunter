"""IBKRBroker — ib-async wrapper. Phase 0 only implements connect / quote / health."""

from __future__ import annotations

import asyncio
import math
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ib_async import IB, LimitOrder, MarketOrder, Stock

from squeeze_hunter.broker.base import BrokerHealth, BrokerOrder, PositionSnapshot, Quote
from squeeze_hunter.logging_setup import get_logger

log = get_logger("broker.ibkr")

# P3: the vocabulary of execution/orders.py. "routed" = live at the exchange
# (Submitted / PreSubmitted, and PendingCancel — still fillable until TWS
# acknowledges the cancel).
_STATUS_MAP = {
    "PendingSubmit": "pending",
    "ApiPending": "pending",
    "PreSubmitted": "routed",
    "Submitted": "routed",
    "PendingCancel": "routed",
    "Filled": "filled",
    "Cancelled": "cancelled",
    "ApiCancelled": "cancelled",
    "Inactive": "rejected",
    # Round-13: set by ib_async for warning codes 105/110/165/321…; it is not
    # a DoneState, so an order in this state sits in openTrades() forever.
    # Treating it as "pending" kept a dead exit as the tracked pending order.
    "ValidationError": "rejected",
}

# Round-13: a quote whose Ticker has not been updated for longer than this is
# reported as zeros (halt, market-data farm outage, lost subscription).
_QUOTE_MAX_AGE_S = 120.0


# ib_async leaves an unset limit price at this sentinel (MarketOrder.lmtPrice).
_UNSET_DOUBLE = 1.7976931348623157e308


def _limit_or_none(raw: Any) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value >= _UNSET_DOUBLE / 2 or value <= 0:
        return None
    return value


def _broker_order_from_trade(trade: Any) -> BrokerOrder:
    contract = trade.contract
    order = trade.order
    st = trade.orderStatus
    filled = int(getattr(st, "filled", 0) or 0)
    status = _translate_status(st.status)
    if status == "routed" and filled > 0:
        status = "partial"
    return BrokerOrder(
        broker_order_id=str(order.orderId),
        ticker=getattr(contract, "symbol", ""),
        side="buy" if order.action == "BUY" else "sell",
        qty=int(order.totalQuantity),
        limit_price=_limit_or_none(getattr(order, "lmtPrice", None)),
        status=status,
        filled_qty=filled,
        avg_fill_price=float(st.avgFillPrice) if getattr(st, "avgFillPrice", 0) else None,
    )


def _age_seconds(ts: datetime) -> float:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return (datetime.now(UTC) - ts).total_seconds()


def require_live_port() -> None:
    """Refuse to construct a LIVE broker on the paper port.

    Round-13: `_ibkr_default_port` falls back to 7497 (TWS paper), so
    `live --confirm-real-money` and `emergency-flatten --mode live` with
    IBKR_PORT unset quietly talked to the paper account while reporting
    success. Live must name its port explicitly.
    """
    raw = os.environ.get("IBKR_PORT", "").strip()
    if not raw:
        raise ValueError(
            "live mode requires IBKR_PORT to be set explicitly "
            "(7496 for TWS live, 4001 for IB Gateway live)"
        )
    if raw == "7497":
        raise ValueError("live mode refuses IBKR_PORT=7497: that is the TWS paper port")


def _translate_status(ibkr_status: str) -> str:
    return _STATUS_MAP.get(ibkr_status, "pending")


def _finite_or_zero(x: object) -> float:
    """Coalesce None / NaN / Inf to 0.0.

    R11: ib-async's Ticker initializes bid/ask/last/close to ``float("nan")``,
    not None. Because nan is truthy, the naive ``float(x or 0.0)`` returned nan
    unchanged whenever market data had not arrived — and nan then slips past the
    lifecycle's ``price <= 0.0`` stop guard (``nan <= 0.0`` is False), silently
    disabling the hard/trailing stops. Coalesce explicitly on finiteness.
    """
    if isinstance(x, int | float) and math.isfinite(x):
        return float(x)
    return 0.0


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
        # P2: client order refs placed this session (ref -> broker order id).
        self._submitted_refs: dict[str, str] = {}

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
        # the drawdown/3-day-loss killswitch arms are dead.
        # Round-13: use the ASYNC variant. ib_async's reqAccountUpdates() is
        # the blocking one (loop.run_until_complete) and raised "This event
        # loop is already running" right here — paper/live never started.
        await self._ib.reqAccountUpdatesAsync(self.account or "")
        managed = [str(a) for a in (self._ib.managedAccounts() or [])]
        if self.account and managed and self.account not in managed:
            # Round-13: IBKR_ACCOUNT=DU0000000 copied from .env.example plus
            # the account filter in get_position_qty made every position read
            # as flat, and the lifecycle reconcile popped real exposure.
            self._ib.disconnect()
            raise ValueError(
                f"IBKR_ACCOUNT={self.account!r} is not managed by this login "
                f"(managed accounts: {managed}); set it to one of them or leave it empty"
            )
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
        # Wait for snapshot — ib-async populates fields as updates arrive.
        # R11: a fresh Ticker's fields are float("nan"), not None, so the old
        # `is not None` break was True on the very first iteration and we
        # returned a premature NaN. Break only once a FINITE bid or last has
        # actually arrived.
        for _ in range(40):
            await asyncio.sleep(0.25)
            if math.isfinite(ticker_data.last) or math.isfinite(ticker_data.bid):
                break
        # Round-13: ib_async caches one Ticker per contract and hands the SAME
        # object back on every reqMktData, so yesterday's finite values pass
        # the isfinite check on the first iteration. Reject a quote whose last
        # update is older than the freshness budget (halt, farm outage, lost
        # subscription): zeros are caught by the lifecycle guard and starve
        # the data_stale killswitch arm instead of freezing the stops.
        last_update = getattr(ticker_data, "time", None)
        if last_update is None or (
            isinstance(last_update, datetime) and _age_seconds(last_update) > _QUOTE_MAX_AGE_S
        ):
            log.warning("quote_stale", ticker=ticker, last_update=str(last_update))
            return Quote(ticker=ticker, bid=0.0, ask=0.0, last=0.0, timestamp_ns=time.time_ns())
        # R11: coalesce non-finite values to 0.0 (see _finite_or_zero) so a
        # missing quote becomes 0.0 — caught by the lifecycle zero-price guard —
        # instead of NaN, which slips past it.
        return Quote(
            ticker=ticker,
            bid=_finite_or_zero(ticker_data.bid),
            ask=_finite_or_zero(ticker_data.ask),
            last=_finite_or_zero(ticker_data.last) or _finite_or_zero(ticker_data.close),
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
        return await self._place(ticker, "buy", qty, limit_price, ts)

    async def submit_sell(
        self: IBKRBroker,
        ticker: str,
        qty: int,
        limit_price: float | None,
        ts: datetime,
    ) -> BrokerOrder:
        return await self._place(ticker, "sell", qty, limit_price, ts)

    @staticmethod
    def _client_ref(ticker: str, side: str, qty: int, ts: datetime) -> str:
        """P2: idempotency key for one logical order. A retry of the same
        (ticker, side, qty, second) — e.g. after a timeout whose first attempt
        actually reached TWS — must not place a second order."""
        return f"sh-{side}-{ticker}-{qty}-{ts.astimezone(UTC).strftime('%Y%m%dT%H%M%S')}"

    def _find_open_by_ref(self: IBKRBroker, ref: str):  # noqa: ANN202 (ib_async Trade)
        for trade in self._ib.openTrades():
            if getattr(trade.order, "orderRef", None) == ref:
                return trade
        return None

    async def _place(
        self: IBKRBroker,
        ticker: str,
        side: str,
        qty: int,
        limit_price: float | None,
        ts: datetime,
    ) -> BrokerOrder:
        ref = self._client_ref(ticker, side, qty, ts)
        existing = self._find_open_by_ref(ref)
        if existing is not None or ref in self._submitted_refs:
            log.warning("order_deduped_by_client_ref", ticker=ticker, side=side, ref=ref)
            if existing is not None:
                return _broker_order_from_trade(existing)
            # Placed earlier this session and no longer open: terminal. Report
            # it as pending so the caller reconciles against the position
            # rather than assuming a fill or resubmitting.
            return BrokerOrder(
                broker_order_id=self._submitted_refs[ref],
                ticker=ticker,
                side=side,
                qty=qty,
                limit_price=limit_price,
                status="pending",
            )
        contract = self._make_stock(ticker)
        await self._ib.qualifyContractsAsync(contract)
        action = side.upper()
        order = LimitOrder(action, qty, limit_price) if limit_price else MarketOrder(action, qty)
        order.orderRef = ref
        trade = self._ib.placeOrder(contract, order)
        self._submitted_refs[ref] = str(trade.order.orderId)
        log.info(
            "order_submitted",
            ticker=ticker,
            side=side,
            qty=qty,
            limit=limit_price,
            broker_order_id=trade.order.orderId,
            client_ref=ref,
        )
        return BrokerOrder(
            broker_order_id=str(trade.order.orderId),
            ticker=ticker,
            side=side,
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
            if getattr(pos.contract, "symbol", None) != ticker:
                continue
            # Round-12: on a multi-account login (FA / linked accounts) rows
            # from other accounts inflated the qty, so the pending-exit
            # reconcile concluded "didn't fill" and resubmitted a full-size
            # sell. Filter by the configured account when one is set.
            pos_account = getattr(pos, "account", None)
            if self.account and pos_account and pos_account != self.account:
                continue
            total += int(pos.position)
        return total

    async def get_positions(
        self: IBKRBroker, *, refresh_timeout_s: float = 5.0
    ) -> list[PositionSnapshot]:
        """P2: every non-flat holding for the configured account, freshly
        synced (see get_position_qty for why the cached snapshot is not
        trusted). Rows for the same symbol are aggregated."""
        await asyncio.wait_for(self._ib.reqPositionsAsync(), timeout=refresh_timeout_s)
        agg: dict[str, tuple[int, float]] = {}
        for pos in self._ib.positions():
            symbol = getattr(pos.contract, "symbol", None)
            if not symbol:
                continue
            pos_account = getattr(pos, "account", None)
            if self.account and pos_account and pos_account != self.account:
                continue
            qty = int(pos.position)
            if qty == 0:
                continue
            cost = float(getattr(pos, "avgCost", 0.0) or 0.0)
            prev_qty, prev_cost = agg.get(symbol, (0, 0.0))
            total = prev_qty + qty
            blended = (prev_qty * prev_cost + qty * cost) / total if total else 0.0
            agg[symbol] = (total, blended)
        return [PositionSnapshot(ticker=t, qty=q, avg_cost=c) for t, (q, c) in agg.items() if q > 0]

    async def get_open_orders(self: IBKRBroker) -> list[BrokerOrder]:
        return [_broker_order_from_trade(t) for t in self._ib.openTrades()]

    async def get_order(self: IBKRBroker, broker_order_id: str) -> BrokerOrder | None:
        """P3: `IB.trades()` keeps done trades for the session, so a fill that
        arrived after placeOrder returned is visible here."""
        for trade in self._ib.trades():
            if str(trade.order.orderId) == broker_order_id:
                return _broker_order_from_trade(trade)
        return None

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
            # R6.C4: ib-async maintains the accountValues list via push events
            # on the asyncio event loop. Reading it from a thread (via
            # to_thread) races with those updates and can produce an
            # inconsistent partial view. accountValues() is a cheap in-memory
            # snapshot read, so call it directly on the event loop.
            # Round-13: the per-tick "idempotent re-subscribe" that lived here
            # was the BLOCKING reqAccountUpdates(); its RuntimeError was
            # swallowed below and this returned None on every tick, so the
            # drawdown and 3-day-loss killswitch arms were dead on IBKR. The
            # subscription is made once, in connect().
            values = self._ib.accountValues()
        except (ConnectionError, TimeoutError, OSError) as e:
            # R8.Q-I3: narrow per CLAUDE.md. AttributeError / RuntimeError
            # must propagate so a broker-impl bug surfaces as a real bug.
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
