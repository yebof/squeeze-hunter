"""Position lifecycle daemon — runs once per intraday tick (default 60s)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from squeeze_hunter.broker.base import IBroker
from squeeze_hunter.logging_setup import get_logger
from squeeze_hunter.risk.stops import StopState, evaluate_stops

log = get_logger("execution.lifecycle")


@dataclass
class LifecycleState:
    positions: dict[str, dict[str, Any]] = field(default_factory=dict)
    exits: list[dict[str, Any]] = field(default_factory=list)
    # I5: per-ticker in-flight set + lock to prevent concurrent ticks from
    # double-processing the same position.
    in_flight: set[str] = field(default_factory=set)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # R5.I1: cap exits history so it doesn't grow unbounded over long runs.
    # 1000 entries is plenty for any practical lookback; older entries should
    # be persisted to the DB if needed (Phase 4).
    exits_max_entries: int = 1000

    def record_exit(self: LifecycleState, entry: dict[str, Any]) -> None:
        """Append a new exit and trim to the configured max length."""
        self.exits.append(entry)
        if len(self.exits) > self.exits_max_entries:
            del self.exits[: len(self.exits) - self.exits_max_entries]


# Errors that are transient — log and continue. Other tickers in the same
# tick should still be processed; the position is re-evaluated next tick.
_TRANSIENT_FETCH_ERRORS = (ConnectionError, TimeoutError, OSError)


async def _process_one_position(
    state: LifecycleState,
    broker: IBroker,
    now: datetime,
    ticker: str,
) -> None:
    """Process a single ticker's position. Caller is responsible for the in_flight lock."""
    meta = state.positions.get(ticker)
    if meta is None:
        # Already removed by another caller while we were trying to acquire
        return
    try:
        q = await broker.fetch_quote(ticker)
    except _TRANSIENT_FETCH_ERRORS as e:
        log.warning(
            "quote_transient_error",
            ticker=ticker,
            err=str(e),
            err_type=type(e).__name__,
        )
        return
    # Programming / contract errors (AttributeError, NotImplementedError,
    # TypeError) are NOT caught here — they propagate up to tick_safe,
    # which logs them and surfaces them via structured logging. Masking
    # them as "transient" hid real bugs in earlier reviews.

    price = q.last or q.bid or q.ask
    if price <= 0.0:
        # Stale snapshot or halt — all three fields are zero. Don't
        # evaluate stops with a bogus price; come back next tick.
        log.warning(
            "quote_zero_price",
            ticker=ticker,
            bid=q.bid,
            ask=q.ask,
            last=q.last,
        )
        return

    meta["peak_price"] = max(meta["peak_price"], price)
    stop_state = StopState(
        entry_price=meta["entry_price"],
        peak_price=meta["peak_price"],
        current_score=meta["current_score"],
        entry_score=meta["entry_score"],
        bars_held=meta["bars_held"],
        setup_type=meta["setup_type"],
    )
    sig = evaluate_stops(stop_state, current_price=price)
    if sig.action == "hold":
        return
    if sig.action in {"halve", "exit"}:
        qty = meta["qty"] // 2 if sig.action == "halve" else meta["qty"]
        if qty <= 0:
            return
        order = await broker.submit_sell(
            ticker=ticker,
            qty=qty,
            limit_price=None,
            ts=now,
        )
        log.info(
            "lifecycle_exit",
            ticker=ticker,
            qty=qty,
            reason=sig.reason,
            broker_order_id=order.broker_order_id,
        )
        state.record_exit({"ts": now, "ticker": ticker, "qty": qty, "reason": sig.reason or "exit"})
        if sig.action == "exit":
            state.positions.pop(ticker, None)
        else:
            meta["qty"] -= qty


async def manage_positions(
    state: LifecycleState,
    broker: IBroker,
    now: datetime,
) -> LifecycleState:
    for ticker in list(state.positions):
        # I5: per-ticker check-and-set under the state lock. If another
        # concurrent tick is already processing this ticker, skip it — the
        # running task will handle any stop evaluation. The lock is held
        # only for the brief read-modify on in_flight; the actual
        # quote/stop/sell work happens outside the lock so other tickers
        # can proceed in parallel.
        async with state.lock:
            if ticker in state.in_flight:
                continue
            state.in_flight.add(ticker)
        try:
            await _process_one_position(state, broker, now, ticker)
        finally:
            async with state.lock:
                state.in_flight.discard(ticker)
    return state
