"""Position lifecycle daemon — runs once per intraday tick (default 60s)."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from squeeze_hunter.broker.base import IBroker
from squeeze_hunter.logging_setup import get_logger
from squeeze_hunter.risk.stops import (
    _DEFAULT_TRAILING_CAR,
    _DEFAULT_TRAILING_GME,
    _DEFAULT_TRAILING_MIXED,
    StopState,
    evaluate_stops,
)

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
    # R9.1: stops parameters live on the state so RuntimeContext can populate
    # them from settings. Defaults match the prior hardcoded values so existing
    # tests that construct LifecycleState() directly behave identically.
    hard_stop: float = -0.12
    time_stop_bars: int = 21
    signal_decay_halve: float = 0.50
    signal_decay_exit: float = 0.75
    trailing_car: float = _DEFAULT_TRAILING_CAR
    trailing_gme: float = _DEFAULT_TRAILING_GME
    trailing_mixed: float = _DEFAULT_TRAILING_MIXED

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
    if not math.isfinite(price) or price <= 0.0:
        # Stale snapshot, halt, or a non-finite (NaN) quote. Don't evaluate stops
        # with a bogus price; come back next tick. R11 defense-in-depth: `nan`
        # is truthy and `nan <= 0.0` is False, so without the isfinite check a
        # NaN price slips past this guard and the hard/trailing stops silently
        # never fire (every comparison against nan is False).
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
    # R9.1: pass the YAML-driven stops parameters (carried on LifecycleState)
    # so paper/live behave identically to the backtest. Previously this call
    # used only defaults — every YAML knob in `stops:` was silently dead.
    sig = evaluate_stops(
        stop_state,
        current_price=price,
        hard_stop=state.hard_stop,
        time_stop_bars=state.time_stop_bars,
        signal_decay_halve=state.signal_decay_halve,
        signal_decay_exit=state.signal_decay_exit,
        trailing_car=state.trailing_car,
        trailing_gme=state.trailing_gme,
        trailing_mixed=state.trailing_mixed,
    )
    if sig.action == "hold":
        return
    if sig.action in {"halve", "exit"}:
        stale_pending: list[str] = list(meta.get("pending_exits", []))
        # CDX-P1-3: reconcile a stale pending exit against the broker's REAL
        # position BEFORE cancelling/resubmitting. A stale order that filled
        # between ticks is gone from open orders — cancel_order just returns
        # False without error — and R9.3's blind full-qty resubmit would then
        # SHORT an already-flat account. The broker's position is the
        # authoritative truth; meta["qty"] is a local mirror that can lag.
        if stale_pending:
            try:
                broker_qty = await broker.get_position_qty(ticker)
            except _TRANSIENT_FETCH_ERRORS as e:
                # Can't confirm broker state → conservative: do NOT resubmit
                # a full-size sell this tick (shorting risk). Next tick retries.
                log.warning(
                    "lifecycle_reconcile_failed",
                    ticker=ticker,
                    err=str(e),
                    err_type=type(e).__name__,
                    note="pending exit not reconciled; skipping resubmit this tick",
                )
                return
            if broker_qty <= 0:
                # Prior exit fully filled — broker is flat. Close out locally;
                # there is nothing to cancel (order is terminal) or resubmit.
                log.info(
                    "lifecycle_pending_exit_reconciled_flat",
                    ticker=ticker,
                    local_qty=meta["qty"],
                )
                meta["pending_exits"] = []
                state.record_exit(
                    {
                        "ts": now,
                        "ticker": ticker,
                        "qty": meta["qty"],
                        "reason": "reconciled_filled_exit",
                    }
                )
                state.positions.pop(ticker, None)
                return
            if broker_qty < meta["qty"]:
                # Stale exit partially filled — trust the broker's remaining
                # quantity so the resubmit doesn't oversell into a short.
                log.warning(
                    "lifecycle_pending_exit_partial_reconciled",
                    ticker=ticker,
                    local_qty=meta["qty"],
                    broker_qty=broker_qty,
                )
                meta["qty"] = broker_qty
            # else broker_qty >= meta["qty"]: stale exit didn't fill — fall
            # through to the R9.3 cancel-then-resubmit below.

        qty = meta["qty"] // 2 if sig.action == "halve" else meta["qty"]
        if qty <= 0:
            return
        # R9.3: cancel any prior STILL-OPEN exit orders before resubmitting.
        # Without this, a stop that triggers two ticks in a row leaves TWO
        # live sell orders on the broker. If both fill the position flips
        # short — a serious safety bug. Transient cancel failures are logged
        # and we proceed with the new exit (the broker will eventually time
        # out the stale order or the next tick will retry the cancel).
        # Programming errors (AttributeError on a misconfigured broker) MUST
        # propagate per the same rule as fetch_quote above.
        for stale_id in stale_pending:
            try:
                await broker.cancel_order(stale_id)
            except _TRANSIENT_FETCH_ERRORS as e:
                log.warning(
                    "lifecycle_cancel_stale_failed",
                    ticker=ticker,
                    broker_order_id=stale_id,
                    err=str(e),
                    err_type=type(e).__name__,
                )
        # Clear the list now — the new submit will repopulate it if the
        # fresh order also goes pending.
        if stale_pending:
            meta["pending_exits"] = []
        # R7.I7: emit a marketable limit (50 bps below current bid/last)
        # instead of a market order. Market sells route into whatever
        # liquidity is available — during halts, AH, or thin sessions
        # they can fill 5-20% adverse on small-float tickers. A limit
        # 50 bps below mid is still aggressive enough to hit the inside
        # in normal liquidity but caps catastrophic fills.
        marketable_limit = round(price * 0.995, 4)  # 50 bps below current
        order = await broker.submit_sell(
            ticker=ticker,
            qty=qty,
            limit_price=marketable_limit,
            ts=now,
        )
        log.info(
            "lifecycle_exit",
            ticker=ticker,
            qty=qty,
            reason=sig.reason,
            broker_order_id=order.broker_order_id,
            status=order.status,
        )
        # R8.S-C1: only pop the position on a CONFIRMED fill. With market
        # orders (R7.I7 replaced them) the simulator always filled, so
        # popping immediately was safe. With marketable limits, IBKR may
        # return "pending"/"submitted" — popping immediately leaves real
        # exposure at the broker while the lifecycle daemon thinks it's
        # closed. On a gap-down through the limit, the order may never
        # fill at the original price and needs to be retried (cancel-replace
        # or escalate). Until then, keep the position so the next tick
        # re-evaluates the stop with a fresh quote.
        if order.status == "filled":
            state.record_exit(
                {"ts": now, "ticker": ticker, "qty": qty, "reason": sig.reason or "exit"}
            )
            if sig.action == "exit":
                state.positions.pop(ticker, None)
            else:
                meta["qty"] -= qty
        else:
            # Pending / partial — leave the position in place; next tick will
            # re-evaluate. Track the pending order on the meta so a future
            # cancel-replace path can find it.
            meta.setdefault("pending_exits", []).append(order.broker_order_id)
            log.warning(
                "lifecycle_exit_unfilled",
                ticker=ticker,
                qty=qty,
                status=order.status,
                broker_order_id=order.broker_order_id,
                note="position retained; next tick re-evaluates the stop",
            )


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
