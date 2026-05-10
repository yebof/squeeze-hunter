"""Position lifecycle daemon — runs once per intraday tick (default 60s)."""

from __future__ import annotations

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


async def manage_positions(
    state: LifecycleState,
    broker: IBroker,
    now: datetime,
) -> LifecycleState:
    for ticker in list(state.positions):
        meta = state.positions[ticker]
        try:
            q = await broker.fetch_quote(ticker)
        except Exception as e:
            log.warning("quote_unavailable", ticker=ticker, err=str(e))
            continue
        price = q.last or q.bid or q.ask
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
            continue
        if sig.action in {"halve", "exit"}:
            qty = meta["qty"] // 2 if sig.action == "halve" else meta["qty"]
            if qty <= 0:
                continue
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
            state.exits.append(
                {"ts": now, "ticker": ticker, "qty": qty, "reason": sig.reason or "exit"}
            )
            if sig.action == "exit":
                state.positions.pop(ticker, None)
            else:
                meta["qty"] -= qty
    return state
