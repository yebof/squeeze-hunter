"""Order Management — drives a TwapPlan against an IBroker, re-prices each
slice from a fresh quote, escalates aggression when fills lag.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from squeeze_hunter.broker.base import BrokerOrder, IBroker
from squeeze_hunter.execution.pricing import round_to_tick
from squeeze_hunter.execution.slicing import (
    TwapPlan,
    default_aggression_schedule,
    escalate_aggression,
)
from squeeze_hunter.logging_setup import get_logger

log = get_logger("execution.oms")

_TERMINAL = frozenset({"filled", "cancelled", "rejected", "expired"})


@dataclass
class ExecutionResult:
    filled_qty: int = 0
    unfilled_qty: int = 0
    avg_fill_price: float = 0.0
    orders: list[BrokerOrder] = field(default_factory=list)


@dataclass
class OrderManager:
    broker: IBroker
    clock: Callable[[], datetime]

    async def execute(
        self: OrderManager,
        plan: TwapPlan,
        *,
        max_wall_seconds: int = 600,
        marketable_bps: float = 50.0,
        poll_interval_s: float = 0.5,
        slice_fill_wait_s: float | None = None,
    ) -> ExecutionResult:
        result = ExecutionResult()
        cumulative_qty = 0
        cumulative_value = 0.0
        slices_filled = 0
        n_slices = len(plan.slices)
        base_schedule = default_aggression_schedule(n_slices)

        for i, slc in enumerate(plan.slices):
            now = self.clock()
            if slc.submit_at > now and max_wall_seconds > 0:
                wait_s = min(max_wall_seconds, int((slc.submit_at - now).total_seconds()))
                await asyncio.sleep(wait_s)
                max_wall_seconds -= wait_s

            # Fetch fresh quote; fall back to slice's static price if unavailable.
            # R7.I4: narrow to transient I/O errors per CLAUDE.md. Programming
            # errors (AttributeError on a misconfigured broker, NotImplementedError
            # from a wrong provider) must propagate up instead of silently
            # executing at the slice's stale planned price.
            try:
                q = await self.broker.fetch_quote(plan.ticker)
                mid = (q.bid + q.ask) / 2 if q.bid > 0 and q.ask > 0 else q.last
                if mid <= 0:
                    mid = slc.limit_price
            except (ConnectionError, TimeoutError, OSError) as e:
                log.warning(
                    "oms_quote_fallback",
                    ticker=plan.ticker,
                    err=str(e),
                    err_type=type(e).__name__,
                )
                mid = slc.limit_price

            # Pick aggression: base schedule, escalated if fills are lagging.
            # At iteration i, exactly i slices have been submitted before this one.
            base_bps = base_schedule[i] if i < len(base_schedule) else 30.0
            agg_bps = escalate_aggression(
                base_bps=base_bps,
                slices_submitted=i,
                slices_filled=slices_filled,
                marketable_bps=marketable_bps,
            )

            # Buy: limit above mid (positive bps); sell: limit below mid (negative bps)
            # Round-13: snap to the minimum price variation (IBKR rejects
            # sub-penny limits with error 110).
            if plan.side == "buy":
                limit_price = round_to_tick(mid * (1 + agg_bps / 10_000), side="buy")
            else:
                limit_price = round_to_tick(mid * (1 - agg_bps / 10_000), side="sell")

            submit = self.broker.submit_buy if plan.side == "buy" else self.broker.submit_sell
            order = await submit(
                ticker=plan.ticker,
                qty=slc.qty,
                limit_price=limit_price,
                ts=self.clock(),
            )
            # P3: a live broker returns "pending" on the submitting call and
            # fills later. Poll the order until it is terminal or the slice's
            # time budget is spent, then cancel the remainder so two slices
            # are never working at once. The previous code read the fill off
            # the submit response, which only the simulator ever populated.
            next_submit = plan.slices[i + 1].submit_at if i + 1 < n_slices else None
            budget = slice_fill_wait_s
            if budget is None:
                budget = (
                    max(0.0, (next_submit - self.clock()).total_seconds())
                    if next_submit is not None
                    else 60.0
                )
            order = await self._await_terminal(order, budget_s=budget, poll_s=poll_interval_s)
            result.orders.append(order)
            filled = (
                order.filled_qty
                if order.filled_qty
                else (order.qty if order.status == "filled" else 0)
            )
            # R9.5: only fully-filled slices advance `slices_filled`. The prior
            # `if filled > 0: slices_filled += 1` counted partial fills as full,
            # so a string of small partials looked like full success to
            # escalate_aggression — TWAP never bumped to marketable bps and
            # ended with most of the order unfilled at end-of-window.
            if order.status == "filled":
                slices_filled += 1
            if filled > 0:
                cumulative_qty += filled
                # R11: a live broker can return status="filled" with
                # avg_fill_price=None (orderStatus not yet synced when placeOrder
                # returns an immediate fill). Fall back to the slice limit so
                # qty and value advance together — otherwise the fill enters the
                # denominator but not the numerator and the reported
                # avg_fill_price is dragged toward 0 (exactly 0.0 if every slice
                # is priceless), corrupting cost basis / P&L / stop math.
                fill_px = order.avg_fill_price if order.avg_fill_price else limit_price
                if fill_px:
                    cumulative_value += filled * fill_px
            log.info(
                "slice_submitted",
                ticker=plan.ticker,
                slice_qty=slc.qty,
                mid=mid,
                aggression_bps=agg_bps,
                limit=limit_price,
                filled=filled,
                broker_order_id=order.broker_order_id,
            )

        result.filled_qty = cumulative_qty
        result.unfilled_qty = sum(s.qty for s in plan.slices) - cumulative_qty
        result.avg_fill_price = cumulative_value / cumulative_qty if cumulative_qty > 0 else 0.0
        return result

    async def _await_terminal(
        self: OrderManager, order: BrokerOrder, *, budget_s: float, poll_s: float
    ) -> BrokerOrder:
        """Poll `get_order` until the order is terminal or `budget_s` elapses;
        then cancel and report whatever filled. Transient broker errors end
        the wait early (the order stays live at the broker and is cancelled)."""
        if order.status in _TERMINAL:
            return order
        max_polls = int(budget_s / poll_s) if poll_s > 0 else min(int(budget_s * 1000), 10_000)
        latest = order
        for _ in range(max_polls):
            if poll_s > 0:
                await asyncio.sleep(poll_s)
            try:
                fetched = await self.broker.get_order(order.broker_order_id)
            except (ConnectionError, TimeoutError, OSError) as e:
                log.warning("oms_poll_failed", broker_order_id=order.broker_order_id, err=str(e))
                break
            if fetched is None:
                break  # the broker no longer knows it; nothing more to learn
            latest = fetched
            if latest.status in _TERMINAL:
                return latest
        try:
            await self.broker.cancel_order(order.broker_order_id)
        except (ConnectionError, TimeoutError, OSError) as e:
            log.warning("oms_cancel_failed", broker_order_id=order.broker_order_id, err=str(e))
            return latest
        for _ in range(20):
            try:
                fetched = await self.broker.get_order(order.broker_order_id)
            except (ConnectionError, TimeoutError, OSError):
                break
            if fetched is None:
                break
            latest = fetched
            if latest.status in _TERMINAL:
                break
            if poll_s > 0:
                await asyncio.sleep(poll_s)
        log.warning(
            "oms_slice_cancelled",
            broker_order_id=order.broker_order_id,
            status=latest.status,
            filled=latest.filled_qty,
        )
        return latest
