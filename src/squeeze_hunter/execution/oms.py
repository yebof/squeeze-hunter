"""Order Management — drives a TwapPlan against an IBroker, re-prices each
slice from a fresh quote, escalates aggression when fills lag.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from squeeze_hunter.broker.base import BrokerOrder, IBroker
from squeeze_hunter.execution.slicing import (
    TwapPlan,
    default_aggression_schedule,
    escalate_aggression,
)
from squeeze_hunter.logging_setup import get_logger

log = get_logger("execution.oms")


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

            # Fetch fresh quote; fall back to slice's static price if unavailable
            try:
                q = await self.broker.fetch_quote(plan.ticker)
                mid = (q.bid + q.ask) / 2 if q.bid > 0 and q.ask > 0 else q.last
                if mid <= 0:
                    mid = slc.limit_price
            except Exception:
                log.warning("oms_quote_fallback", ticker=plan.ticker)
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
            if plan.side == "buy":
                limit_price = mid * (1 + agg_bps / 10_000)
            else:
                limit_price = mid * (1 - agg_bps / 10_000)

            submit = self.broker.submit_buy if plan.side == "buy" else self.broker.submit_sell
            order = await submit(
                ticker=plan.ticker,
                qty=slc.qty,
                limit_price=limit_price,
                ts=self.clock(),
            )
            result.orders.append(order)
            filled = (
                order.filled_qty
                if order.filled_qty
                else (order.qty if order.status == "filled" else 0)
            )
            if filled > 0:
                slices_filled += 1
                cumulative_qty += filled
                if order.avg_fill_price:
                    cumulative_value += filled * order.avg_fill_price
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
