"""Order Management — drives a TwapPlan against an IBroker, returns realized fills."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from squeeze_hunter.broker.base import BrokerOrder, IBroker
from squeeze_hunter.execution.slicing import TwapPlan
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
    clock: Callable[[], datetime]  # injected for testability

    async def execute(
        self: OrderManager,
        plan: TwapPlan,
        *,
        max_wall_seconds: int = 600,
    ) -> ExecutionResult:
        result = ExecutionResult()
        cumulative_qty = 0
        cumulative_value = 0.0

        for slc in plan.slices:
            now = self.clock()
            if slc.submit_at > now and max_wall_seconds > 0:
                wait_s = min(max_wall_seconds, int((slc.submit_at - now).total_seconds()))
                await asyncio.sleep(wait_s)
                max_wall_seconds -= wait_s

            submit = self.broker.submit_buy if plan.side == "buy" else self.broker.submit_sell
            order = await submit(
                ticker=plan.ticker,
                qty=slc.qty,
                limit_price=slc.limit_price,
                ts=self.clock(),
            )
            result.orders.append(order)
            filled = (
                order.filled_qty
                if order.filled_qty
                else (order.qty if order.status == "filled" else 0)
            )
            cumulative_qty += filled
            if filled and order.avg_fill_price:
                cumulative_value += filled * order.avg_fill_price
            log.info(
                "slice_submitted",
                ticker=plan.ticker,
                slice_qty=slc.qty,
                limit=slc.limit_price,
                filled=filled,
                broker_order_id=order.broker_order_id,
            )

        result.filled_qty = cumulative_qty
        result.unfilled_qty = sum(s.qty for s in plan.slices) - cumulative_qty
        result.avg_fill_price = cumulative_value / cumulative_qty if cumulative_qty > 0 else 0.0
        return result
