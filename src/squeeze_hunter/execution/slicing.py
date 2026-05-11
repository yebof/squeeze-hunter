"""TWAP slicer for entry orders.

Spec Section 6:
    09:30-09:35  no orders (avoid open-auction noise)
    09:35-09:55  TWAP 6-8 slices, ~150-180s apart
                 limit price = mid + 0.5 * spread
                 after 5 unfilled → switch to mid + 1 * spread on remaining
                 after 80% unfilled → marketable limits
                 hard cap at 09:55 → marketable limits

This module produces the *plan* — the OMS (Task 3.4) executes it and re-prices
slices on the fly using current quotes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


def default_aggression_schedule(
    n_slices: int,
    starting_bps: float = 5.0,
    ending_bps: float = 30.0,
) -> list[float]:
    """Per-slice aggression bps under normal (well-filling) conditions.

    Returns a list of length n_slices that linearly interpolates from
    starting_bps to ending_bps.
    """
    if n_slices <= 0:
        return []
    if n_slices == 1:
        return [starting_bps]
    return [
        starting_bps + (ending_bps - starting_bps) * i / (n_slices - 1) for i in range(n_slices)
    ]


def escalate_aggression(
    base_bps: float,
    slices_submitted: int,
    slices_filled: int,
    marketable_bps: float = 50.0,
    marketable_unfilled_threshold: float = 0.80,
) -> float:
    """If fills are lagging, escalate to marketable territory.

    Returns the aggression bps to use for the next slice.
    """
    if slices_submitted == 0:
        return base_bps
    unfilled_pct = (slices_submitted - slices_filled) / slices_submitted
    if unfilled_pct >= marketable_unfilled_threshold:
        return marketable_bps
    return base_bps


@dataclass(slots=True, frozen=True)
class Slice:
    submit_at: datetime
    qty: int
    limit_price: float
    aggression_bps: float  # 0 = mid, +50 = marketable buy


@dataclass(slots=True, frozen=True)
class TwapPlan:
    ticker: str
    side: str  # "buy" | "sell"
    slices: list[Slice]


def build_twap_plan(
    total_qty: int,
    reference_price: float,
    market_open: datetime,
    *,
    ticker: str = "",
    side: str = "buy",
    n_slices: int = 6,
    window_minutes: int = 20,
    slice_offset_minutes: int = 5,
    starting_aggression_bps: float = 5.0,
    ending_aggression_bps: float = 30.0,
) -> TwapPlan:
    if total_qty <= 0 or n_slices <= 0:
        return TwapPlan(ticker=ticker, side=side, slices=[])
    # R8.S-I3: when total_qty < n_slices, base_qty=0 and only the last slice
    # carries the remainder. Submitting zero-qty slices wastes broker
    # round-trips and confuses the OMS aggression-escalation accounting.
    # Cap n_slices so every emitted slice has qty >= 1.
    n_slices = min(n_slices, total_qty)
    base_qty = total_qty // n_slices
    remainder = total_qty - base_qty * n_slices
    spacing = timedelta(seconds=(window_minutes * 60) // max(n_slices - 1, 1))
    start = market_open + timedelta(minutes=slice_offset_minutes)
    slices: list[Slice] = []
    for i in range(n_slices):
        qty = base_qty + (remainder if i == n_slices - 1 else 0)
        if qty <= 0:
            # Defensive — shouldn't happen given the n_slices clamp above.
            continue
        agg = starting_aggression_bps + (
            (ending_aggression_bps - starting_aggression_bps) * i / max(n_slices - 1, 1)
        )
        bps = agg if side == "buy" else -agg
        limit_price = reference_price * (1 + bps / 10_000)
        slices.append(
            Slice(
                submit_at=start + spacing * i,
                qty=qty,
                limit_price=limit_price,
                aggression_bps=agg,
            )
        )
    return TwapPlan(ticker=ticker, side=side, slices=slices)
