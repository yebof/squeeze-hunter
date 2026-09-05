"""Price helpers shared by the lifecycle daemon and the OMS."""

from __future__ import annotations

import math


def round_to_tick(price: float, side: str) -> float:
    """Snap a limit price to the US equity minimum price variation.

    Round-13: `round(price * 0.995, 4)` produced sub-penny limits (24.53 →
    24.4074) for every stock at or above $1.00. IBKR rejects those with error
    110, ib_async marks the trade Cancelled, and the daemon resubmitted the
    same bad price every tick — the hard stop never executed. Sells round
    DOWN and buys round UP so the order stays at least as marketable as the
    caller intended.
    """
    tick = 0.01 if price >= 1.0 else 0.0001
    units = round(price / tick, 6)
    snapped = math.floor(units) if side == "sell" else math.ceil(units)
    return round(snapped * tick, 4)
