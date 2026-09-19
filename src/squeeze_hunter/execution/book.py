"""The position book: one record shape for backtest, paper and live.

P1 of the architecture-hardening plan. Positions are plain dicts (the
lifecycle daemon, its tests and the runtime all address them by key) with
the keys documented here; `new_position_meta` is the only constructor so
the two paths cannot drift on which keys exist.
"""

from __future__ import annotations

from typing import Any

# Keys every position carries. `pending_*` and `last_mark` are optional and
# added by the daemon / runner as needed.
POSITION_KEYS: tuple[str, ...] = (
    "ticker",
    "qty",
    "entry_price",
    "peak_price",
    "entry_score",
    "current_score",
    "bars_held",
    "setup_type",
    "halved",
    "entry_commission_per_share",
)


def new_position_meta(
    *,
    ticker: str,
    qty: int,
    entry_price: float,
    score: float,
    setup_type: str,
    entry_commission_per_share: float = 0.0,
) -> dict[str, Any]:
    """A freshly filled position: peak = entry, no bars held, not halved."""
    return {
        "ticker": ticker,
        "qty": int(qty),
        "entry_price": float(entry_price),
        "peak_price": float(entry_price),
        "entry_score": float(score),
        "current_score": float(score),
        "bars_held": 0,
        "setup_type": setup_type,
        "halved": False,
        "entry_commission_per_share": float(entry_commission_per_share),
    }


def realized_pnl(
    meta: dict[str, Any], *, qty: int, fill_price: float, sell_commission: float
) -> tuple[float, float]:
    """(realized USD, pct return on cost) for selling `qty` of this position.

    R9.10 + R10.4: the entry commission is charged per share so a halve leg
    and the final exit each pay exactly their share, once. R7.C4: the pct
    return is what Kelly's payoff uses — invariant to position size and
    equity growth.
    """
    entry = float(meta["entry_price"])
    gross = (fill_price - entry) * qty
    realized = gross - float(meta.get("entry_commission_per_share", 0.0)) * qty - sell_commission
    cost_basis = entry * qty
    return realized, (realized / cost_basis if cost_basis > 0 else 0.0)
