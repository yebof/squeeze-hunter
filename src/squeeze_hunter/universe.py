"""Universe builder. Pure function over a dataframe of universe candidates."""

from __future__ import annotations

from datetime import date

import pandas as pd

from squeeze_hunter.config import UniverseCfg


def build_universe(rows: pd.DataFrame, as_of: date, cfg: UniverseCfg) -> pd.DataFrame:
    """Add `included` and `exclusion_reason` columns."""
    out = rows.copy()
    reasons = []
    included = []
    for _, r in out.iterrows():
        if r["market_cap"] < cfg.min_market_cap:
            reasons.append("market_cap_below_floor")
            included.append(False)
        elif r["market_cap"] > cfg.max_market_cap:
            reasons.append("market_cap_above_ceiling")
            included.append(False)
        elif r["close"] < cfg.min_price:
            reasons.append("price_below_floor")
            included.append(False)
        elif r["days_listed"] < cfg.min_days_listed:
            reasons.append("listed_too_recently")
            included.append(False)
        else:
            reasons.append(None)
            included.append(True)
    out["included"] = included
    out["exclusion_reason"] = reasons
    out["as_of"] = as_of
    return out
