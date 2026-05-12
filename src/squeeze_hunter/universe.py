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
        # R10.10: reject data-quality red flags (non-positive price or market
        # cap) BEFORE the configured floor checks. The floor checks would
        # already reject them by accident with realistic min_price /
        # min_market_cap settings, but a custom cfg with a 0-floor would let
        # delisted/halted rows through to the scan, where zero prices divide
        # by zero in factor computation. A distinct reason also helps ops
        # distinguish "bad data" from "below threshold."
        if not (r["close"] > 0):
            reasons.append("price_nonpositive")
            included.append(False)
        elif not (r["market_cap"] > 0):
            reasons.append("market_cap_nonpositive")
            included.append(False)
        elif r["market_cap"] < cfg.min_market_cap:
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
