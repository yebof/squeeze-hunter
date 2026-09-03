"""Setup classifier — pure rule-based labeling.

A = z[f1_si_pct] + z[f3_earnings_reaction]    # CAR strength
B = z[f4_wsb_mention] + z[f5_call_oi_velocity] # GME strength

R9.8: the CAR/GME "other-axis" cutoff is the same as Mixed's lower bound (3.0).
Prior cutoff of 2.0 left a dead zone (e.g., A=4.5, B=2.5) where strong-A but
moderately-noisy-B setups were dropped into Weak — Weak gets fraction=0 Kelly
sizing, suppressing real CAR trades.

if   A >= 4 and B  < 3  → CAR
elif B >= 4 and A  < 3  → GME
elif A >= 3 and B >= 3  → Mixed
else                    → Weak
"""

from __future__ import annotations

import pandas as pd


def classify_setups(
    df: pd.DataFrame, *, strong: float = 4.0, mixed_floor: float = 3.0
) -> pd.DataFrame:
    """Label rows CAR / GME / Mixed / Weak.

    Round-12: cutoffs are parameters (fed from settings.score.setup_thresholds
    by scan.run_scan) instead of hardcoded 4.0 / 3.0 — the YAML key existed but
    was never read, violating the one-config-file rule.
    """
    out = df.copy()

    # Use .get() pattern to handle missing factor columns gracefully (e.g., when
    # upstream data source returns no rows for a factor like short interest).
    def _col(name: str) -> pd.Series:
        if name in out.columns:
            return out[name].fillna(0.0)
        return pd.Series(0.0, index=out.index)

    a = _col("f1_si_pct") + _col("f3_earnings_reaction")
    b = _col("f4_wsb_mention") + _col("f5_call_oi_velocity")

    def label(row_a: float, row_b: float) -> str:
        # R9.8: cutoff matches Mixed's floor (3.0). Strong CAR with moderate
        # GME signal still classifies as CAR rather than the prior Weak.
        if row_a >= strong and row_b < mixed_floor:
            return "CAR"
        if row_b >= strong and row_a < mixed_floor:
            return "GME"
        if row_a >= mixed_floor and row_b >= mixed_floor:
            return "Mixed"
        return "Weak"

    out["car_strength"] = a
    out["gme_strength"] = b
    out["setup_type"] = [label(x, y) for x, y in zip(a, b, strict=False)]
    return out
