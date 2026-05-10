"""Setup classifier — pure rule-based labeling.

A = z[f1_si_pct] + z[f3_earnings_reaction]    # CAR strength
B = z[f4_wsb_mention] + z[f5_call_oi_velocity] # GME strength

if   A >= 4 and B  < 2  → CAR
elif B >= 4 and A  < 2  → GME
elif A >= 3 and B >= 3  → Mixed
else                    → Weak
"""

from __future__ import annotations

import pandas as pd


def classify_setups(df: pd.DataFrame) -> pd.DataFrame:
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
        if row_a >= 4.0 and row_b < 2.0:
            return "CAR"
        if row_b >= 4.0 and row_a < 2.0:
            return "GME"
        if row_a >= 3.0 and row_b >= 3.0:
            return "Mixed"
        return "Weak"

    out["car_strength"] = a
    out["gme_strength"] = b
    out["setup_type"] = [label(x, y) for x, y in zip(a, b, strict=False)]
    return out
