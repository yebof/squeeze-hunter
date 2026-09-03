"""Round-12: classifier cutoffs come from settings.score.setup_thresholds."""

from __future__ import annotations

import pandas as pd

from squeeze_hunter.score.classifier import classify_setups


def _row(a1: float, a3: float, b4: float = 0.0, b5: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "X",
                "f1_si_pct": a1,
                "f3_earnings_reaction": a3,
                "f4_wsb_mention": b4,
                "f5_call_oi_velocity": b5,
                "score": a1 + a3 + b4 + b5,
            }
        ]
    )


def test_classifier_honours_strong_threshold() -> None:
    assert classify_setups(_row(2.5, 2.0)).iloc[0]["setup_type"] == "CAR"
    assert classify_setups(_row(2.5, 2.0), strong=5.0).iloc[0]["setup_type"] == "Weak"


def test_classifier_honours_mixed_floor() -> None:
    df = _row(2.0, 2.0, 1.5, 1.5)  # A=4.0, B=3.0
    assert classify_setups(df).iloc[0]["setup_type"] == "Mixed"
    assert classify_setups(df, mixed_floor=3.5).iloc[0]["setup_type"] == "CAR"
