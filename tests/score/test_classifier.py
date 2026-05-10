import pandas as pd

from squeeze_hunter.score.classifier import classify_setups


def test_classifier_labels_each_type() -> None:
    df = pd.DataFrame(
        [
            # CAR-strong
            {
                "ticker": "HTZ",
                "score": 9.0,
                "f1_si_pct": 2.5,
                "f3_earnings_reaction": 2.0,
                "f4_wsb_mention": 0.5,
                "f5_call_oi_velocity": 0.5,
            },
            # GME-strong
            {
                "ticker": "GME",
                "score": 9.0,
                "f1_si_pct": 1.0,
                "f3_earnings_reaction": 0.0,
                "f4_wsb_mention": 2.5,
                "f5_call_oi_velocity": 2.5,
            },
            # Mixed
            {
                "ticker": "OKLO",
                "score": 8.5,
                "f1_si_pct": 1.5,
                "f3_earnings_reaction": 1.6,
                "f4_wsb_mention": 1.6,
                "f5_call_oi_velocity": 1.4,
            },
            # Weak
            {
                "ticker": "AAPL",
                "score": 1.0,
                "f1_si_pct": 0.1,
                "f3_earnings_reaction": 0.1,
                "f4_wsb_mention": 0.1,
                "f5_call_oi_velocity": 0.1,
            },
        ]
    )
    out = classify_setups(df)
    out_idx = out.set_index("ticker")
    assert out_idx.loc["HTZ", "setup_type"] == "CAR"
    assert out_idx.loc["GME", "setup_type"] == "GME"
    assert out_idx.loc["OKLO", "setup_type"] == "Mixed"
    assert out_idx.loc["AAPL", "setup_type"] == "Weak"
