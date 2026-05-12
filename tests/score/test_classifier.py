import pandas as pd

from squeeze_hunter.score.classifier import classify_setups


def test_classifier_strong_a_moderate_b_classifies_as_car() -> None:
    """R9.8 regression: strong CAR strength (A>=4) with moderate GME signal
    (2<=B<3) used to fall into "Weak" because the CAR rule required B<2.
    A "Weak" setup gets the Weak Kelly priors (fraction=0) so the would-be
    CAR trade was silently suppressed despite a clearly strong A.
    Loosened CAR rule to B<3 — matches Mixed's lower bound symmetrically.
    """
    df = pd.DataFrame(
        [
            {
                "ticker": "HTZ",
                "score": 10.0,
                "f1_si_pct": 2.5,
                "f3_earnings_reaction": 2.0,  # A = 4.5
                "f4_wsb_mention": 1.2,
                "f5_call_oi_velocity": 1.3,  # B = 2.5 (in dead zone)
            },
        ]
    )
    out = classify_setups(df)
    assert out.set_index("ticker").loc["HTZ", "setup_type"] == "CAR"


def test_classifier_strong_b_moderate_a_classifies_as_gme() -> None:
    """R9.8 symmetric: strong GME (B>=4) with moderate CAR signal (2<=A<3)
    should classify as GME, not Weak.
    """
    df = pd.DataFrame(
        [
            {
                "ticker": "GME",
                "score": 10.0,
                "f1_si_pct": 1.5,
                "f3_earnings_reaction": 1.0,  # A = 2.5 (in dead zone)
                "f4_wsb_mention": 2.5,
                "f5_call_oi_velocity": 2.0,  # B = 4.5
            },
        ]
    )
    out = classify_setups(df)
    assert out.set_index("ticker").loc["GME", "setup_type"] == "GME"


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
