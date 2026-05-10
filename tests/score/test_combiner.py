import pandas as pd
import pytest

from squeeze_hunter.score.combiner import combine


def test_score_is_weighted_sum_of_z() -> None:
    df = pd.DataFrame(
        [
            {"ticker": "GME", "factor_name": "f1_si_pct", "raw_value": 0.20, "z_score": 3.0},
            {
                "ticker": "GME",
                "factor_name": "f3_earnings_reaction",
                "raw_value": 0.05,
                "z_score": 1.0,
            },
        ]
    )
    weights = {"f1_si_pct": 2.0, "f3_earnings_reaction": 2.0}
    result = combine(df, weights=weights)
    row = result.set_index("ticker").loc["GME"]
    assert row["score"] == pytest.approx(2.0 * 3.0 + 2.0 * 1.0)


def test_unknown_factor_zero_weight() -> None:
    df = pd.DataFrame(
        [{"ticker": "GME", "factor_name": "f9_mystery", "raw_value": 1.0, "z_score": 5.0}]
    )
    result = combine(df, weights={"f1_si_pct": 2.0})
    assert result.loc[result["ticker"] == "GME", "score"].iloc[0] == 0.0


def test_combiner_logs_warning_on_unmapped_factor(caplog) -> None:
    """B5: a YAML typo means a real factor name maps to no weight. Currently
    silent. Should emit a structured warning so operators catch it.
    """
    import logging

    df = pd.DataFrame(
        [
            {
                "ticker": "GME",
                "factor_name": "f7_volume_spke",  # typo intentional
                "raw_value": 1.0,
                "z_score": 5.0,
            }
        ]
    )
    weights = {"f7_volume_spike": 1.0}  # the correct name; the typo doesn't match
    with caplog.at_level(logging.WARNING):
        combine(df, weights=weights)
    msg = " ".join(rec.message for rec in caplog.records)
    assert "f7_volume_spke" in msg or "unmapped" in msg.lower()
