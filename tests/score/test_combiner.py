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


def test_unknown_factor_treated_as_zero_when_others_mapped() -> None:
    """When SOME factors map (but one is unknown), the unknown contributes 0
    and the known ones contribute normally. Tests partial-mapping behavior."""
    df = pd.DataFrame(
        [
            {"ticker": "GME", "factor_name": "f9_mystery", "raw_value": 1.0, "z_score": 5.0},
            {"ticker": "GME", "factor_name": "f1_si_pct", "raw_value": 0.2, "z_score": 3.0},
        ]
    )
    result = combine(df, weights={"f1_si_pct": 2.0})
    # f1 contributes 2.0 * 3.0 = 6.0; f9 unknown → 0
    assert result.loc[result["ticker"] == "GME", "score"].iloc[0] == pytest.approx(6.0)


def test_combiner_raises_when_all_factors_unmapped() -> None:
    """R5.C6 regression: an empty score.weights dict (or a YAML where every
    factor name is misspelled) used to silently produce all-zero scores → the
    scanner appeared to run but emitted no actionable candidates forever.
    Now the combiner fails loud.
    """
    df = pd.DataFrame(
        [{"ticker": "GME", "factor_name": "f7_volume_spke", "raw_value": 1.0, "z_score": 5.0}]
    )
    with pytest.raises(ValueError, match="no factor in the input has a weight"):
        combine(df, weights={"f7_volume_spike": 1.0})


def test_combiner_raises_when_weights_empty() -> None:
    """R5.C6: empty weights dict is a configuration error — fail loud."""
    df = pd.DataFrame(
        [{"ticker": "GME", "factor_name": "f1_si_pct", "raw_value": 0.2, "z_score": 3.0}]
    )
    with pytest.raises(ValueError, match="no factor in the input has a weight"):
        combine(df, weights={})
