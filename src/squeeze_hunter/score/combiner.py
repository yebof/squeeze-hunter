"""Linear weighted z-score combiner. Pure pandas."""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger("score.combiner")


def combine(factors_long: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """factors_long: columns ticker, factor_name, raw_value, z_score.
    Returns wide df: ticker, score, plus each factor's z as a column."""
    if factors_long.empty:
        return pd.DataFrame(columns=["ticker", "score"])
    df = factors_long.copy()
    mapped = df["factor_name"].map(weights)
    unmapped = df.loc[mapped.isna(), "factor_name"].unique().tolist()
    if unmapped:
        log.warning("score_combiner_unmapped_factors factors=%s", unmapped)
    df["weighted"] = mapped.fillna(0.0) * df["z_score"]
    score = (
        df.groupby("ticker", as_index=False)["weighted"].sum().rename(columns={"weighted": "score"})
    )
    pivot = df.pivot_table(index="ticker", columns="factor_name", values="z_score").reset_index()
    return score.merge(pivot, on="ticker", how="left")
