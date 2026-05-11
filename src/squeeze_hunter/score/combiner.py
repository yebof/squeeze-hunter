"""Linear weighted z-score combiner. Pure pandas."""

from __future__ import annotations

import pandas as pd

from squeeze_hunter.logging_setup import get_logger

# R5.I2: use structlog (via get_logger), not stdlib logging. Previously,
# stdlib `logging.warning(...)` only surfaced in test environments (where
# pytest configures the root logger). In production, structlog's
# PrintLoggerFactory is the only sink, so unmapped-factor warnings —
# critical for catching YAML typos that produce all-zero scores — were
# silently dropped.
log = get_logger("score.combiner")


def combine(factors_long: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """factors_long: columns ticker, factor_name, raw_value, z_score.
    Returns wide df: ticker, score, plus each factor's z as a column.

    R5.C6: if `weights` is empty or every factor in `factors_long` is unmapped,
    every score is 0. This is a configuration error (missing YAML or typo'd
    factor names) — raise explicitly so the scanner doesn't silently emit
    zero candidates forever.
    """
    if factors_long.empty:
        return pd.DataFrame(columns=["ticker", "score"])
    df = factors_long.copy()
    mapped = df["factor_name"].map(weights)
    unmapped = df.loc[mapped.isna(), "factor_name"].unique().tolist()
    if unmapped:
        log.warning("score_combiner_unmapped_factors", factors=unmapped)
    # R5.C6: if NO factor in this batch is mapped, every score will be 0 →
    # silent no-trade mode. Fail loud instead.
    if mapped.notna().sum() == 0:
        raise ValueError(
            f"score.combine: no factor in the input has a weight. "
            f"weights={list(weights)}; factors_in_input={unmapped}. "
            "This usually means config/settings.yml is missing or score.weights is empty."
        )
    df["weighted"] = mapped.fillna(0.0) * df["z_score"]
    score = (
        df.groupby("ticker", as_index=False)["weighted"].sum().rename(columns={"weighted": "score"})
    )
    pivot = df.pivot_table(index="ticker", columns="factor_name", values="z_score").reset_index()
    return score.merge(pivot, on="ticker", how="left")
