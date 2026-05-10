"""Cross-sectional normalization."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_z(series: pd.Series, clip: float = 3.0) -> pd.Series:
    """Cross-sectional z-score with clipping.

    Handles three pathological inputs without polluting downstream factors:
    - All-equal series → returns zeros (std=0 path).
    - NaN entries → excluded from mean/std; emit 0 z for those positions.
    - Inf entries (e.g., days-to-cover when ADV20=0) → excluded from mean/std;
      emit 0 z for those positions. Without this, a single infinite value
      pushed mean to inf and std to NaN, zeroing out the entire universe.
    """
    finite = series.replace([np.inf, -np.inf], np.nan)
    valid = finite.dropna()
    if len(valid) == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    mean = float(valid.mean())
    std = float(valid.std(ddof=0))
    if std == 0.0 or not np.isfinite(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    z = (finite - mean) / std
    z = z.clip(-clip, clip)
    # Replace NaN (originally inf or NaN) entries with 0 so callers don't
    # propagate NaN into the score combiner.
    return z.fillna(0.0)
