"""Cross-sectional normalization."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_z(series: pd.Series, clip: float = 3.0) -> pd.Series:
    mean = float(series.mean())
    std = float(series.std(ddof=0))
    if std == 0.0 or np.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    z = (series - mean) / std
    return z.clip(-clip, clip)
