"""Random-shuffle test for entry timing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats  # type: ignore[import-untyped]


def _trend_quality(pct_rets: np.ndarray) -> float:
    """OLS slope / standard-error of the log-equity trend.

    This statistic is order-sensitive (unlike plain Sharpe which is invariant
    to return permutation).  A strategy with consistent, well-timed entries
    produces a smooth equity staircase with small residuals → high slope/stderr.
    """
    eq = np.cumprod(1.0 + pct_rets)
    x = np.arange(len(eq), dtype=float)
    result = scipy_stats.linregress(x, np.log(np.maximum(eq, 1e-10)))
    return result.slope / max(float(result.stderr), 1e-9)


def random_shuffle_pvalue(equity: pd.Series, *, n_permutations: int = 200, seed: int = 0) -> float:
    """Permute daily returns `n_permutations` times and measure equity trend quality.

    Returns the fraction of permutations whose trend-quality score is ≥ the
    observed score.  A *low* p-value (e.g. < 0.05) means the observed entry
    timing produces a significantly smoother equity path than random timing
    would, suggesting a genuine timing edge rather than luck.

    The trend-quality metric is the OLS slope / standard-error of the
    log-equity trend, which is sensitive to the order of returns (unlike the
    plain Sharpe ratio, which is invariant to permutation).
    """
    rets = equity.pct_change().dropna().values
    if len(rets) < 5 or rets.std() == 0:
        return 1.0
    obs_score = _trend_quality(rets)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_permutations):
        perm = rng.permutation(rets)
        if _trend_quality(perm) >= obs_score:
            hits += 1
    return hits / n_permutations
