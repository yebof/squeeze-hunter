"""Deflated Sharpe Ratio (López de Prado)."""

from __future__ import annotations

from math import sqrt

from scipy.stats import norm  # type: ignore[import-untyped]


def deflated_sharpe(
    observed_sr: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Bonferroni-style penalty: returns the probability that the true SR > 0
    after accounting for `n_trials` parameter sets evaluated on `n_obs` daily returns.

    The expected maximum SR under the null (E[SR*]) is computed as the expected
    maximum of `n_trials` iid N(0,1) draws (Euler-Mascheroni approximation).
    The observed SR is then tested against this benchmark, deflated by the SR
    standard error.  A higher `n_trials` raises the E[SR*] benchmark and thus
    reduces the returned confidence value.
    """
    # R10.7: with too few observations or no trials, the deflation formula has
    # nothing meaningful to compute. Returning the raw observed_sr would let
    # Gate 1 silently "PASS" a degenerate short-window run. Return 0.0 so the
    # caller's threshold check (gate1.deflated_sharpe_min ≥ 0) trips it as
    # "insufficient evidence."
    if n_obs < 20 or n_trials < 1:
        return 0.0
    # Clamp probabilities to (0, 1) so ppf never receives 0 or 1.
    p1 = max(1e-12, min(1 - 1e-12, 1 - 1 / n_trials))
    p2 = max(1e-12, min(1 - 1e-12, 1 - 1 / (n_trials * 2.71828)))
    e_max = (1 - 0.5772) * norm.ppf(p1) + 0.5772 * norm.ppf(p2)
    sr_std = sqrt((1 - skew * observed_sr + (kurtosis - 1) / 4 * observed_sr**2) / (n_obs - 1))
    # z-score: how many standard errors above the expected-max benchmark.
    z = (observed_sr - e_max) / max(sr_std, 1e-9)
    return float(norm.cdf(z))
