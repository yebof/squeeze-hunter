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
    periods_per_year: int = 252,
) -> float:
    """Bonferroni-style penalty: returns the probability that the true SR > 0
    after accounting for `n_trials` parameter sets evaluated on `n_obs` daily returns.

    `observed_sr` is the ANNUALIZED Sharpe (as produced by `metrics.sharpe`,
    which multiplies the per-period mean/std ratio by sqrt(252)).

    The expected maximum SR under the null (SR0) is the expected maximum of
    `n_trials` iid N(0,1) draws (Euler-Mascheroni approximation), scaled into
    per-period SR units. The observed SR is tested against this benchmark,
    deflated by the SR standard error. A higher `n_trials` raises SR0 and thus
    reduces the returned confidence value.
    """
    # R10.7: with too few observations or no trials, the deflation formula has
    # nothing meaningful to compute. Returning the raw observed_sr would let
    # Gate 1 silently "PASS" a degenerate short-window run. Return 0.0 so the
    # caller's threshold check (gate1.deflated_sharpe_min ≥ 0) trips it as
    # "insufficient evidence."
    if n_obs < 20 or n_trials < 1:
        return 0.0
    # R11: de-annualize first. `observed_sr` arrives annualized, but the SR
    # standard error and the expected-max benchmark below are in PER-PERIOD SR
    # units. Feeding the annualized SR straight in — and dropping the `* sr_std`
    # scaling on the benchmark (the shipped code did both) — made the z-score
    # dimensionally inconsistent and turned Gate 1's pass/fail boundary into a
    # units artifact (~annualized 2.0), rejecting good strategies at 1.0-1.5.
    sr = observed_sr / sqrt(periods_per_year)
    # Clamp probabilities to (0, 1) so ppf never receives 0 or 1.
    p1 = max(1e-12, min(1 - 1e-12, 1 - 1 / n_trials))
    p2 = max(1e-12, min(1 - 1e-12, 1 - 1 / (n_trials * 2.71828)))
    # Expected maximum of n_trials iid N(0,1) draws, in standardized units.
    e_max = (1 - 0.5772) * norm.ppf(p1) + 0.5772 * norm.ppf(p2)
    # Standard error of the per-period SR estimate over n_obs returns.
    sr_std = sqrt((1 - skew * sr + (kurtosis - 1) / 4 * sr**2) / (n_obs - 1))
    # SR0 = e_max * sr_std lives in per-period SR units. z-score: how many
    # standard errors the observed per-period SR sits above that benchmark.
    z = (sr - e_max * sr_std) / max(sr_std, 1e-9)
    return float(norm.cdf(z))
