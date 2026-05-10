"""Fractional Kelly with Bayesian shrinkage on observed win-rate / payoff."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class KellyParams:
    prior_win_rate: float = 0.20
    prior_payoff: float = 2.0
    prior_n: int = 30
    fraction: float = 0.20
    cap: float = 0.08


def kelly_position_pct(
    observed_wins: int,
    observed_trades: int,
    observed_avg_payoff: float,
    params: KellyParams,
) -> float:
    n = observed_trades
    weight = n / (n + params.prior_n) if (n + params.prior_n) > 0 else 0.0
    p_obs = (observed_wins / n) if n > 0 else params.prior_win_rate
    b_obs = observed_avg_payoff if n > 0 else params.prior_payoff
    p = weight * p_obs + (1 - weight) * params.prior_win_rate
    b = weight * b_obs + (1 - weight) * params.prior_payoff
    if b <= 0:
        return 0.0
    raw = (p * b - (1 - p)) / b
    sized = max(0.0, params.fraction * raw)
    return min(sized, params.cap)
