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


# Per-setup-type priors derived from the design spec Section 6.
# Each must produce a positive raw Kelly to avoid forcing the runtime fallback.
# CAR payoff bumped from spec's 3.0 to 3.5: at 3.0 raw Kelly = (0.25*3.0-0.75)/3.0 = 0
# (exactly zero), which clips to 0. At 3.5: (0.25*3.5-0.75)/3.5 = 0.025 (positive).
# GME: (0.15*8.0-0.85)/8.0 = 0.04375 (positive, no adjustment needed).
# Mixed: average of CAR/GME = (0.20*5.5-0.80)/5.5 = 0.0545 (positive).
_PRIORS_BY_SETUP: dict[str, tuple[float, float]] = {
    # (win_rate, payoff)
    "CAR": (0.25, 3.5),  # bumped from spec's 3.0 so raw Kelly is positive (0.025)
    "GME": (0.15, 8.0),  # raw Kelly = 0.04375
    "Mixed": (0.20, 5.5),  # raw Kelly = 0.0545
}


def kelly_priors_for_setup(setup_type: str) -> KellyParams:
    """Return KellyParams with per-setup-type priors.

    Falls back to Mixed priors for unknown setup_type so we always have a
    positive raw Kelly and avoid the flat 4% fallback.
    """
    pwr, ppay = _PRIORS_BY_SETUP.get(setup_type, _PRIORS_BY_SETUP["Mixed"])
    return KellyParams(
        prior_win_rate=pwr,
        prior_payoff=ppay,
        prior_n=30,
        fraction=0.20,
        cap=0.08,
    )
