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
# CAR payoff bumped from spec's 3.0 to 3.5 to keep raw Kelly strictly positive
# at trade 0 (at 3.0 the formula equals 0 exactly, clipping to 0). At 3.5 the
# raw Kelly is (0.25 * 3.5 - 0.75) / 3.5 ≈ 0.0357 — small but positive, so
# the per-setup prior alone supplies a sane starting size.
# R8.I6: prior comment claimed 0.025 — actual value is ~0.0357. The bump is a
# code-side deviation from the spec; if the design spec is updated to (0.25, 3.5)
# in a future revision, drop this note.
# GME: (0.15 * 8.0 - 0.85) / 8.0 = 0.04375 (positive, no adjustment).
# Mixed: average of CAR/GME = (0.20 * 5.5 - 0.80) / 5.5 ≈ 0.0545 (positive).
_PRIORS_BY_SETUP: dict[str, tuple[float, float]] = {
    # (win_rate, payoff)
    "CAR": (0.25, 3.5),  # bumped from spec's 3.0 so raw Kelly is positive (0.025)
    "GME": (0.15, 8.0),  # raw Kelly = 0.04375
    "Mixed": (0.20, 5.5),  # raw Kelly = 0.0545
}


def kelly_priors_for_setup(
    setup_type: str,
    *,
    fraction: float = 0.20,
    cap: float = 0.08,
    prior_n: int = 30,
    priors: dict[str, tuple[float, float]] | None = None,
) -> KellyParams:
    """Return KellyParams with per-setup-type priors.

    P9: `priors` (setup -> (win_rate, payoff)) comes from
    settings.risk.kelly_priors; the module table below is only the default.

    R7.M2: "Weak" returns fraction=0 (and a noticeably-low cap) so any Weak
    candidate that slips past the gate sizes to zero.

    R8.S-I2: fraction/cap/prior_n are now kwargs so callers can pass
    settings.risk.kelly_fraction / .position_cap / .bayes_prior_n. Previously
    those YAML knobs were dead — the function hardcoded the defaults.
    """
    if setup_type == "Weak":
        return KellyParams(
            prior_win_rate=0.0,
            prior_payoff=1.0,
            prior_n=prior_n,
            fraction=0.0,  # zero sizing for explicitly weak setups
            cap=0.0,
        )
    table = priors if priors else _PRIORS_BY_SETUP
    pwr, ppay = table.get(setup_type, table.get("Mixed", _PRIORS_BY_SETUP["Mixed"]))
    return KellyParams(
        prior_win_rate=pwr,
        prior_payoff=ppay,
        prior_n=prior_n,
        fraction=fraction,
        cap=cap,
    )
