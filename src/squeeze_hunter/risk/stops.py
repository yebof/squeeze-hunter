"""Layered stops: hard / trailing / time / signal-decay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class StopState:
    entry_price: float
    peak_price: float
    current_score: float
    entry_score: float
    bars_held: int
    setup_type: str  # CAR, GME, Mixed


@dataclass(slots=True, frozen=True)
class StopSignal:
    action: Literal["hold", "halve", "exit"]
    reason: str | None = None


# R9.1: defaults retained for backwards compat with tests / ad-hoc callers.
# The values are also the defaults exposed through evaluate_stops kwargs below.
# Production callers (runner.py, lifecycle.py) now pass these via settings so
# the YAML `stops.trailing_car` / `trailing_gme` / `trailing_mixed` knobs are
# wired end-to-end. Trailing magnitudes are POSITIVE here (the comparison at
# `from_peak <= -trailing` flips the sign); callers reading a YAML value that
# may be stored negative should pass `abs(...)`.
_DEFAULT_TRAILING_CAR = 0.20
_DEFAULT_TRAILING_GME = 0.25
_DEFAULT_TRAILING_MIXED = 0.22


def evaluate_stops(
    state: StopState,
    current_price: float,
    *,
    hard_stop: float = -0.12,
    time_stop_bars: int = 21,
    signal_decay_halve: float = 0.50,
    signal_decay_exit: float = 0.75,
    trailing_car: float = _DEFAULT_TRAILING_CAR,
    trailing_gme: float = _DEFAULT_TRAILING_GME,
    trailing_mixed: float = _DEFAULT_TRAILING_MIXED,
) -> StopSignal:
    pnl_pct = (current_price - state.entry_price) / state.entry_price
    if pnl_pct <= hard_stop:
        return StopSignal("exit", "hard_stop")

    if state.setup_type == "CAR":
        trailing = trailing_car
    elif state.setup_type == "GME":
        trailing = trailing_gme
    else:  # Mixed (or any unrecognized type) gets the Mixed trailing
        trailing = trailing_mixed
    if state.peak_price > state.entry_price:
        from_peak = (current_price - state.peak_price) / state.peak_price
        if from_peak <= -trailing:
            return StopSignal("exit", "trailing_stop")

    if state.bars_held >= time_stop_bars:
        return StopSignal("exit", "time_stop")

    if state.entry_score > 0:
        decay = (state.entry_score - state.current_score) / state.entry_score
        if decay >= signal_decay_exit:
            return StopSignal("exit", "signal_decay_75")
        if decay >= signal_decay_halve:
            return StopSignal("halve", "signal_decay_50")

    return StopSignal("hold")
