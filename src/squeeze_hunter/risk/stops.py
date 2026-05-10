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


_TRAILING_BY_SETUP: dict[str, float] = {"CAR": 0.20, "GME": 0.25, "Mixed": 0.22}


def evaluate_stops(
    state: StopState,
    current_price: float,
    *,
    hard_stop: float = -0.12,
    time_stop_bars: int = 21,
    signal_decay_halve: float = 0.50,
    signal_decay_exit: float = 0.75,
) -> StopSignal:
    pnl_pct = (current_price - state.entry_price) / state.entry_price
    if pnl_pct <= hard_stop:
        return StopSignal("exit", "hard_stop")

    trailing = _TRAILING_BY_SETUP.get(state.setup_type, 0.22)
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
