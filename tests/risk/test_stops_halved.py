"""Round-12: the signal-decay halve is a one-shot per position."""

from __future__ import annotations

from squeeze_hunter.risk.stops import StopState, evaluate_stops


def _state(current_score: float, halved: bool) -> StopState:
    return StopState(
        entry_price=100.0,
        peak_price=100.0,
        current_score=current_score,
        entry_score=10.0,
        bars_held=2,
        setup_type="CAR",
        halved=halved,
    )


def test_halve_band_returns_hold_once_already_halved() -> None:
    assert evaluate_stops(_state(4.0, halved=False), current_price=100.0).action == "halve"
    assert evaluate_stops(_state(4.0, halved=True), current_price=100.0).action == "hold"


def test_exit_band_still_exits_after_a_halve() -> None:
    sig = evaluate_stops(_state(2.0, halved=True), current_price=100.0)
    assert sig.action == "exit"
    assert sig.reason == "signal_decay_75"
