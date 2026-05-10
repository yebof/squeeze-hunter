from squeeze_hunter.risk.stops import StopState, evaluate_stops


def test_hard_stop_triggers_below_threshold() -> None:
    state = StopState(
        entry_price=100.0,
        peak_price=100.0,
        current_score=10.0,
        entry_score=10.0,
        bars_held=2,
        setup_type="CAR",
    )
    sig = evaluate_stops(state, current_price=87.0)
    assert sig.action == "exit"
    assert sig.reason == "hard_stop"


def test_trailing_stop_triggers_after_peak() -> None:
    state = StopState(
        entry_price=100.0,
        peak_price=140.0,
        current_score=8.0,
        entry_score=10.0,
        bars_held=3,
        setup_type="CAR",
    )
    sig = evaluate_stops(state, current_price=110.0)  # 21% from peak (CAR uses 20%)
    assert sig.action == "exit"
    assert sig.reason == "trailing_stop"


def test_time_stop_at_21_bars() -> None:
    state = StopState(
        entry_price=100.0,
        peak_price=110.0,
        current_score=10.0,
        entry_score=10.0,
        bars_held=21,
        setup_type="CAR",
    )
    sig = evaluate_stops(state, current_price=105.0)
    assert sig.action == "exit"
    assert sig.reason == "time_stop"


def test_signal_decay_halves_then_exits() -> None:
    state_half = StopState(
        entry_price=100.0,
        peak_price=110.0,
        current_score=4.5,
        entry_score=10.0,  # decayed >=50%
        bars_held=5,
        setup_type="GME",
    )
    sig = evaluate_stops(state_half, current_price=108.0)
    assert sig.action == "halve"
    assert sig.reason == "signal_decay_50"

    state_exit = StopState(
        entry_price=100.0,
        peak_price=110.0,
        current_score=2.0,
        entry_score=10.0,  # decayed >=75%
        bars_held=6,
        setup_type="GME",
    )
    sig = evaluate_stops(state_exit, current_price=108.0)
    assert sig.action == "exit"
    assert sig.reason == "signal_decay_75"


def test_gme_uses_25_pct_trailing() -> None:
    state = StopState(
        entry_price=100.0,
        peak_price=200.0,
        current_score=10.0,
        entry_score=10.0,
        bars_held=5,
        setup_type="GME",
    )
    sig = evaluate_stops(state, current_price=149.0)  # 25.5% from peak
    assert sig.action == "exit"
    assert sig.reason == "trailing_stop"
