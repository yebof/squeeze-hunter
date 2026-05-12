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


def test_trailing_stops_yaml_overrides_apply() -> None:
    """R9.1 regression: trailing-stop magnitudes must be tunable via the
    `trailing_car`/`trailing_gme`/`trailing_mixed` settings — they were
    previously hardcoded in `_TRAILING_BY_SETUP` and the YAML keys were
    silently dead code, violating the "one config file" design principle.
    """
    # Same state as the CAR trailing-stop test (110 / 140 = -21% from peak,
    # which fires at the default 20%). With trailing_car bumped to 30%,
    # this should NOT fire — the loosened threshold accepts the drawdown.
    state = StopState(
        entry_price=100.0,
        peak_price=140.0,
        current_score=8.0,
        entry_score=10.0,
        bars_held=3,
        setup_type="CAR",
    )
    sig = evaluate_stops(
        state,
        current_price=110.0,
        trailing_car=0.30,  # loosened from default 0.20
    )
    assert sig.action != "exit", (
        f"trailing_car kwarg ignored; got action={sig.action} reason={sig.reason}"
    )


def test_trailing_stops_yaml_override_tightens_too() -> None:
    """Symmetry check: tightening trailing_gme should fire earlier than default."""
    state = StopState(
        entry_price=100.0,
        peak_price=200.0,
        current_score=10.0,
        entry_score=10.0,
        bars_held=5,
        setup_type="GME",
    )
    # 180 from peak 200 = -10%. Default GME is 25% (won't fire). With
    # trailing_gme tightened to 8%, this should now fire.
    sig = evaluate_stops(state, current_price=180.0, trailing_gme=0.08)
    assert sig.action == "exit"
    assert sig.reason == "trailing_stop"


def test_trailing_mixed_override() -> None:
    """trailing_mixed kwarg controls the Mixed setup's trailing magnitude."""
    state = StopState(
        entry_price=100.0,
        peak_price=200.0,
        current_score=10.0,
        entry_score=10.0,
        bars_held=5,
        setup_type="Mixed",
    )
    # 180 / 200 = -10%. Default Mixed is 22% → no fire. Tighten to 8% → fire.
    sig = evaluate_stops(state, current_price=180.0, trailing_mixed=0.08)
    assert sig.action == "exit"
    assert sig.reason == "trailing_stop"
