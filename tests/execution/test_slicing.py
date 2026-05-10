from datetime import UTC, datetime, timedelta

import pytest

from squeeze_hunter.execution.slicing import build_twap_plan


def test_plan_starts_at_open_plus_5min_default() -> None:
    market_open = datetime(2026, 5, 14, 13, 30, tzinfo=UTC)  # 09:30 ET in May
    plan = build_twap_plan(
        total_qty=600,
        reference_price=20.0,
        market_open=market_open,
        n_slices=6,
        window_minutes=20,
        slice_offset_minutes=5,
    )
    assert plan.slices[0].submit_at >= market_open + timedelta(minutes=5)
    assert len(plan.slices) == 6
    assert sum(s.qty for s in plan.slices) == 600


def test_aggression_escalates_after_threshold() -> None:
    plan = build_twap_plan(
        total_qty=600,
        reference_price=20.0,
        market_open=datetime(2026, 5, 14, 13, 30, tzinfo=UTC),
        n_slices=6,
        window_minutes=20,
        slice_offset_minutes=5,
    )
    assert plan.slices[0].limit_price < plan.slices[-1].limit_price


def test_plan_remainder_goes_to_last_slice() -> None:
    plan = build_twap_plan(
        total_qty=601,
        reference_price=20.0,
        market_open=datetime(2026, 5, 14, 13, 30, tzinfo=UTC),
        n_slices=6,
        window_minutes=20,
        slice_offset_minutes=5,
    )
    assert sum(s.qty for s in plan.slices) == 601
    assert plan.slices[-1].qty == 101


def test_aggression_schedule_starts_passive() -> None:
    """Default schedule starts at 5bps (passive)."""
    from squeeze_hunter.execution.slicing import default_aggression_schedule

    sched = default_aggression_schedule(n_slices=6)
    assert sched[0] == pytest.approx(5.0)


def test_aggression_schedule_ends_at_30_bps() -> None:
    from squeeze_hunter.execution.slicing import default_aggression_schedule

    sched = default_aggression_schedule(n_slices=6)
    assert sched[-1] == pytest.approx(30.0)


def test_aggression_escalation_for_lagging_fills() -> None:
    """When fills are lagging, escalation function bumps aggression."""
    from squeeze_hunter.execution.slicing import escalate_aggression

    # 5 slices submitted, 1 filled → 80% unfilled → marketable territory
    out = escalate_aggression(
        base_bps=5.0,
        slices_submitted=5,
        slices_filled=1,
        marketable_bps=50.0,
        marketable_unfilled_threshold=0.80,
    )
    assert out >= 50.0  # marketable

    # 3 submitted, 2 filled → 33% unfilled → no escalation needed
    out = escalate_aggression(
        base_bps=5.0,
        slices_submitted=3,
        slices_filled=2,
        marketable_bps=50.0,
        marketable_unfilled_threshold=0.80,
    )
    assert out == pytest.approx(5.0)
