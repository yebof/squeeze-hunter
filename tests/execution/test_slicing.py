from datetime import UTC, datetime, timedelta

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
