from datetime import UTC, date, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from squeeze_hunter.store.models import (
    Base,
    KillSwitchEvent,
    Order,
    Position,
    SignalDaily,
    UniverseRow,
)


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def test_universe_row_roundtrip(session: Session) -> None:
    session.add(
        UniverseRow(
            ticker="GME",
            as_of=date(2024, 5, 13),
            market_cap_usd=10_000_000_000,
            close_price=20.0,
            included=True,
        )
    )
    session.commit()
    row = session.query(UniverseRow).one()
    assert row.ticker == "GME"
    assert row.included


def test_signal_daily_roundtrip(session: Session) -> None:
    session.add(
        SignalDaily(
            as_of=date(2024, 5, 13),
            ticker="GME",
            factor_name="f1_si_pct",
            raw_value=15.0,
            z_score=3.2,
        )
    )
    session.commit()
    assert session.query(SignalDaily).count() == 1


def test_position_lifecycle(session: Session) -> None:
    p = Position(
        ticker="GME",
        opened_at=datetime(2024, 5, 13, 14, 0, tzinfo=UTC),
        side="long",
        quantity=100,
        avg_price=18.0,
        setup_type="GME",
        instrument="stock",
        status="open",
    )
    session.add(p)
    session.commit()
    p.status = "closed"
    p.closed_at = datetime(2024, 5, 17, 20, 0, tzinfo=UTC)
    session.commit()
    assert session.query(Position).filter_by(status="closed").count() == 1


def test_kill_switch_event(session: Session) -> None:
    session.add(
        KillSwitchEvent(
            triggered_at=datetime(2024, 5, 13, 19, 0, tzinfo=UTC),
            reason="monthly_drawdown",
            details_json={"drawdown_pct": -0.12},
        )
    )
    session.commit()
    assert session.query(KillSwitchEvent).count() == 1


def test_order_states(session: Session) -> None:
    session.add(
        Order(
            client_order_id="abc-1",
            ticker="GME",
            side="buy",
            instrument="stock",
            quantity=100,
            limit_price=18.5,
            status="pending",
            submitted_at=datetime(2024, 5, 13, 13, 35, tzinfo=UTC),
        )
    )
    session.commit()
    assert session.query(Order).filter_by(status="pending").count() == 1
