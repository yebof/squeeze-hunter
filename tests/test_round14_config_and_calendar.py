"""P5 / P9 of the architecture-hardening plan: one calendar module, no
risk tunables left in code."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from squeeze_hunter.config import Settings, load_settings
from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.risk.kelly import kelly_priors_for_setup
from squeeze_hunter.runtime import RuntimeContext
from squeeze_hunter.trading_calendar import (
    is_regular_session,
    is_trading_day,
    next_session,
    session_open_utc,
    trading_sessions,
)
from tests.runtime.test_session_clamp import _seed


def test_calendar_module_is_the_single_source_of_truth() -> None:
    assert not is_trading_day(date(2024, 3, 29))  # Good Friday
    assert is_trading_day(date(2024, 10, 14))  # Columbus Day: open
    assert next_session(date(2025, 1, 3)) == date(2025, 1, 6)  # Fri → Mon
    assert next_session(date(2024, 3, 28)) == date(2024, 4, 1)  # Thu → Mon over Good Friday
    days = trading_sessions(datetime(2024, 3, 25, tzinfo=UTC), datetime(2024, 4, 1, tzinfo=UTC))
    assert [d.date() for d in days] == [
        date(2024, 3, 25),
        date(2024, 3, 26),
        date(2024, 3, 27),
        date(2024, 3, 28),
        date(2024, 4, 1),
    ]
    assert is_regular_session(datetime(2026, 5, 14, 14, 0, tzinfo=UTC))
    assert not is_regular_session(datetime(2026, 5, 16, 14, 0, tzinfo=UTC))  # Saturday
    assert session_open_utc(datetime(2026, 5, 14, 14, 0, tzinfo=UTC)) == datetime(
        2026, 5, 14, 13, 30, tzinfo=UTC
    )


def test_example_yaml_carries_every_risk_tunable() -> None:
    yaml_path = Path(__file__).resolve().parents[1] / "config" / "settings.example.yml"
    s = load_settings(yaml_path)
    assert s.risk.kelly_priors["CAR"].win_rate == 0.25
    assert s.risk.kelly_priors["GME"].payoff == 8.0
    assert s.risk.killswitch.broker_outage_max_seconds == 300
    assert s.risk.killswitch.cooldown_days == 7
    assert s.risk.gates.min_adv20_multiple == 100.0
    # YAML and code defaults agree.
    assert s.risk.killswitch == Settings().risk.killswitch
    assert s.risk.gates == Settings().risk.gates
    assert s.risk.kelly_priors == Settings().risk.kelly_priors


def test_kelly_priors_come_from_settings() -> None:
    params = kelly_priors_for_setup("CAR", priors={"CAR": (0.30, 4.0), "Mixed": (0.2, 5.5)})
    assert params.prior_win_rate == 0.30
    assert params.prior_payoff == 4.0
    # Unknown setup falls back to the Mixed prior of the SAME table.
    assert kelly_priors_for_setup("Odd", priors={"Mixed": (0.21, 5.0)}).prior_payoff == 5.0


@pytest.mark.asyncio
async def test_runtime_killswitch_thresholds_come_from_settings(tmp_path: Path) -> None:
    """A 60 s in-session outage trips when the YAML budget is 30 s."""
    cache = ParquetCache(root=tmp_path)
    _seed(cache)
    settings = Settings()
    settings.score.weights = {"f6_bollinger_breakout": 1.0, "f7_volume_spike": 1.0}
    settings.risk.killswitch.broker_outage_max_seconds = 30
    rc = RuntimeContext(cache=cache, settings=settings, tickers=["GME"], mode="sim")
    await rc.setup()
    assert rc.broker is not None
    rc.telemetry.record_broker_heartbeat(datetime(2026, 5, 13, 19, 59, tzinfo=UTC))
    rc.broker.health = AsyncMock(side_effect=ConnectionError("down"))  # type: ignore[method-assign]
    await rc.tick(now=datetime(2026, 5, 14, 13, 31, tzinfo=UTC))
    assert rc.kill_switch_active
    assert rc._kill_reason == "broker_outage"
