"""HealthSnapshot regression tests.

R9.4: a tripped killswitch means the system is actively halted; the health
endpoint MUST reflect that as unhealthy. Prior behavior returned ok=True even
when kill_switch_active=True, hiding the halt from monitoring/alerts.
"""

from __future__ import annotations

import json

from squeeze_hunter.monitor.healthcheck import HealthSnapshot


def test_healthy_when_all_good() -> None:
    snap = HealthSnapshot(
        db_connected=True,
        broker_connected=True,
        last_data_ingest_age_seconds=60,
        kill_switch_active=False,
    )
    assert json.loads(snap.json())["ok"] is True


def test_unhealthy_when_db_down() -> None:
    snap = HealthSnapshot(
        db_connected=False,
        broker_connected=True,
        last_data_ingest_age_seconds=60,
        kill_switch_active=False,
    )
    assert json.loads(snap.json())["ok"] is False


def test_unhealthy_when_broker_disconnected() -> None:
    snap = HealthSnapshot(
        db_connected=True,
        broker_connected=False,
        last_data_ingest_age_seconds=60,
        kill_switch_active=False,
    )
    assert json.loads(snap.json())["ok"] is False


def test_unhealthy_when_killswitch_active() -> None:
    """R9.4 regression: a tripped killswitch means the system is halted; that
    is NOT a healthy state, even if db + broker + data are all fine."""
    snap = HealthSnapshot(
        db_connected=True,
        broker_connected=True,
        last_data_ingest_age_seconds=60,
        kill_switch_active=True,
    )
    assert json.loads(snap.json())["ok"] is False
