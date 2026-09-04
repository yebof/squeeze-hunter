"""Round-12: killswitch trips push a HIGH-severity alert (AlertSender was dead code)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from squeeze_hunter.monitor.alerts import AlertSender
from tests.runtime.test_session_clamp import _runtime


@pytest.mark.asyncio
async def test_killswitch_trip_sends_high_severity_alert(tmp_path: Path) -> None:
    rc = await _runtime(tmp_path)
    assert rc.broker is not None
    rc.alerts = AlertSender(telegram_bot_token="t", telegram_chat_id="c", slack_webhook_url=None)
    rc.alerts._send_telegram = AsyncMock()  # type: ignore[method-assign]

    yesterday_close = datetime(2026, 5, 13, 19, 59, tzinfo=UTC)
    rc.telemetry.record_broker_heartbeat(yesterday_close)
    rc.telemetry.record_data_freshness("ibkr_quotes", yesterday_close)
    rc.broker.health = AsyncMock(side_effect=ConnectionError("tws down"))  # type: ignore[method-assign]

    await rc.tick(now=datetime(2026, 5, 14, 13, 31, tzinfo=UTC))
    rc.alerts._send_telegram.assert_not_called()

    await rc.tick(now=datetime(2026, 5, 14, 13, 36, tzinfo=UTC))
    assert rc.kill_switch_active
    rc.alerts._send_telegram.assert_awaited_once()
    text = rc.alerts._send_telegram.await_args.args[0]
    assert "broker_outage" in text

    # Sticky window: no repeat alert on the next tick.
    await rc.tick(now=datetime(2026, 5, 14, 13, 37, tzinfo=UTC))
    rc.alerts._send_telegram.assert_awaited_once()


@pytest.mark.asyncio
async def test_setup_builds_alert_sender_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "c")
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    rc = await _runtime(tmp_path)
    assert rc.alerts is not None
    assert rc.alerts.telegram_chat_id == "c"
    assert rc.alerts.slack_webhook_url is None


@pytest.mark.asyncio
async def test_setup_without_alert_env_leaves_alerts_unconfigured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "SLACK_WEBHOOK_URL"):
        monkeypatch.delenv(key, raising=False)
    rc = await _runtime(tmp_path)
    assert rc.alerts is None
