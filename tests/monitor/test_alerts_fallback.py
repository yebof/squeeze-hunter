"""Round-13: a HIGH-severity alert must reach Slack when Telegram is not set."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from squeeze_hunter.monitor.alerts import AlertSender, Severity


@pytest.mark.asyncio
async def test_high_severity_falls_back_to_slack() -> None:
    sender = AlertSender(
        telegram_bot_token=None, telegram_chat_id=None, slack_webhook_url="https://hooks.slack/x"
    )
    sender._send_slack = AsyncMock()  # type: ignore[method-assign]
    await sender.send("killswitch tripped: broker_outage", severity=Severity.HIGH)
    sender._send_slack.assert_awaited_once()
