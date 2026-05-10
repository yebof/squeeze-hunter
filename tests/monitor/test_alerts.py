from unittest.mock import AsyncMock

import pytest

from squeeze_hunter.monitor.alerts import AlertSender, Severity


@pytest.mark.asyncio
async def test_telegram_sent_for_high_severity() -> None:
    sender = AlertSender(
        telegram_bot_token="t",
        telegram_chat_id="123",
        slack_webhook_url=None,
    )
    sender._send_telegram = AsyncMock()
    sender._send_slack = AsyncMock()
    await sender.send("kill-switch tripped: monthly_drawdown", severity=Severity.HIGH)
    assert sender._send_telegram.called
    assert not sender._send_slack.called


@pytest.mark.asyncio
async def test_slack_sent_for_low_severity() -> None:
    sender = AlertSender(
        telegram_bot_token=None,
        telegram_chat_id=None,
        slack_webhook_url="https://hooks.slack.com/x",
    )
    sender._send_telegram = AsyncMock()
    sender._send_slack = AsyncMock()
    await sender.send("daily P&L: +1.2%", severity=Severity.LOW)
    assert sender._send_slack.called
    assert not sender._send_telegram.called
