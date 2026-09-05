"""Multi-channel alerting. Telegram = high severity. Slack = low severity. Email later."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import httpx

from squeeze_hunter.logging_setup import get_logger

log = get_logger("monitor.alerts")


class Severity(Enum):
    HIGH = "high"  # Telegram (mobile push)
    LOW = "low"  # Slack (work hours digest)


@dataclass
class AlertSender:
    telegram_bot_token: str | None
    telegram_chat_id: str | None
    slack_webhook_url: str | None
    timeout_s: float = 10.0

    async def send(self: AlertSender, text: str, severity: Severity) -> None:
        if severity == Severity.HIGH:
            if self.telegram_bot_token and self.telegram_chat_id:
                await self._send_telegram(text)
            elif self.slack_webhook_url:
                # Round-13: a Slack-only operator used to lose every HIGH alert
                # (killswitch trips) to a WARNING log line. Degrade to Slack.
                log.warning("telegram_not_configured_falling_back_to_slack")
                await self._send_slack(text)
            else:
                log.warning("telegram_not_configured", text=text)
        else:
            if self.slack_webhook_url:
                await self._send_slack(text)
            else:
                log.warning("slack_not_configured", text=text)

    async def _send_telegram(self: AlertSender, text: str) -> None:
        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(url, json={"chat_id": self.telegram_chat_id, "text": text})
            if r.status_code >= 400:
                log.error("telegram_send_failed", status=r.status_code, body=r.text)

    async def _send_slack(self: AlertSender, text: str) -> None:
        # R4.8: explicit None check rather than `assert` — `assert` is stripped
        # by `python -O` and would yield a cryptic httpx.InvalidURL instead of
        # a clear ValueError in a hot path.
        if self.slack_webhook_url is None:
            raise ValueError("_send_slack called without a configured webhook")
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(self.slack_webhook_url, json={"text": text})
            if r.status_code >= 400:
                log.error("slack_send_failed", status=r.status_code, body=r.text)
