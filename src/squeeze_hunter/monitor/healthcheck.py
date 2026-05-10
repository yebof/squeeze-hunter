"""Health check endpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class HealthSnapshot:
    db_connected: bool
    broker_connected: bool
    last_data_ingest_age_seconds: int
    kill_switch_active: bool

    def json(self: HealthSnapshot) -> str:
        return json.dumps(
            {
                "db_connected": self.db_connected,
                "broker_connected": self.broker_connected,
                "last_data_ingest_age_seconds": self.last_data_ingest_age_seconds,
                "kill_switch_active": self.kill_switch_active,
                "ok": self._is_healthy(),
            }
        )

    def _is_healthy(self: HealthSnapshot) -> bool:
        return (
            self.db_connected
            and self.broker_connected
            and self.last_data_ingest_age_seconds < 60 * 60 * 24
        )
