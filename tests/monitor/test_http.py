"""Round-12: /metrics and /health are actually served (Prometheus scraped nothing)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from squeeze_hunter.monitor.http import start_monitor_server
from squeeze_hunter.monitor.metrics import MetricsRegistry
from squeeze_hunter.runtime import PortfolioTelemetry


def _stub_runtime(kill: bool = False) -> MagicMock:
    rc = MagicMock()
    rc.metrics_registry = MetricsRegistry()
    rc.metrics_registry.set_equity(123.0)
    rc.telemetry = PortfolioTelemetry()
    rc.telemetry.record_data_freshness("ibkr_quotes", datetime.now(UTC))
    rc.last_broker_healthy = True
    rc.kill_switch_active = kill
    return rc


def test_monitor_server_serves_metrics_and_health() -> None:
    server = start_monitor_server(_stub_runtime(), port=0, host="127.0.0.1")
    try:
        base = f"http://127.0.0.1:{server.port}"
        metrics = urllib.request.urlopen(base + "/metrics", timeout=5).read().decode()
        assert "sh_equity_usd 123.0" in metrics

        health = json.loads(urllib.request.urlopen(base + "/health", timeout=5).read())
        assert health["ok"] is True
        assert health["kill_switch_active"] is False

        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(base + "/nope", timeout=5)
        assert exc.value.code == 404
    finally:
        server.stop()


def test_health_reports_503_while_killswitch_is_active() -> None:
    server = start_monitor_server(_stub_runtime(kill=True), port=0, host="127.0.0.1")
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(f"http://127.0.0.1:{server.port}/health", timeout=5)
        assert exc.value.code == 503
        body = json.loads(exc.value.read())
        assert body["kill_switch_active"] is True
        assert body["ok"] is False
    finally:
        server.stop()
