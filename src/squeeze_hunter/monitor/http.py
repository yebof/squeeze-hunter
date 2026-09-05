"""Minimal stdlib HTTP endpoint for `GET /metrics` and `GET /health`.

Round-12: the spec (§7) and docker/prometheus.yml assume the runtime serves
these on :8080, but nothing ever started a server — Prometheus scraped an
empty port and the HealthSnapshot class had no caller. A ThreadingHTTPServer
on a daemon thread keeps this dependency-free and off the event loop.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING

from squeeze_hunter.logging_setup import get_logger
from squeeze_hunter.monitor.healthcheck import HealthSnapshot

if TYPE_CHECKING:
    from squeeze_hunter.runtime import RuntimeContext

log = get_logger("monitor.http")

# Reported when no critical data source has been refreshed yet (before the
# first in-session tick): large enough to read as "stale" in HealthSnapshot.
_NEVER_REFRESHED_S = 10**9


def health_snapshot(rc: RuntimeContext, now: datetime | None = None) -> HealthSnapshot:
    now = now or datetime.now(UTC)
    telemetry = rc.telemetry
    has_critical = any(src in telemetry.critical_sources for src in telemetry.data_freshness)
    age = telemetry.critical_data_stale_for_seconds(now) if has_critical else _NEVER_REFRESHED_S
    return HealthSnapshot(
        # Postgres is schema-only in Phase 3 (spec §8): nothing at runtime
        # reads or writes it, so it cannot be "down" from the loop's point of
        # view. Revisit when the Phase 4 persistence path lands.
        db_connected=True,
        broker_connected=bool(getattr(rc, "last_broker_healthy", False)),
        last_data_ingest_age_seconds=int(age),
        kill_switch_active=bool(rc.kill_switch_active),
    )


class MonitorServer:
    def __init__(self: MonitorServer, rc: RuntimeContext, host: str, port: int) -> None:
        runtime = rc

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self: _Handler) -> None:
                if self.path == "/metrics":
                    registry = runtime.metrics_registry
                    if registry is None:
                        self._reply(503, "text/plain", "metrics registry not ready\n")
                        return
                    self._reply(200, "text/plain; version=0.0.4; charset=utf-8", registry.render())
                elif self.path == "/health":
                    snap = health_snapshot(runtime)
                    body = snap.json()
                    self._reply(200 if snap._is_healthy() else 503, "application/json", body)
                else:
                    self._reply(404, "text/plain", "not found\n")

            def _reply(self: _Handler, status: int, content_type: str, body: str) -> None:
                raw = body.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self: _Handler, format: str, *args: object) -> None:
                # Keep the stdlib's per-request stderr chatter out of the
                # structured log stream.
                return

        self._httpd = ThreadingHTTPServer((host, port), _Handler)
        self._httpd.daemon_threads = True
        self.host = host
        self.port = int(self._httpd.server_address[1])
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="sh-monitor-http", daemon=True
        )

    def start(self: MonitorServer) -> None:
        self._thread.start()
        log.info("monitor_http_started", host=self.host, port=self.port)

    def stop(self: MonitorServer) -> None:
        # Round-13: BaseServer.shutdown() blocks until serve_forever() exits —
        # forever if serve_forever never started. Guard on the thread.
        if self._thread.is_alive():
            self._httpd.shutdown()
        self._httpd.server_close()
        log.info("monitor_http_stopped", port=self.port)


def start_monitor_server(rc: RuntimeContext, port: int, host: str = "127.0.0.1") -> MonitorServer:
    server = MonitorServer(rc, host=host, port=port)
    server.start()
    return server
