"""P10 — the deployment files describe a real, restartable service."""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_compose_runs_the_app_with_restart_and_healthcheck() -> None:
    compose = yaml.safe_load((ROOT / "docker" / "compose.yml").read_text())
    svc = compose["services"]["squeeze-hunter"]
    assert svc["restart"] == "unless-stopped"
    assert svc["build"]["dockerfile"] == "docker/Dockerfile"
    assert "/health" in " ".join(map(str, svc["healthcheck"]["test"]))
    assert any(v.startswith("./data:") or v.startswith("../data:") for v in svc["volumes"])
    assert ".env" in " ".join(svc["env_file"])
    # The monitor endpoint must bind all interfaces INSIDE the container so
    # Prometheus can scrape it over the compose network.
    assert svc["environment"]["SH_MONITOR__HTTP_HOST"] == "0.0.0.0"
    assert "ports" not in svc, "the unauthenticated /metrics endpoint must not be published"


def test_prometheus_scrapes_the_app_service() -> None:
    prom = yaml.safe_load((ROOT / "docker" / "prometheus.yml").read_text())
    targets = prom["scrape_configs"][0]["static_configs"][0]["targets"]
    assert "squeeze-hunter:8080" in targets


def test_backup_script_is_executable_and_prunes() -> None:
    script = ROOT / "scripts" / "backup.sh"
    assert os.access(script, os.X_OK)
    text = script.read_text()
    assert "data/state" in text
    assert "data/parquet" in text
    assert re.search(r"-mtime", text)
    assert "set -euo pipefail" in text


def test_runbook_describes_what_exists() -> None:
    runbook = (ROOT / "docs" / "runbooks" / "disaster-recovery.md").read_text()
    assert "scripts/backup.sh" in runbook
    assert "data/state" in runbook
    assert "already automated" not in runbook
