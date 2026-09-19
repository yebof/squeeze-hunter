"""P6 — stale critical data blocks automatic entries; the EOD ingest job is wired."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from squeeze_hunter.cli import _build_runtime_callbacks
from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.ingest.eod import EodIngestReport
from squeeze_hunter.ingest.freshness import record_freshness
from squeeze_hunter.monitor.alerts import AlertSender
from squeeze_hunter.runtime import RuntimeContext
from tests.runtime.test_session_clamp import _seed

_PREMARKET = datetime(2026, 6, 10, 12, 0, tzinfo=UTC)


async def _rc(tmp_path: Path) -> RuntimeContext:
    cache = ParquetCache(root=tmp_path)
    _seed(cache)
    settings = Settings()
    settings.score.weights = {"f6_bollinger_breakout": 1.0, "f7_volume_spike": 1.0}
    settings.execution.auto_enter = True
    rc = RuntimeContext(cache=cache, settings=settings, tickers=["GME"], mode="sim")
    await rc.setup()
    rc.last_candidates = pd.DataFrame(
        [{"ticker": "GME", "score": 99.0, "setup_type": "CAR", "rank": 1, "as_of": _PREMARKET}]
    )
    return rc


@pytest.mark.asyncio
async def test_stale_bars_block_planned_entries_and_alert(tmp_path: Path) -> None:
    rc = await _rc(tmp_path)
    rc.alerts = AlertSender(telegram_bot_token="t", telegram_chat_id="c", slack_webhook_url=None)
    rc.alerts._send_telegram = AsyncMock()  # type: ignore[method-assign]
    record_freshness(tmp_path, "bars", as_of=_PREMARKET - timedelta(days=9), rows=1, now=_PREMARKET)
    await rc.premarket_verify(now=_PREMARKET)
    assert rc.planned_entries == []
    rc.alerts._send_telegram.assert_awaited_once()
    assert "bars" in rc.alerts._send_telegram.await_args.args[0]


@pytest.mark.asyncio
async def test_unknown_freshness_blocks_when_required(tmp_path: Path) -> None:
    rc = await _rc(tmp_path)  # no _freshness.json at all
    await rc.premarket_verify(now=_PREMARKET)
    assert rc.planned_entries == []


@pytest.mark.asyncio
async def test_fresh_data_lets_entries_through(tmp_path: Path) -> None:
    rc = await _rc(tmp_path)
    record_freshness(
        tmp_path, "bars", as_of=_PREMARKET - timedelta(hours=14), rows=1, now=_PREMARKET
    )
    await rc.premarket_verify(now=_PREMARKET)
    assert [p.ticker for p in rc.planned_entries] == ["GME"]


@pytest.mark.asyncio
async def test_freshness_gate_can_be_disabled(tmp_path: Path) -> None:
    rc = await _rc(tmp_path)
    rc.settings.data.require_fresh_for_entries = False
    await rc.premarket_verify(now=_PREMARKET)
    assert [p.ticker for p in rc.planned_entries] == ["GME"]


@pytest.mark.asyncio
async def test_ingest_eod_job_runs_the_pipeline_and_is_wired(tmp_path: Path) -> None:
    rc = await _rc(tmp_path)
    report = EodIngestReport(
        bars_updated={"GME": 1}, bars_failed=[], finra="skipped:fresh", earnings="skipped:no_key"
    )
    with patch("squeeze_hunter.runtime.ingest_eod", AsyncMock(return_value=report)) as job:
        assert await rc.ingest_eod_safe(now=_PREMARKET)
    job.assert_awaited_once()
    cbs = _build_runtime_callbacks(rc)
    assert cbs["ingest_eod"] is not None
