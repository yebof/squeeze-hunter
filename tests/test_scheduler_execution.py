"""Round-12 regression: scheduler callbacks must actually run on the event loop.

`_build_runtime_callbacks` used to register plain *sync* lambdas that called
`asyncio.create_task`. APScheduler's AsyncIOExecutor runs non-coroutine job
functions in a worker thread — where there is no running event loop — so every
paper/live job died with `RuntimeError: no running event loop` and the runtime
never ticked. The wiring tests only checked that the dict had keys.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger

from squeeze_hunter.cli import _build_runtime_callbacks
from squeeze_hunter.scheduler import build_scheduler, list_job_specs


@pytest.mark.asyncio
async def test_scheduler_executes_runtime_callbacks_on_event_loop() -> None:
    rc = MagicMock()
    rc.tick_safe = AsyncMock(return_value=True)
    pending: set[asyncio.Task[Any]] = set()
    cbs = _build_runtime_callbacks(rc, pending)
    cb = cbs["intraday_loop"]
    assert cb is not None

    sched = AsyncIOScheduler()
    sched.add_job(
        cb,
        trigger=DateTrigger(run_date=datetime.now(UTC) + timedelta(milliseconds=50)),
        id="probe",
        misfire_grace_time=10,
    )
    sched.start()
    try:
        await asyncio.sleep(0.5)
        if pending:
            await asyncio.gather(*list(pending))
    finally:
        sched.shutdown(wait=False)

    rc.tick_safe.assert_awaited_once()


def test_build_scheduler_tolerates_short_event_loop_stalls() -> None:
    """APScheduler's default misfire_grace_time is 1s: a cron fire (eod_close,
    nightly_scan) coinciding with a >1s stall is silently dropped. Give every
    job a generous grace window and coalesce duplicate misfires."""
    sched = build_scheduler(callbacks={spec["id"]: (lambda: None) for spec in list_job_specs()})
    jobs = sched.get_jobs()
    assert jobs, "scheduler registered no jobs"
    for job in jobs:
        assert job.misfire_grace_time is not None, job.id
        assert job.misfire_grace_time >= 60, job.id
        assert job.coalesce is True, job.id
