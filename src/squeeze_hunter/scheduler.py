"""APScheduler job graph for the daily run loop (spec Section 7)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

# Times are ET. APScheduler stores its tz name on each trigger.
_TZ = "America/New_York"


def list_job_specs() -> list[dict[str, Any]]:
    return [
        {
            "id": "ingest_eod",
            "trigger": "cron",
            "hour": 17,
            "minute": 0,
            "tz": _TZ,
            "doc": "Backfill bars + sentiment for the day just closed",
        },
        {
            "id": "nightly_scan",
            "trigger": "cron",
            "hour": 22,
            "minute": 0,
            "tz": _TZ,
            "doc": "Full-universe scan, persist candidate ranking, push alert",
        },
        {
            "id": "premarket_data",
            "trigger": "cron",
            "hour": 4,
            "minute": 0,
            "tz": _TZ,
            "doc": "Overnight news + halt list ingest",
        },
        {
            "id": "premarket_verify",
            "trigger": "cron",
            "hour": 8,
            "minute": 0,
            "tz": _TZ,
            "doc": "Sanity check candidate list against overnight info",
        },
        {
            "id": "intraday_loop",
            "trigger": "interval",
            "seconds": 60,
            "doc": "Position lifecycle + risk gate (manage_positions, killswitch)",
        },
        {
            "id": "moc_decision",
            "trigger": "cron",
            "hour": 15,
            "minute": 55,
            "tz": _TZ,
            "doc": "MoC: which positions stay overnight",
        },
        {
            "id": "eod_close",
            "trigger": "cron",
            "hour": 16,
            "minute": 30,
            "tz": _TZ,
            "doc": "Close cleanup, daily metrics snapshot",
        },
    ]


def build_scheduler(
    callbacks: dict[str, Callable[[], Any]],
) -> AsyncIOScheduler:
    sched = AsyncIOScheduler()
    for spec in list_job_specs():
        cb = callbacks.get(spec["id"])
        if cb is None:
            continue
        if spec["trigger"] == "cron":
            trigger = CronTrigger(
                hour=spec.get("hour"),
                minute=spec.get("minute"),
                day_of_week="mon-fri",
                timezone=spec.get("tz"),
            )
        else:
            trigger = IntervalTrigger(seconds=spec["seconds"])
        sched.add_job(cb, trigger=trigger, id=spec["id"])
    return sched
