from squeeze_hunter.scheduler import build_scheduler, list_job_specs


def test_scheduler_has_expected_jobs() -> None:
    specs = list_job_specs()
    job_ids = {s["id"] for s in specs}
    assert {
        "ingest_eod",
        "nightly_scan",
        "premarket_data",
        "premarket_verify",
        "intraday_loop",
        "moc_decision",
        "eod_close",
    }.issubset(job_ids)


def test_intraday_uses_60s_interval() -> None:
    specs = {s["id"]: s for s in list_job_specs()}
    assert specs["intraday_loop"]["trigger"] == "interval"
    assert specs["intraday_loop"]["seconds"] == 60


def test_build_scheduler_registers_all_jobs() -> None:
    sched = build_scheduler(
        callbacks={
            "ingest_eod": lambda: None,
            "nightly_scan": lambda: None,
            "premarket_data": lambda: None,
            "premarket_verify": lambda: None,
            "intraday_loop": lambda: None,
            "moc_decision": lambda: None,
            "eod_close": lambda: None,
        }
    )
    job_ids = {j.id for j in sched.get_jobs()}
    assert {"ingest_eod", "nightly_scan", "intraday_loop"}.issubset(job_ids)
