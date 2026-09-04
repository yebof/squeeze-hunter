"""Squeeze-hunter CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from squeeze_hunter.broker.ibkr import IBKRBroker
from squeeze_hunter.config import load_settings
from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.data.providers.backtest import BacktestProvider, Clock
from squeeze_hunter.logging_setup import configure_logging, get_logger
from squeeze_hunter.scan import run_scan

if TYPE_CHECKING:
    from squeeze_hunter.runtime import RuntimeContext

app = typer.Typer(no_args_is_help=True)
log = get_logger("cli")


def _load_dotenv(start: Path | None = None) -> Path | None:
    """Load the nearest `.env` at or above `start` (default: cwd), never
    overriding variables already present in the real environment.

    Round-12: README told operators to `cp .env.example .env`, but nothing
    ever read the file — Settings drops the dotenv source and IBKR_* /
    FINNHUB_KEY are plain os.environ lookups — so credentials silently
    resolved to the paper-account defaults.
    """
    from dotenv import load_dotenv

    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        env_file = candidate / ".env"
        if env_file.is_file():
            load_dotenv(env_file, override=False)
            return env_file
    return None


@app.callback()
def _root() -> None:
    """Squeeze-hunter CLI."""
    _load_dotenv()


def _build_runtime_callbacks(
    rc: RuntimeContext,
    pending_tasks: set[asyncio.Task[Any]] | None = None,
) -> dict[str, Callable[[], Any] | None]:
    """Map scheduler job IDs to callbacks for paper/live runtime modes.

    Phase 3 wires: nightly_scan, premarket_verify, intraday_loop, eod_close.
    The other three (ingest_eod, premarket_data, moc_decision) are explicit None
    with comments documenting their Phase 4 ownership so the scheduler silently
    skips them rather than crashing on a missing key.

    R6.C2: `pending_tasks` is an external set the caller maintains so it can
    await in-flight tasks at shutdown. Each scheduled task is added to the
    set on creation and removed via done-callback. APScheduler's
    `sched.shutdown(wait=True)` only waits for the synchronous lambda to
    return — NOT for the coroutine the lambda spawned via create_task —
    so without external tracking, in-flight ticks get cancelled mid-flight
    when the process tries to exit.
    """
    tracker = pending_tasks if pending_tasks is not None else set()

    def _spawn(coro: Any) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro)
        tracker.add(task)
        task.add_done_callback(tracker.discard)
        return task

    def _job(make_coro: Callable[[datetime], Any]) -> Callable[[], Any]:
        # Round-12: APScheduler's AsyncIOExecutor runs *sync* job functions in
        # a worker thread (loop.run_in_executor), where asyncio.create_task
        # raises "no running event loop" — so the previous sync lambdas never
        # ran a single tick in paper/live. A coroutine job is scheduled on the
        # loop itself; it spawns the tracked task and returns immediately, so
        # R6.C2's shutdown handling (await pending_tasks) still applies.
        async def _run() -> None:
            _spawn(make_coro(datetime.now(UTC)))

        return _run

    return {
        # Phase 4: integrate live EOD data ingest (yfinance → parquet backfill)
        "ingest_eod": None,
        "nightly_scan": _job(lambda now: rc.nightly_scan_safe(now=now)),
        # Phase 4: overnight news + halt-list ingest
        "premarket_data": None,
        "premarket_verify": _job(lambda now: rc.premarket_verify_safe(now=now)),
        "intraday_loop": _job(lambda now: rc.tick_safe(now=now)),
        # Phase 4: MoC / EOL flatten logic
        "moc_decision": None,
        "eod_close": _job(lambda now: rc.eod_close_safe(now=now)),
    }


@app.command()
def hello(ticker: str = "AAPL") -> None:
    """Connect to IBKR (paper) and print a quote."""
    configure_logging()

    async def run() -> None:
        broker = IBKRBroker()
        await broker.connect()
        try:
            q = await broker.fetch_quote(ticker)
            log.info("quote", ticker=q.ticker, bid=q.bid, ask=q.ask, last=q.last)
            typer.echo(f"{q.ticker}: bid={q.bid} ask={q.ask} last={q.last}")
        finally:
            await broker.disconnect()

    asyncio.run(run())


@app.command()
def scan(
    date_str: Annotated[str, typer.Option("--date", help="YYYY-MM-DD")],
    parquet_root: Annotated[Path, typer.Option("--data")] = Path("data/parquet"),
    config_path: Annotated[Path, typer.Option("--config")] = Path("config/settings.example.yml"),
    tickers_file: Annotated[Path, typer.Option("--tickers")] = Path("config/universe.txt"),
    out: Annotated[Path, typer.Option("--out")] = Path("data/scans"),
) -> None:
    """Run a single-day scan against the parquet cache."""
    configure_logging()
    settings = load_settings(config_path)
    cache = ParquetCache(root=parquet_root)
    clock_dt = datetime.fromisoformat(date_str).replace(tzinfo=UTC, hour=23, minute=59)
    clock = Clock(now=clock_dt)
    provider = BacktestProvider(
        cache=cache,
        clock=clock,
        finra_publication_lag_bdays=settings.data.finra_publication_lag_bdays,
    )
    tickers = [line.strip() for line in tickers_file.read_text().splitlines() if line.strip()]
    # R5.M5: surface an empty tickers file rather than silently doing nothing.
    if not tickers:
        typer.echo(f"WARNING: {tickers_file} is empty — no tickers to scan", err=True)

    ranked = asyncio.run(run_scan(tickers, provider, clock_dt, settings))
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / f"{date_str}.csv"
    ranked.to_csv(out_path, index=False)
    # R5.M3: warn loudly when the scan produced zero candidates so the user
    # doesn't mistake "wrote scan.csv with 0 rows" for a healthy run.
    if len(ranked) == 0:
        typer.echo(
            f"WARNING: scan produced 0 candidates for {date_str}. "
            "Likely causes: parquet missing for that date, empty universe, or "
            "empty score.weights in config.",
            err=True,
        )
    typer.echo(f"wrote {out_path} ({len(ranked)} tickers)")


ingest_app = typer.Typer(help="Historical backfill commands")
app.add_typer(ingest_app, name="ingest")


def _parse_iso_to_utc(s: str) -> datetime:
    """R6.M7: parse an ISO date string into a UTC datetime.

    If the input already carries a tz offset, convert (not replace) so a
    timestamp like '10:00+05:00' becomes '05:00 UTC' instead of being
    silently relabeled to '10:00 UTC' (off by 5 hours).
    """
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


@ingest_app.command("bars")
def ingest_bars(
    start: Annotated[str, typer.Option("--start", help="YYYY-MM-DD")],
    end: Annotated[str, typer.Option("--end", help="YYYY-MM-DD")],
    tickers_file: Annotated[Path, typer.Option("--tickers")] = Path("config/universe.txt"),
    parquet_root: Annotated[Path, typer.Option("--data")] = Path("data/parquet"),
) -> None:
    """Backfill EOD bars from yfinance into parquet cache."""
    from squeeze_hunter.ingest.backfill_bars import backfill_bars_for_universe

    configure_logging()
    cache = ParquetCache(root=parquet_root)
    tickers = [line.strip() for line in tickers_file.read_text().splitlines() if line.strip()]
    if not tickers:
        typer.echo(f"WARNING: {tickers_file} is empty — nothing to backfill", err=True)
        return

    start_dt = _parse_iso_to_utc(start)
    end_dt = _parse_iso_to_utc(end)
    # R6.I4: reject inverted ranges loudly. yfinance silently returns empty
    # data for `start > end`, leaving no on-disk signal of the misconfiguration.
    if start_dt >= end_dt:
        typer.echo(
            f"ERROR: --start ({start}) must be before --end ({end})",
            err=True,
        )
        raise typer.Exit(code=2)

    asyncio.run(backfill_bars_for_universe(tickers, start_dt, end_dt, cache))


@ingest_app.command("finra")
def ingest_finra(
    tickers_file: Annotated[Path, typer.Option("--tickers")] = Path("config/universe.txt"),
    parquet_root: Annotated[Path, typer.Option("--data")] = Path("data/parquet"),
) -> None:
    """Backfill FINRA short-interest into parquet cache."""
    from squeeze_hunter.ingest.backfill_finra import backfill_finra

    configure_logging()
    cache = ParquetCache(root=parquet_root)
    tickers = [line.strip() for line in tickers_file.read_text().splitlines() if line.strip()]
    try:
        asyncio.run(backfill_finra(tickers, cache))
    except RuntimeError as e:
        # Round-12: a fully failed FINRA download used to exit 0 silently.
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(code=2) from e


@ingest_app.command("earnings")
def ingest_earnings(
    tickers_file: Annotated[Path, typer.Option("--tickers")] = Path("config/universe.txt"),
    parquet_root: Annotated[Path, typer.Option("--data")] = Path("data/parquet"),
) -> None:
    """Backfill earnings calendar into parquet cache."""
    from squeeze_hunter.ingest.backfill_earnings import backfill_earnings

    configure_logging()
    cache = ParquetCache(root=parquet_root)
    tickers = [line.strip() for line in tickers_file.read_text().splitlines() if line.strip()]
    try:
        asyncio.run(backfill_earnings(tickers, cache))
    except ValueError as e:
        # Round-12: a missing FINNHUB_KEY used to exit 0 having written nothing.
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(code=2) from e


@app.command()
def backtest(
    train_start: Annotated[str, typer.Option("--train-start")],
    train_end: Annotated[str, typer.Option("--train-end")],
    test_windows: Annotated[
        list[str], typer.Option("--test-window", help="ISO range, e.g. 2022-01-01:2022-12-31")
    ],
    holdout_range: Annotated[str, typer.Option("--holdout")],
    tickers_file: Annotated[Path, typer.Option("--tickers")] = Path("config/universe.txt"),
    parquet_root: Annotated[Path, typer.Option("--data")] = Path("data/parquet"),
    config_path: Annotated[Path, typer.Option("--config")] = Path("config/settings.example.yml"),
    out: Annotated[Path, typer.Option("--out")] = Path("data/backtests"),
    n_trials: Annotated[int, typer.Option("--n-trials")] = 1,
) -> None:
    """Run walk-forward backtest and produce a Gate 1 verdict."""
    from squeeze_hunter.backtest.gate1 import evaluate_gate1
    from squeeze_hunter.backtest.walk_forward import WalkForwardConfig, run_walk_forward

    configure_logging()
    settings = load_settings(config_path)
    cache = ParquetCache(root=parquet_root)
    tickers = [line.strip() for line in tickers_file.read_text().splitlines() if line.strip()]

    def _parse_range(s: str) -> tuple[datetime, datetime]:
        a, b = s.split(":")
        return (
            datetime.fromisoformat(a).replace(tzinfo=UTC),
            datetime.fromisoformat(b).replace(tzinfo=UTC),
        )

    cfg = WalkForwardConfig(
        tickers=tickers,
        train_start=datetime.fromisoformat(train_start).replace(tzinfo=UTC),
        train_end=datetime.fromisoformat(train_end).replace(tzinfo=UTC),
        test_windows=[_parse_range(w) for w in test_windows],
        holdout=_parse_range(holdout_range),
        # R6.C3: wire score.threshold from YAML so the backtest uses the
        # same entry gate as production. Previously this defaulted to 8.0
        # in WalkForwardConfig regardless of YAML override → backtests
        # were inconsistent with live trading.
        score_threshold=settings.score.threshold,
        # Round-12: the spec's 8-event case set was never passed through, so
        # the ">= 5/8 captured" Gate 1 condition was silently skipped.
        validation_events=[
            (e.ticker, datetime(e.date.year, e.date.month, e.date.day, tzinfo=UTC))
            for e in settings.backtest.validation_events
        ],
    )
    try:
        report = asyncio.run(run_walk_forward(cfg, cache=cache, settings=settings))
    except ValueError as e:
        # R6.I1: combine() raises ValueError when score.weights is empty
        # (R5.C6 fail-loud behavior). Bubble a clean error to the user
        # rather than a raw Python traceback.
        typer.echo(f"ERROR: backtest configuration is invalid: {e}", err=True)
        raise typer.Exit(code=2) from e
    out.mkdir(parents=True, exist_ok=True)
    holdout_eq = report["raw"]["holdout_equity"]
    n_obs = max(20, len(holdout_eq.dropna()))
    verdict = evaluate_gate1(report["holdout"], n_trials=n_trials, n_obs=n_obs)
    holdout_eq.to_csv(out / "holdout_equity.csv")
    report["raw"]["trades"].to_csv(out / "holdout_trades.csv", index=False)
    # R9.2: surface setup-type coverage so a "PASSED" with only CAR trades is
    # explicitly flagged. f4/f5 are 0-by-design today → no GME/Mixed labels.
    from squeeze_hunter.backtest.gate1 import setup_type_coverage_warning

    coverage_warning = setup_type_coverage_warning(report["raw"]["trades"])
    summary_path = out / "gate1_report.txt"
    summary_path.write_text(_format_report(report, verdict, coverage_warning))
    typer.echo(summary_path.read_text())


def _format_report(report: dict, verdict: Any, coverage_warning: str | None = None) -> str:
    lines = ["=== Walk-forward report ==="]
    for label, m in [
        ("Train", report["train"]),
        *[(f"Test[{i}]", m) for i, m in enumerate(report["test_windows"])],
        ("Holdout", report["holdout"]),
    ]:
        lines.append(
            f"{label:8s}  Sharpe={m['sharpe']:.2f}  Sortino={m['sortino']:.2f}  "
            f"MaxDD={m['max_drawdown']:.2%}  Hit={m['hit_rate']:.2f}  "
            f"Payoff={m['avg_payoff']:.2f}  Captured={m['captured_events']}  "
            f"ShuffleP={m['shuffle_pvalue']:.3f}  Trades={m['n_trades']}"
        )
    if report.get("n_validation_events"):
        lines.append(
            f"Captured events (all out-of-sample windows): "
            f"{report['captured_events_total']} / {report['n_validation_events']}"
        )
    lines.append("")
    lines.append("=== Gate 1 verdict ===")
    lines.append(f"PASSED: {verdict.passed}")
    if verdict.failures:
        for f in verdict.failures:
            lines.append(f"  - {f}")
    lines.append(f"deflated_sharpe = {verdict.deflated_sharpe_value:.3f}")
    # R9.2: setup-type coverage banner — warns if Gate 1 passed on a subset
    # of setups (typically the CAR-only case while f4/f5 are still zero).
    if coverage_warning:
        lines.append("")
        lines.append("=== Coverage warning ===")
        lines.append(coverage_warning)
    return "\n".join(lines)


@app.command()
def paper(
    config_path: Annotated[Path, typer.Option("--config")] = Path("config/settings.example.yml"),
    parquet_root: Annotated[Path, typer.Option("--data")] = Path("data/parquet"),
    tickers_file: Annotated[Path, typer.Option("--tickers")] = Path("config/universe.txt"),
    connect_timeout_s: Annotated[
        float, typer.Option("--connect-timeout", help="seconds to wait for TWS connect")
    ] = 30.0,
) -> None:
    """Run the paper-trading loop indefinitely."""
    from squeeze_hunter.runtime import RuntimeContext
    from squeeze_hunter.scheduler import build_scheduler

    configure_logging()
    settings = load_settings(config_path)
    cache = ParquetCache(root=parquet_root)
    tickers = [line.strip() for line in tickers_file.read_text().splitlines() if line.strip()]

    rc = RuntimeContext(cache=cache, settings=settings, tickers=tickers, mode="paper")
    pending_tasks: set[asyncio.Task[Any]] = set()

    async def main_loop() -> None:
        await rc.setup(connect_timeout_s=connect_timeout_s)
        sched = build_scheduler(callbacks=_build_runtime_callbacks(rc, pending_tasks))
        sched.start()
        try:
            await asyncio.Event().wait()  # block forever
        finally:
            # R6.C2: APScheduler's shutdown(wait=True) only awaits the
            # synchronous lambda, not the create_task it spawned. We must
            # explicitly await pending tasks so an in-flight tick can finish
            # cleanly before broker.disconnect tears down the socket.
            sched.shutdown()
            if pending_tasks:
                log.info("awaiting_in_flight_tasks", count=len(pending_tasks))
                # Snapshot the set — done callbacks will mutate it during await
                await asyncio.gather(*list(pending_tasks), return_exceptions=True)
            await rc.shutdown()

    asyncio.run(main_loop())


@app.command()
def live(
    config_path: Annotated[Path, typer.Option("--config")] = Path("config/settings.example.yml"),
    parquet_root: Annotated[Path, typer.Option("--data")] = Path("data/parquet"),
    tickers_file: Annotated[Path, typer.Option("--tickers")] = Path("config/universe.txt"),
    confirm: Annotated[bool, typer.Option("--confirm-real-money")] = False,
    connect_timeout_s: Annotated[
        float, typer.Option("--connect-timeout", help="seconds to wait for TWS connect")
    ] = 30.0,
) -> None:
    """Run the live-trading loop. Requires --confirm-real-money."""
    if not confirm:
        typer.echo("Refusing to start live without --confirm-real-money", err=True)
        raise typer.Exit(code=2)

    from squeeze_hunter.runtime import RuntimeContext
    from squeeze_hunter.scheduler import build_scheduler

    configure_logging()
    settings = load_settings(config_path)
    cache = ParquetCache(root=parquet_root)
    tickers = [line.strip() for line in tickers_file.read_text().splitlines() if line.strip()]

    rc = RuntimeContext(cache=cache, settings=settings, tickers=tickers, mode="live")
    pending_tasks: set[asyncio.Task[Any]] = set()

    async def main_loop() -> None:
        await rc.setup(connect_timeout_s=connect_timeout_s)
        sched = build_scheduler(callbacks=_build_runtime_callbacks(rc, pending_tasks))
        sched.start()
        try:
            await asyncio.Event().wait()
        finally:
            # R6.C2: see paper() above for rationale.
            sched.shutdown()
            if pending_tasks:
                log.info("awaiting_in_flight_tasks", count=len(pending_tasks))
                await asyncio.gather(*list(pending_tasks), return_exceptions=True)
            await rc.shutdown()

    asyncio.run(main_loop())


@app.command("emergency-flatten")
def emergency_flatten(
    confirm: Annotated[bool, typer.Option("--confirm")] = False,
    mode: Annotated[str, typer.Option("--mode", help="paper|live")] = "paper",
    client_id: Annotated[
        int,
        typer.Option(
            "--client-id",
            help="IBKR client ID for the flatten connection. Pick something that doesn't collide with the main loop's IBKR_CLIENT_ID (default 42).",
        ),
    ] = 199,
) -> None:
    """Market-flatten every open position. Requires --confirm."""
    # R7.C8: validate mode against the allowed set up front. Previously any
    # value other than "paper" silently fell into the IBKRBroker (live)
    # branch — including "sim" or typos like "papr". On a live account
    # this is dangerous.
    if mode not in {"paper", "live"}:
        typer.echo(f"emergency-flatten: --mode must be 'paper' or 'live', got {mode!r}", err=True)
        raise typer.Exit(code=2)
    if not confirm:
        typer.echo("Refusing without --confirm", err=True)
        raise typer.Exit(code=2)
    from squeeze_hunter.broker.paper import PaperBroker

    configure_logging()
    # R5.C5: client_id is configurable so an operator using IBKR_CLIENT_ID=199
    # (or whatever) for the main loop can pick a different one for the flatten
    # connection. IBKR rejects duplicate client IDs.
    broker: IBKRBroker | PaperBroker = (
        PaperBroker(client_id=client_id) if mode == "paper" else IBKRBroker(client_id=client_id)
    )

    async def go() -> None:
        await broker.connect()
        try:
            # R5: ib-async's IB.positions() is populated from TWS push events;
            # right after connectAsync the snapshot may be empty. Force a sync
            # before reading. reqPositionsAsync blocks until TWS responds.
            try:
                await broker._ib.reqPositionsAsync()
            except (AttributeError, NotImplementedError):
                # Older ib-async versions or simulator brokers: best-effort fallback
                await asyncio.sleep(2)
            opens = await broker.get_open_orders()
            for o in opens:
                await broker.cancel_order(o.broker_order_id)
            # Enumerate positions via ib-async — now safely populated.
            positions = broker._ib.positions()
            if not positions:
                log.info("emergency_flatten_no_positions")
                return
            for pos in positions:
                sym = pos.contract.symbol
                qty = abs(int(pos.position))
                if qty <= 0:
                    continue
                side = "sell" if pos.position > 0 else "buy"
                log.info("emergency_flatten", ticker=sym, qty=qty, side=side)
                if side == "sell":
                    await broker.submit_sell(
                        ticker=sym, qty=qty, limit_price=None, ts=datetime.now(UTC)
                    )
                else:
                    await broker.submit_buy(
                        ticker=sym, qty=qty, limit_price=None, ts=datetime.now(UTC)
                    )
        finally:
            # R5.M7: always disconnect, even if a submit raises mid-loop, so
            # we don't leak the IBKR client ID slot.
            try:
                await broker.disconnect()
            except (ConnectionError, TimeoutError, OSError, RuntimeError) as e:
                # R8.Q-I4: narrow per CLAUDE.md. Programming errors must
                # propagate even in shutdown-cleanup paths.
                log.warning(
                    "emergency_flatten_disconnect_failed",
                    err=str(e),
                    err_type=type(e).__name__,
                )

    asyncio.run(go())


if __name__ == "__main__":
    app()
