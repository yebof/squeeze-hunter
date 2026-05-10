"""Squeeze-hunter CLI."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from squeeze_hunter.broker.ibkr import IBKRBroker
from squeeze_hunter.config import load_settings
from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.data.providers.backtest import BacktestProvider, Clock
from squeeze_hunter.logging_setup import configure_logging, get_logger
from squeeze_hunter.scan import run_scan

app = typer.Typer(no_args_is_help=True)
log = get_logger("cli")


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
    provider = BacktestProvider(cache=cache, clock=clock)
    tickers = [line.strip() for line in tickers_file.read_text().splitlines() if line.strip()]

    ranked = asyncio.run(run_scan(tickers, provider, clock_dt, settings))
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / f"{date_str}.csv"
    ranked.to_csv(out_path, index=False)
    typer.echo(f"wrote {out_path} ({len(ranked)} tickers)")


ingest_app = typer.Typer(help="Historical backfill commands")
app.add_typer(ingest_app, name="ingest")


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
    asyncio.run(
        backfill_bars_for_universe(
            tickers,
            datetime.fromisoformat(start).replace(tzinfo=UTC),
            datetime.fromisoformat(end).replace(tzinfo=UTC),
            cache,
        )
    )


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
    asyncio.run(backfill_finra(tickers, cache))


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
    asyncio.run(backfill_earnings(tickers, cache))


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
    )
    report = asyncio.run(run_walk_forward(cfg, cache=cache, settings=settings))
    out.mkdir(parents=True, exist_ok=True)
    holdout_eq = report["raw"]["holdout_equity"]
    n_obs = max(20, len(holdout_eq.dropna()))
    verdict = evaluate_gate1(report["holdout"], n_trials=n_trials, n_obs=n_obs)
    holdout_eq.to_csv(out / "holdout_equity.csv")
    report["raw"]["trades"].to_csv(out / "holdout_trades.csv", index=False)
    summary_path = out / "gate1_report.txt"
    summary_path.write_text(_format_report(report, verdict))
    typer.echo(summary_path.read_text())


def _format_report(report: dict, verdict: Any) -> str:
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
    lines.append("")
    lines.append("=== Gate 1 verdict ===")
    lines.append(f"PASSED: {verdict.passed}")
    if verdict.failures:
        for f in verdict.failures:
            lines.append(f"  - {f}")
    lines.append(f"deflated_sharpe = {verdict.deflated_sharpe_value:.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    app()
