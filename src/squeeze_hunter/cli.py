"""Squeeze-hunter CLI."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

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


if __name__ == "__main__":
    app()
