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


if __name__ == "__main__":
    app()
