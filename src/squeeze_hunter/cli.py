"""Squeeze-hunter CLI."""

from __future__ import annotations

import asyncio

import typer

from squeeze_hunter.broker.ibkr import IBKRBroker
from squeeze_hunter.logging_setup import configure_logging, get_logger

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


if __name__ == "__main__":
    app()
