"""Broker ↔ book reconciliation (P2, extracted in P4).

Pure with respect to the runtime: takes the broker, the book and the
telemetry, returns the list of drifts it corrected. Logging, alerting and
the sim-mode skip rule stay with the caller.
"""

from __future__ import annotations

from collections.abc import Set as AbstractSet

from squeeze_hunter.broker.base import IBroker
from squeeze_hunter.execution.book import new_position_meta
from squeeze_hunter.execution.lifecycle import LifecycleState
from squeeze_hunter.telemetry import PortfolioTelemetry

_TRANSIENT_IO_ERRORS = (ConnectionError, TimeoutError, OSError)


async def reconcile_book(
    broker: IBroker,
    book: LifecycleState,
    telemetry: PortfolioTelemetry,
    *,
    full: bool,
    skip_tickers: AbstractSet[str] = frozenset(),
) -> list[str]:
    """Make the local book agree with the broker.

    - A local position the broker does not hold is a phantom: dropped.
    - A quantity mismatch adopts the broker's quantity.
    - A broker holding we do not know is adopted with conservative meta
      (entry = avg cost, setup Mixed, score 0 so signal-decay never fires,
      bars_held 0) so the hard / trailing / time stops manage it.
    Positions with an exit in flight are skipped unless `full`: the daemon's
    own pending-exit reconcile owns them. `skip_tickers` are holdings a
    pending BUY is about to explain (P3). Raises transient I/O errors from
    `get_positions` to the caller.
    """
    snapshots = await broker.get_positions()
    at_broker = {p.ticker: p for p in snapshots if p.qty > 0}
    drift: list[str] = []
    for ticker in list(book.positions):
        meta = book.positions[ticker]
        if not full and meta.get("pending_exits"):
            continue
        held = at_broker.get(ticker)
        if held is None:
            drift.append(f"{ticker}: local {meta['qty']} but broker flat -> dropped")
            book.positions.pop(ticker, None)
            telemetry.clear_position(ticker)
            continue
        if int(held.qty) != int(meta["qty"]):
            drift.append(f"{ticker}: local {meta['qty']} vs broker {held.qty} -> adopted")
            meta["qty"] = int(held.qty)
    for ticker, held in at_broker.items():
        if ticker in book.positions or ticker in skip_tickers:
            continue
        entry = float(held.avg_cost) if held.avg_cost > 0 else 0.0
        if entry <= 0:
            try:
                q = await broker.fetch_quote(ticker)
                entry = float(q.last or q.bid or q.ask or 0.0)
            except _TRANSIENT_IO_ERRORS:
                entry = 0.0
        if entry <= 0:
            drift.append(f"{ticker}: broker holds {held.qty} but no price to adopt it -> ignored")
            continue
        book.positions[ticker] = new_position_meta(
            ticker=ticker,
            qty=int(held.qty),
            entry_price=entry,
            score=0.0,
            setup_type="Mixed",
        )
        telemetry.record_position(ticker, entry, entry)
        drift.append(f"{ticker}: broker holds {held.qty} unknown locally -> adopted as Mixed")
    return drift
