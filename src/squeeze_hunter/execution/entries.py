"""The live entry path (P1 + P3, extracted in P4): plan premarket, buy after
the opening window, settle pending buys, all against one book.

Everything the backtest does with `propose_entries` the runtime does here
with the same function; this module only supplies the live inputs (gate
context from the cache, portfolio state from the broker) and the live
outputs (orders, positions, telemetry, the decision log).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pandas as pd

from squeeze_hunter.broker.base import IBroker
from squeeze_hunter.broker.simulator import SimulatorBroker
from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.data.providers.backtest import BacktestProvider, Clock
from squeeze_hunter.execution.book import new_position_meta
from squeeze_hunter.execution.context import build_gate_context
from squeeze_hunter.execution.decision_log import DecisionLog
from squeeze_hunter.execution.decisions import EntryDecision, SetupStats, propose_entries
from squeeze_hunter.execution.lifecycle import LifecycleState
from squeeze_hunter.execution.orders import OrderRecord, OrderState, OrderTracker
from squeeze_hunter.execution.pricing import round_to_tick
from squeeze_hunter.logging_setup import get_logger
from squeeze_hunter.risk.gates import PortfolioState
from squeeze_hunter.telemetry import PortfolioTelemetry
from squeeze_hunter.trading_calendar import NY, SESSION_OPEN

log = get_logger("execution.entries")

_TRANSIENT_IO_ERRORS = (ConnectionError, TimeoutError, OSError)


@dataclass
class EntryPath:
    cache: ParquetCache
    settings: Settings
    tickers: list[str]
    book: LifecycleState
    telemetry: PortfolioTelemetry
    mode: str = "paper"
    broker: IBroker | None = None
    # Sized proposals produced by plan(); consumed once by execute_planned().
    planned: list[EntryDecision] = field(default_factory=list)
    # Entry orders that did not fill on the submitting tick (P3).
    pending: OrderTracker = field(default_factory=OrderTracker)

    # ------------------------------------------------------------------ plan
    async def plan(
        self,
        now: datetime,
        candidates: pd.DataFrame | None,
        *,
        kill_active: bool,
        kill_reason: str | None,
    ) -> list[EntryDecision]:
        """Premarket: size last night's candidates with the shared core.
        Returns every decision (accepted or not); `planned` keeps the
        accepted ones. The caller has already applied the freshness gate."""
        self.planned = []
        if kill_active:
            log.warning("premarket_entries_suppressed_killswitch", reason=kill_reason)
            return []
        if self.broker is None or candidates is None or candidates.empty:
            return []
        provider = BacktestProvider(
            cache=self.cache,
            clock=Clock(now=now),
            finra_publication_lag_bdays=self.settings.data.finra_publication_lag_bdays,
        )
        # The gate context is built for the LAST session (the bars the scan
        # saw), which is what "today's" ADV20 / price floor mean premarket.
        last_session = now.astimezone(NY).date() - timedelta(days=1)
        as_of = datetime.combine(last_session, datetime.max.time(), tzinfo=UTC)
        ctx = await build_gate_context(
            provider, self.cache, self.tickers, as_of, self.settings, kill_switch_active=False
        )
        try:
            equity = await self.broker.get_equity_usd()
        except _TRANSIENT_IO_ERRORS as e:
            log.warning("premarket_equity_unavailable", err=str(e))
            return []
        if equity is None or equity <= 0:
            log.warning("premarket_entries_skipped_no_equity")
            return []
        positions = {t: int(m["qty"]) for t, m in self.book.positions.items()}
        gross = 0.0
        for t, m in self.book.positions.items():
            mark = self.telemetry.position_marks.get(t, (m["entry_price"], m["entry_price"]))[1]
            gross += mark * m["qty"]
        state = PortfolioState(
            equity_usd=equity,
            cash_usd=equity - gross,
            gross_exposure_pct=gross / equity,
            positions=positions,
            opened_today=0,
        )
        decisions = propose_entries(
            candidates,
            state,
            ctx,
            self.settings,
            score_threshold=self.settings.score.threshold,
            # No realized-trade history is kept live yet: prior-only Kelly.
            stats_for_setup=lambda s: SetupStats(wins=0, trades=0, avg_payoff=None),
        )
        for d in decisions:
            log.info(
                "premarket_entry_decision",
                ticker=d.ticker,
                accepted=d.accepted,
                reason=d.reason,
                size_usd=round(d.size_usd, 2),
            )
        # P8: append to the live decision log (parquet partition decisions/live).
        dlog = DecisionLog()
        dlog.record(now, decisions, source=f"premarket:{self.mode}")
        if dlog.rows:
            self.cache.append_partition(
                "decisions", "live", dlog.to_frame(), dedup_keys=["date", "ticker", "source"]
            )
        self.planned = [d for d in decisions if d.accepted]
        return decisions

    # --------------------------------------------------------------- execute
    def _window_open(self, now: datetime) -> datetime:
        et = now.astimezone(NY)
        return datetime.combine(et.date(), SESSION_OPEN, tzinfo=NY) + timedelta(
            minutes=self.settings.execution.entry_after_minutes
        )

    async def execute_planned(self, now: datetime, *, kill_active: bool) -> None:
        """Buy the premarket proposals once, after the opening window: a
        marketable limit `entry_limit_bps` above the ask, snapped to the tick.
        A fill becomes a position through `new_position_meta`; anything else
        is tracked in `pending` and settled by `settle_pending`."""
        if not self.planned or self.broker is None:
            return
        if now.astimezone(NY) < self._window_open(now):
            return
        planned, self.planned = self.planned, []
        if kill_active:
            log.warning("planned_entries_dropped_killswitch", n=len(planned))
            return
        for d in planned:
            if d.ticker in self.book.positions:
                continue
            try:
                q = await self.broker.fetch_quote(d.ticker)
            except _TRANSIENT_IO_ERRORS as e:
                log.warning("entry_quote_failed", ticker=d.ticker, err=str(e))
                continue
            ref = q.ask or q.last or q.bid
            if (not ref or ref <= 0) and isinstance(self.broker, SimulatorBroker):
                # sim mode: the simulator only knows prices it has been marked
                # with; take the latest cached close, as the backtest would.
                ref = await self._latest_cached_close(d.ticker, now)
            if not ref or ref <= 0:
                log.warning("entry_skipped_no_quote", ticker=d.ticker)
                continue
            qty = int(d.size_usd // ref)
            if qty <= 0:
                log.info("entry_skipped_size_below_one_share", ticker=d.ticker, ref=ref)
                continue
            limit = round_to_tick(
                ref * (1 + self.settings.execution.entry_limit_bps / 10_000), "buy"
            )
            order = await self.broker.submit_buy(
                ticker=d.ticker, qty=qty, limit_price=limit, ts=now
            )
            log.info(
                "entry_submitted",
                ticker=d.ticker,
                qty=qty,
                limit=limit,
                status=order.status,
                broker_order_id=order.broker_order_id,
            )
            rec = OrderRecord.from_broker_order(
                order,
                "entry",
                now,
                meta={"score": d.score, "setup_type": d.setup_type, "size_usd": d.size_usd},
            )
            if rec.is_terminal:
                self._finish_entry(rec)
            else:
                self.pending.add(rec)
                log.info("entry_pending", ticker=d.ticker, broker_order_id=rec.order_id)

    async def _latest_cached_close(self, ticker: str, now: datetime) -> float:
        provider = BacktestProvider(cache=self.cache, clock=Clock(now=now))
        try:
            bars = await provider.fetch_bars(ticker, now - timedelta(days=7), now)
        except LookupError:
            return 0.0
        return float(bars[-1].close) if bars else 0.0

    # ---------------------------------------------------------------- settle
    def _finish_entry(self, rec: OrderRecord) -> None:
        """Turn a terminal entry order into a position (or drop it)."""
        self.pending.forget(rec.order_id)
        filled = int(rec.filled_qty or (rec.qty if rec.state is OrderState.FILLED else 0))
        if filled <= 0:
            log.warning(
                "entry_not_filled",
                ticker=rec.ticker,
                status=rec.state.value,
                broker_order_id=rec.order_id,
            )
            return
        entry_px = float(rec.avg_fill_price or rec.limit_price or 0.0)
        if entry_px <= 0:
            log.error("entry_filled_without_price", ticker=rec.ticker, broker_order_id=rec.order_id)
            return
        existing = self.book.positions.get(rec.ticker)
        if existing is not None:
            # Two fills for one ticker (propose_entries rejects held names, so
            # this is defensive) — merge rather than lose either.
            total = int(existing["qty"]) + filled
            existing["entry_price"] = (
                float(existing["entry_price"]) * int(existing["qty"]) + entry_px * filled
            ) / total
            existing["qty"] = total
            existing["peak_price"] = max(float(existing["peak_price"]), entry_px)
            log.warning("entry_merged_into_existing_position", ticker=rec.ticker, qty=total)
        else:
            self.book.positions[rec.ticker] = new_position_meta(
                ticker=rec.ticker,
                qty=filled,
                entry_price=entry_px,
                score=float(rec.meta.get("score", 0.0)),
                setup_type=str(rec.meta.get("setup_type", "Mixed")),
                entry_commission_per_share=(rec.commission_usd / filled) if filled else 0.0,
            )
        self.telemetry.record_position(rec.ticker, entry_px, entry_px)
        log.info(
            "entry_filled",
            ticker=rec.ticker,
            qty=filled,
            price=entry_px,
            status=rec.state.value,
            broker_order_id=rec.order_id,
        )

    async def settle_pending(self, now: datetime) -> None:
        """P3: advance every open entry order through broker.get_order; fills
        become positions, anything still working past the entry window is
        cancelled and its filled part kept."""
        if self.broker is None:
            return
        open_records = self.pending.open()
        if not open_records:
            return
        et = now.astimezone(NY)
        window_end = self._window_open(now) + timedelta(
            minutes=self.settings.execution.entry_window_minutes
        )
        for rec in open_records:
            try:
                bo = await self.broker.get_order(rec.order_id)
            except _TRANSIENT_IO_ERRORS as e:
                log.warning("pending_buy_poll_failed", broker_order_id=rec.order_id, err=str(e))
                continue
            if bo is None:
                if now - rec.submitted_at > timedelta(days=1):
                    log.warning("pending_buy_unknown_to_broker", broker_order_id=rec.order_id)
                    self.pending.forget(rec.order_id)
                continue
            updated = self.pending.update(bo, now=now) or rec
            if updated.is_terminal:
                self._finish_entry(updated)
                continue
            past_window = et >= window_end or et.date() != rec.submitted_at.astimezone(NY).date()
            if not past_window:
                continue
            try:
                await self.broker.cancel_order(rec.order_id)
                bo = await self.broker.get_order(rec.order_id)
            except _TRANSIENT_IO_ERRORS as e:
                log.warning("pending_buy_cancel_failed", broker_order_id=rec.order_id, err=str(e))
                continue
            if bo is not None:
                updated = self.pending.update(bo, now=now) or updated
            if updated.is_terminal:
                self._finish_entry(updated)
            else:
                log.warning(
                    "pending_buy_cancel_not_acknowledged",
                    broker_order_id=rec.order_id,
                    status=updated.state.value,
                )
