# Architecture hardening plan (2026-09-06)

Companion to design spec §10. Same rules as the earlier plans: design → spec →
plan → TDD; every task starts with a failing test; nothing lands without
`ruff`, `ty` and the full suite green; no change to risk parameters without a
backtest re-run and a commit message stating why.

Status legend: `[x]` done, `[ ]` open.

## P5 — Infrastructure: trading calendar and clock  `[x]` calendar, `[ ]` clock

- [x] `squeeze_hunter/trading_calendar.py`: NYSE closures (pandas_market_calendars), `is_trading_day`, `next_session`, `trading_sessions(start, end)`, `is_regular_session`, `session_open_utc`. Every former copy (runtime session helpers, `signals/earnings_reaction._us_business_holidays`, runner day loop, metrics captured-events, FINRA lag) imports from it; old names stay as aliases.
- [ ] `Clock` protocol: `now()`, `today()`, `is_session_open()`. `BacktestClock` (advances per session) and `WallClock`. Replaces the runner's `day_label` / `cur` split and the `Clock` dataclass in `data/providers/backtest.py`.
  - Acceptance: no module other than `trading_calendar` imports `pandas_market_calendars` or defines session times.

## P9 — Remaining tunables to YAML  `[x]`

- [x] `risk.kelly_priors` (per setup win rate / payoff), `risk.killswitch.*` (three-day loss, gap-through-stop, broker outage, data stale, cooldown days), `risk.gates.*` (ADV20 multiple, max correlation). Code defaults equal the YAML values; runner and runtime read settings, never the function defaults.
  - Acceptance: `grep` for the old literals in `runner.py` / `runtime.py` finds none; a test overrides each via `Settings` and observes the behaviour change.

## P1 — Unified position core  `[x]`  (2026-09-20)

Goal: one implementation of "given the book, the quotes/bars and the clock, what do we sell, halve or buy", called by both the backtest and the live daemon.

- [x] `execution/book.py`: positions stay plain dicts (the daemon, its tests and the runtime address them by key) but `new_position_meta` is the only constructor and `realized_pnl` the only P&L rule, so the two paths cannot drift on shape.
- [x] `execution/decisions.py` (pure): `decide_exit(meta, MarkSnapshot, StopParams) -> ExitDecision` — `MarkSnapshot.from_quote` (live tick) or `.from_bar` (daily bar: evaluate at the low, peak-before = open, peak-after = close, price-stops fill at the low, other stops at the close). `propose_entries(ranked, state, ctx, settings, ...) -> list[EntryDecision]` — Kelly + gates with slot reservation; every candidate yields an accepted/rejected decision with the gate reason (the seed for the P8 decision log). `setup_stats_from_trades` gives the runner its observed win/payoff stats.
- [x] `execution/context.py`: `build_gate_context` assembles ADV20 / price floor / earnings proximity from the cache — identical inputs for backtest and premarket.
- [x] `risk/killswitch.py`: `advance_killswitch(state, verdict, now, cooldown_days)` is the sticky-cooldown step; runtime and runner both call it. `telemetry.PortfolioTelemetry` moved out of `runtime.py` and the backtest feeds the same class (its private `_build_killswitch_inputs` is gone).
- [x] `backtest/runner.py` and `execution/lifecycle.py` call the core; the runner's private stop / halve / peak / sizing code is deleted.
- [x] Live entry path behind `execution.auto_enter` (default false): `premarket_verify` sizes `last_candidates` with `propose_entries`; `_execute_planned_entries` buys them once after `execution.entry_after_minutes` past the open, as a marketable limit `entry_limit_bps` above the ask, and registers the position through `new_position_meta` + telemetry. TWAP slicing via the OMS is deferred to P3 (the OMS assumes synchronous fills, which IBKR never gives).
- [x] Acceptance met: `evaluate_stops`, `kelly_priors_for_setup` and `evaluate_gates` each have exactly one caller in `src/`; `tests/backtest/test_runner_parity.py` replays the runner's bars through `decide_exit` by hand and gets the same exit.
- Not yet: pending BUY orders are not tracked (a non-filled entry is logged and dropped, never re-sent) — P3.

## P2 — Persistent state and reconciliation  `[x]`  (2026-09-20)

- [x] `store/state.py`: `StateStore` protocol; `JsonStateStore` writes `data.state_path` (example YAML: `data/state/runtime.json`; empty = off) atomically via temp file + fsync + rename after every tick, nightly scan, premarket, EOD close and at shutdown. Snapshot = positions (with pending exit ids / action / qty), planned entries, killswitch state and active reasons, telemetry equity history and per-day peaks. A corrupt file is moved aside and the runtime starts clean.
- [x] `IBroker.get_positions() -> list[PositionSnapshot]` (ticker, qty, avg cost) on the simulator and IBKR (fresh `reqPositionsAsync`, account-filtered, rows aggregated).
- [x] `RuntimeContext.setup()` restores the snapshot, then `_reconcile_with_broker(full=True)`: phantoms (local, broker flat) are dropped, quantity mismatches adopt the broker's quantity, unknown broker holdings are adopted as `Mixed` with score 0 (signal-decay off; hard / trailing / time stops still apply) at the broker's average cost — every drift is alerted.
- [x] 60 s reconcile in the tick (adopt quantities, log only; positions with an exit in flight are left to the daemon's own reconcile) and a full EOD reconcile with an alert on any drift. A pure in-memory sim without a state store skips reconciliation (test harness); paper / live always reconcile.
- [x] IBKR client order refs: `sh-<side>-<ticker>-<qty>-<utc second>` in `orderRef`; a repeat of the same logical order returns the existing trade instead of placing a second one.
- [x] Acceptance met: `tests/runtime/test_persistence_and_reconcile.py` restarts a context over the same simulator and store and finds positions, pending exits, the killswitch lockout and equity history intact; adoption / phantom / quantity / EOD-drift cases each have a test.

## P3 — Order state machine and fake-IB contract tests  `[x]`  (2026-09-20)

- [x] `execution/orders.py`: `OrderState` (PENDING → ROUTED → PARTIAL → FILLED | REJECTED | CANCELLED | EXPIRED), `OrderRecord` (immutable, transition-validated `apply`, terminal states are final, a partial fill survives a cancel) and `OrderTracker` (open orders, snapshot round-trip).
- [x] `IBroker.get_order(id)` on the simulator (order history) and IBKR (`IB.trades()`, so fills that arrived after `placeOrder` are visible). IBKR status vocabulary: PendingSubmit/ApiPending → pending, PreSubmitted/Submitted/PendingCancel → routed (partial when anything filled), ValidationError/Inactive → rejected; `MarketOrder.lmtPrice == UNSET_DOUBLE` reads as no limit.
- [x] The OMS polls `get_order` until a slice is terminal or its time budget is spent, then cancels the remainder — it no longer reads fills off the submit response (only the simulator ever populated that).
- [x] Pending buys: an entry that does not fill on the submitting tick is tracked in `RuntimeContext.pending_buys`, persisted, settled on later ticks (or after a restart, before reconciliation could adopt the fill as an unknown lot), and cancelled once `execution.entry_window_minutes` have passed with the filled part kept as the position.
- [x] `tests/broker/fake_ib.py` reproduces the ib_async semantics the MagicMocks hid (blocking `reqAccountUpdates`, PendingSubmit on place, PendingCancel on cancel, one cached Ticker per contract, done trades in `trades()`); `tests/broker/test_ibkr_contract.py` runs the real `IBKRBroker` — and the real lifecycle daemon — against it, including the cancel-must-be-acknowledged-before-resubmit path.
- Not done: exit orders still use the daemon's `pending_exits` ids (tested, reconciled by position); migrating them onto `OrderTracker` and wiring TWAP slicing into the entry path are follow-ups (the OMS sleeps between slices and cannot run inside a 60 s tick — it needs its own task).

## P4 — Split RuntimeContext  `[ ]`

- `KillswitchController` (evaluate + sticky cooldown + reasons + alert + gauges), `TradingSession` (tick / nightly_scan / eod_close / premarket_verify), `RuntimeWiring` (broker, monitor server, alert sender, settings). `RuntimeContext` becomes a thin facade so the CLI and tests keep their entry points.

## P6 — Live data pipeline  `[x]`  (2026-09-20)

- [x] `ingest/eod.py` (`ingest_eod`, wired to the 17:00 ET `ingest_eod` job through `RuntimeContext.ingest_eod_safe`): bars pulled incrementally from the day after each ticker's last cached bar (one failing ticker never blocks the rest); FINRA re-pulled when the last ingest is older than `data.finra_refresh_days`; earnings when older than `data.earnings_refresh_days` (skipped and reported without `FINNHUB_KEY`). Problems go out as a LOW-severity alert.
- [x] `ingest/freshness.py`: per-dataset `as_of` / `recorded_at` stamps in `data/parquet/_freshness.json`. `premarket_verify` refuses to plan automatic entries (and alerts) when a dataset in `data.critical_datasets` (default `[bars]`) is older than its `*_max_age_days` or never ingested; `nightly_scan` warns about every stale dataset; `data.require_fresh_for_entries: false` disables the gate.
- [x] `data/providers/finra_api.py`: FINRA Query API client (client-credentials token, paged `consolidatedShortInterest` queries). `backfill_finra` falls back to it when the CDN download fails and `FINRA_API_CLIENT_ID` / `FINRA_API_CLIENT_SECRET` are set; without credentials the CDN failure stays loud. Field names follow the published dataset definition and have not yet been exercised against the live API.
- Not done: Reddit / options ingest (f4 / f5 stay dead by design until those sources are plugged in).

## P7 — Golden-number and invariant tests  `[x]`  (2026-09-20)

- [x] `tests/backtest/synthetic_universe.py`: 10 tickers, 2 years of NYSE sessions, seeded random walks with six scripted squeeze episodes (elevated SI, an earnings report the evening before, a +22% gap on 8x volume). `tests/backtest/test_golden.py` runs the full pipeline over it and compares Sharpe, Sortino, drawdown, hit rate, payoff, trade counts, exit reasons and realized P&L with `golden/expected.json` (rel 1e-6). Regenerate with `UPDATE_GOLDEN=1` only together with the change that explains the new numbers. Marked `slow` (≈50 s): pre-push skips it, CI runs it.
- [x] Invariants over the same run: cash never negative, never net short, every entry within the position cap and the daily new-position cap, no position outlives the time stop.
- [x] Made feasible by memoising prepared partitions in `BacktestProvider` (parsed, sorted, FINRA availability precomputed, Bar objects built once and sliced by bisect); the golden numbers were generated before the memoisation and matched after it.

## P8 — Decision log  `[x]`  (2026-09-20)

- [x] `execution/decision_log.py`: one row per candidate per day (date, ticker, score, setup, accepted, reason, size, source). The runner returns it as `BacktestResult.decisions`; walk-forward concatenates every window (`raw["decisions"]`, labelled) and the CLI writes `data/backtests/decisions.parquet`; the live premarket path appends to the parquet partition `decisions/live`. `squeeze-hunter explain --ticker --date [--source]` prints the matching rows.

## P10 — Deployment  `[ ]`

- `squeeze-hunter` service in `docker/compose.yml` (`restart: unless-stopped`, `.env` mounted, `/health` as the healthcheck) and a nightly backup job for `data/state` and `data/parquet`; the runbook is updated to describe what actually exists.
