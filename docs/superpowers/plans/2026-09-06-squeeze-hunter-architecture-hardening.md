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

## P1 — Unified position core  `[ ]`  (largest; do first among the open items)

Goal: one implementation of "given the book, the quotes/bars and the clock, what do we sell, halve or buy", called by both the backtest and the live daemon.

1. `execution/book.py`: `Position` dataclass (ticker, qty, entry_price, peak, entry_score, current_score, bars_held, setup_type, halved, pending order ids, last_mark) and `PositionBook` (dict + invariants: qty > 0, one position per ticker).
2. `execution/decisions.py` (pure):
   - `decide_exits(book, marks: dict[ticker, MarkSnapshot], params) -> list[ExitDecision]` where `MarkSnapshot` carries `price`, `low`, `open` (backtest) or `last` (live). Trailing evaluation uses `max(peak, open)` when an open is supplied, else `price`.
   - `propose_entries(candidates, book, ctx, settings, stats) -> list[TradeProposal]` wrapping Kelly + gates (moved from the runner).
3. `execution/lifecycle.py` becomes: fetch quotes → build marks → `decide_exits` → submit via broker with the existing reconcile / cancel-confirm logic. `backtest/runner.py` becomes: bars → marks → `decide_exits` → simulator fills; entries via `propose_entries` → next-open fills.
4. Live entry path behind `execution.auto_enter: false` (YAML): when true, `premarket_verify` turns `last_candidates` into proposals and the OMS/TWAP path submits them at 09:35 ET.
5. Delete the runner's private stop/halve/peak code. Tests: the existing runner and lifecycle suites must pass unchanged where they assert behaviour; a new parity test drives the same synthetic bar series through both paths and asserts identical exits.
   - Acceptance: `evaluate_stops` has exactly one caller (`decide_exits`); `kelly_priors_for_setup` and `evaluate_gates` have exactly one caller (`propose_entries`).
   - Estimate: 1–2 sessions. Changes backtest numbers only where the two paths currently disagree.

## P2 — Persistent state and reconciliation  `[ ]`

1. `store/state.py`: `StateStore` protocol with `save(snapshot)` / `load()`; `JsonStateStore` writes `data/state/runtime.json` atomically (temp file + rename) after every tick and job; snapshot = book, pending orders, killswitch state, last scan date.
2. `RuntimeContext.setup()` loads the snapshot, then reconciles against `broker.positions()`: unknown broker positions are adopted with conservative meta (entry = avg cost, setup = Mixed, score 0 → decay disabled, bars_held 0) and an alert; local positions absent at the broker are dropped with an alert.
3. 60 s reconcile inside the tick (qty mismatch → adopt broker qty + warn) and EOD full reconcile with an alert on any drift.
4. Client order ids (`sh-<ticker>-<utc ts>`) passed to IBKR `orderRef` and deduped on submit.
   - Acceptance: kill -9 during a session, restart, positions and pending orders are intact and reconciled; a test simulates it with the simulator.

## P3 — Order state machine and fake-IB contract tests  `[ ]`

1. `execution/orders.py`: `OrderRecord` with the spec's states (PENDING → ROUTED → PARTIAL → FILLED | REJECTED | CANCELLED | EXPIRED), `is_terminal`, transition validation.
2. `IBKRBroker` maps ib_async statuses onto it (ValidationError → REJECTED, PendingCancel → still ROUTED).
3. `tests/broker/fake_ib.py`: a fake `IB` that reproduces the semantics the mocks hid: `placeOrder` returns PendingSubmit and later transitions, `cancelOrder` → PendingCancel then Cancelled after N loop iterations, `reqMktData` returns one cached `Ticker` per contract with a `time` stamp, `reqAccountUpdates` raises inside a running loop. Contract tests run the real `IBKRBroker` against it.

## P4 — Split RuntimeContext  `[ ]`

- `KillswitchController` (evaluate + sticky cooldown + reasons + alert + gauges), `TradingSession` (tick / nightly_scan / eod_close / premarket_verify), `RuntimeWiring` (broker, monitor server, alert sender, settings). `RuntimeContext` becomes a thin facade so the CLI and tests keep their entry points.

## P6 — Live data pipeline  `[ ]`

- `ingest_eod` job: bars (Yahoo) for the universe, FINRA when a new report is due, earnings weekly. Each dataset writes a `freshness` stamp; `premarket_verify` refuses to publish candidates when a critical dataset is older than its budget. FINRA API client as a fallback to the CDN files.

## P7 — Golden-number and invariant tests  `[ ]`

- A checked-in synthetic universe (10 tickers, 2 years, deterministic) with expected Gate 1 metrics; any change in `metrics.py`, the runner or the cost model must update the numbers explicitly. Property tests on the simulator: never net short, gross exposure ≤ cap, cash never negative.

## P8 — Decision log  `[ ]`

- Both paths append `(date, ticker, score, setup, gate_reason, size)` rows to `data/decisions/<run>.parquet`; `squeeze-hunter explain --date --ticker` prints them.

## P10 — Deployment  `[ ]`

- `squeeze-hunter` service in `docker/compose.yml` (`restart: unless-stopped`, `.env` mounted, `/health` as the healthcheck) and a nightly backup job for `data/state` and `data/parquet`; the runbook is updated to describe what actually exists.
