# Squeeze Hunter — Design Document

| Field | Value |
| --- | --- |
| Status | Draft (post-brainstorming, pre-implementation-plan) |
| Author | yebof |
| Date | 2026-05-10 |
| Target | Fully-automated quantitative trading system for short-squeeze events on US main-board mid/small caps |
| Related | (no prior docs — first design pass for this repo) |

## 1. Overview & Scope

### Goal

Build a fully-automated quantitative trading system that detects and profits from short-squeeze events on US main-board mid/small-cap equities. The system captures both:

- **GME-type** squeezes — gamma + retail-driven mania
- **CAR-type** squeezes — mechanical short cover triggered by fundamental catalysts (earnings beat + guidance raise on already-shorted names)

via a single unified scoring model. Trades are taken in long stock and long options (calls/puts) only — **no shorting of stock**.

### Universe

- Exchanges: NYSE + NASDAQ
- Market cap: **$200M – $10B** (~1500–2000 names)
- Price floor: **≥ $5**
- Excluded: OTC, Pink Sheets, ADR-only, recent IPOs (< 30 days listed)

### Trading Tools

- Long stock — used by both setup families
- Long calls — used by GME-type only (gamma plays)
- Long puts — broker capability is preserved but **no v1 setup opens put positions**; revisit when an overshoot mean-reversion or hedge variant is designed (Section 8)
- **No** short stock (avoids locate / hard-to-borrow complexity)

### Holding Period

Adaptive **1–30 trading days**, dispatched per setup classification (Section 3). Time stop hard-capped at 21 trading days.

### Risk Posture

Conservative bias throughout:

- **Fractional Kelly = 0.20×** (with Bayesian shrinkage when sample is thin)
- Single position cap **8%** of equity
- Max simultaneous positions **6**
- Max gross exposure **90%** (10% cash buffer; no leverage)
- Daily new-position cap **3**
- Drawdown kill-switch (Section 6)

### Non-Goals (out of scope for v1)

- Shorting stock (no margin/locate complexity)
- Intraday HFT / scalping
- Crypto, FX, futures
- Multi-strategy framework — this repo runs only the squeeze-hunter strategy
- Microcap (< $200M) and OTC / penny stocks (out of universe)
- Real-time streaming compute (60-second polling is the MVP cadence)
- Multi-account / multi-broker (IBKR only for v1)

### External Dependencies

- **IBKR account** with US options approval Level 2+ (long calls/puts permitted, no naked sells)
- **IBKR Gateway** running locally (Docker) for v1
- **Postgres 14+** for state and reference data
- Free data sources: FINRA, Yahoo (yfinance), IBKR L1, Reddit (PRAW), Finnhub free tier

### MVP Definition (Phase 3 done = success)

Backtest validation passes on the historical squeeze case set (Section 5) and 30 calendar days of paper trading run end-to-end without manual intervention, with logging, monitoring and the kill-switch all functional.

## 2. System Architecture

### Style

**Approach A — Modular monolith.** Single Python process organized into well-bounded packages communicating through narrow interfaces. Designed so that any package can later be lifted into a separate service (Approach B) with minimal refactor, and the score combiner can be replaced with an ML model (Approach C) without touching the rest.

### Daily Runtime Loop

```
17:00 ET   收盘后 EOD 数据落库（含盘后财报反应）
22:00 ET   ✨ 夜间全量扫描 + 候选名单生成 + Telegram/Slack 推送
04:00 ET   盘前数据补充（隔夜新闻/halts）+ 候选名单 sanity check
08:00 ET   最终验证：候选 + 隔夜信息 → 当日交易计划
09:30-09:35 不下单（避开开盘乱）
09:35-09:55 TWAP 切片入场
10:00-15:55 60s 循环：止损 / 信号衰减 / 风控
15:55 ET   MoC 决策（全平 / 留过夜）
16:30 ET   EOD 落库 + 当日复盘
```

The **scan + ranking job runs at night**; mornings are purely verification + new-info integration. Heavy computation never blocks the entry window.

### Module Layout

```
squeeze_hunter/
├── pyproject.toml            # uv-managed
├── src/squeeze_hunter/
│   ├── config.py             # Pydantic Settings (12-factor envs)
│   ├── data/                 # provider abstraction + concrete providers
│   │   ├── providers/        # ibkr / yahoo / finra / reddit / finnhub
│   │   ├── schema.py         # Bar, Quote, OptionChain, ShortInterest, …
│   │   └── cache.py          # in-memory LRU + parquet on disk
│   ├── universe.py           # filter & rebuild universe
│   ├── signals/              # one factor per file, pure functions
│   ├── score/
│   │   ├── combiner.py       # weighted z-score → squeeze_score
│   │   └── classifier.py     # GME / CAR / Mixed / Weak
│   ├── risk/
│   │   ├── kelly.py          # fractional Kelly + Bayesian shrink
│   │   ├── gates.py          # 14 pre-trade gates
│   │   ├── stops.py          # initial / trailing / time / signal-decay
│   │   └── killswitch.py
│   ├── execution/            # ('exec' is a Python builtin — renamed)
│   │   ├── oms.py            # order state machine
│   │   ├── slicing.py        # TWAP / VWAP slicers
│   │   └── lifecycle.py      # entry / manage / exit
│   ├── broker/
│   │   ├── base.py           # IBroker Protocol
│   │   ├── ibkr.py           # ib-async implementation
│   │   ├── paper.py          # paper trading
│   │   └── simulator.py      # backtest "broker"
│   ├── backtest/             # bar-based loop, replay, metrics
│   ├── monitor/              # prometheus exporter, alerts
│   ├── store/                # postgres (state) + parquet (history)
│   ├── scheduler.py          # APScheduler
│   └── cli.py                # typer: scan / backtest / paper / live / emergency-flatten
├── tests/
├── docker/
│   ├── compose.yml           # postgres + ib-gateway + prometheus + grafana + app
│   └── ib-gateway.dockerfile
└── docs/superpowers/specs/   # this file lives here
```

### Core Contracts (do not violate)

- **`IBroker` Protocol** — live / paper / simulator all share one interface
- **`DataProvider` Protocol** — free MVP and future paid providers are hot-swappable via config
- **Signals are pure functions** of the form `(history, clock) → (factor_value, evidence)`
- **Score weights and thresholds live in YAML** — tuning never changes code
- **All timestamps are UTC internally**; ET only on display

### Deliberate MVP Simplifications

- 60-second polling instead of true streaming
- Same code path serves backtest, paper, and live (only `IBroker` and `DataProvider` differ)
- No concurrency tuning — vectorized pandas is sufficient for ~1500-name daily scan
- No multi-account / multi-strategy
- No GUI dashboard — Grafana reads Prometheus exporter directly

## 3. Signal & Scoring Model

### Factor Set (7 factors, free sources only — no placeholders)

| # | Factor | Lean | Source | Weight |
| --- | --- | --- | --- | --- |
| f1 | SI % of Float | CAR | FINRA biweekly | 2.0 |
| f2 | Days-to-Cover (= SI / ADV20) | CAR | FINRA + IBKR | 1.0 |
| f3 | Earnings reaction (gap × volume z-score on report day) | CAR | Finnhub + IBKR | 2.0 |
| f4 | r/wallstreetbets mention count z-score (24h) | GME | Reddit PRAW | 1.5 |
| f5 | ATM call OI 7-day velocity | GME | Yahoo options | 1.5 |
| f6 | Bollinger / Keltner squeeze breakout | both | IBKR price | 1.0 |
| f7 | Volume spike vs ADV20 | both | IBKR price | 1.0 |

Total weight = 10.0.

### Scoring Pipeline

```
For each factor f_i, on each day, across the universe:
  raw_i        = factor's raw value per stock
  z_i          = (raw_i - μ_universe) / σ_universe        # cross-sectional z
  z_i_clipped  = clip(z_i, -3, +3)                        # tail clipping

squeeze_score = Σ_i  w_i × z_i_clipped                    # range ≈ [-30, +30]
```

Linear weighted z-score is intentional: transparent, debuggable, and the `combiner.score(factors) → score` interface stays stable when we later swap in an ML model.

### Setup Classifier (rule-based)

```
A = z[f1] + z[f3]    # CAR strength (SI + earnings reaction)
B = z[f4] + z[f5]    # GME strength (sentiment + call OI)

if   A ≥ 4 and B  < 2  → "CAR-type"
elif B ≥ 4 and A  < 2  → "GME-type"
elif A ≥ 3 and B ≥ 3  → "Mixed"
else                  → "Weak"  (rejected)
```

### Entry Threshold

```
squeeze_score ≥ 8.0   AND   setup_class ≠ "Weak"   AND   all 14 risk gates pass
```

### Setup-Driven Differentiation

| | CAR-type | GME-type |
| --- | --- | --- |
| Holding | 3–7 trading days | 5–21 trading days |
| Instrument mix | 100% stock | 70% stock + 30% OTM call (1.1× spot, 30–60d) |
| Hard stop | -12% | -12% |
| Trailing stop | -20% from peak | -25% from peak |
| Kelly assumption (initial prior) | win_rate 25% / payoff 3:1 | win_rate 15% / payoff 8:1 (fat tail) |

### Explicitly Rejected Factors and Reasons

| Rejected | Reason |
| --- | --- |
| CTB rate / Locate availability | No paid source available; refusing to ship a placeholder |
| Institutional / insider concentration | Cross-sectional signal-to-noise too low |
| FTD trend | Data lag, weak alpha |
| LULD halt pattern | Squeeze-rare, lagging trigger |
| Cross-subreddit resonance | Complexity not worth marginal alpha |
| Higher-highs / distance-from-52w-high | Redundant with Bollinger breakout |
| NLP guidance change | Engineering cost too high for v1 |
| Cashtag / Twitter / Stocktwits | Free-tier data quality poor |
| IV percentile (standalone factor) | Redundant with call OI |
| GEX estimate | Without paid OI data, estimation error dominates signal |

## 4. Data Layer

### Layered Structure

```
Domain Schemas (pydantic)
   Bar · Quote · OptionChain · ShortInterest · EarningsEvent · RedditMention
        ↑
DataProvider Protocol
   get_bars · get_quote · get_option_chain · get_short_interest
   get_earnings · get_sentiment
        ↑                ↑                ↑
Live providers     Historical providers     BacktestProvider
   IBKRProvider          Yahoo (yfinance)      replay parquet
   RedditProvider        FINRA bulk (FTP)      + clock-bound queries
   FinnhubProvider       Finnhub               (prevents lookahead)
        ↓
Cache (in-memory LRU + disk parquet, partitioned by day & ticker)
        ↓
Postgres (state + slow-changing reference)
```

### MVP Providers and Their Limits

| Provider | What it provides | Limits / Quality |
| --- | --- | --- |
| `IBKRProvider` | Real-time bars + quotes (L1), option chain + greeks, recent historical bars | ~50 ticker concurrent streams; Gateway must be local or Docker; uses `ib-async` |
| `Yahoo` (yfinance) | EOD bars (deep history), float / shares outstanding, earnings calendar fallback | No official API; ~1 req/s; occasionally rate-blocked → retry with backoff; used for backfill and as redundancy |
| `FINRA` | Biweekly SI %, SI shares; days-to-cover derived | Settlement dates are the 15th & last business day, but a report is not *published* until ~8 business days later. The backtest reveals each record on `settlement_date + data.finra_publication_lag_bdays` (default 8) so it never acts on SI before it was public. Trading on data up to ~2 weeks stale is accepted (SI is a slow variable); revealing it *early* is not (that is lookahead). FTP/CDN bulk dump, no per-request limit |
| `Reddit (PRAW)` | r/wallstreetbets mention counts, hot/new/top samples | 60 req/min OAuth; per-ticker hourly aggregation |
| `Finnhub free` | Earnings calendar + actual / estimate EPS | 60 req/min; US only |

### Storage Split

- **Parquet on disk** — time-series data: `bars/YYYY-MM-DD/TICKER.parquet`, `options/YYYY-MM-DD/TICKER.parquet`, `sentiment/YYYY-MM-DD-wsb.parquet`, `short_interest.parquet`, `earnings_calendar.parquet`. Partitioned by date, sharded by ticker.
- **Postgres** — state and slow-changing reference: `universe`, `signals_daily`, `setup_classifications`, `positions`, `orders`, `pnl_daily`, `factor_weights_history`, `kill_switch_events`.

### Critical Design Decisions

- **Replay is itself a `DataProvider`.** `BacktestProvider` reads parquet history; the rest of the system cannot tell whether it is live or replaying. Backtest and live therefore share **one** signal-computation code path.
- **Time machine.** `BacktestProvider` holds a `clock`; every read implicitly applies `WHERE ts ≤ clock`, eliminating lookahead bias by construction. The cutoff is the time the data became *knowable*, which is not always the event timestamp: short interest is gated on its FINRA *publication* date (`settlement_date + data.finra_publication_lag_bdays`), not its settlement date, because the report is not public until ~8 business days after settlement.
- **Idempotent ingestion.** Every ingest job is rerunnable; primary-key dedup + parquet append-with-dedup. Crash recovery is "rerun the job."
- **Cache invalidation rules:** bars are immutable once closed (append-only); quotes are not cached; options chain expires after 60s; short interest is keyed to settlement dates (15th & last business day) but only becomes visible on its FINRA publication date ~8 business days later; earnings invalidates when `actual` is published; sentiment expires after 1h.

### Paid-Source Upgrade Path (no business-code changes)

- `Polygon` replaces `Yahoo` → real-time options + better EOD quality
- `Ortex` replaces `FINRA` → near-real-time SI / CTB / locate availability
- `Quiver` replaces `Reddit` → multi-source sentiment (Reddit + Twitter)

Switching is a config change, not a code change.

## 5. Backtest & Validation

### Engine

A custom **bar-based loop** — no `vectorbt` / `backtrader` / `zipline`. The loop reuses all production code via `BacktestProvider` + `SimulatorBroker`:

```python
for date in trading_days[start:end]:
    clock.advance_to(date)
    universe   = build_universe(clock)
    factors    = signals.compute_all(universe, clock)
    score      = score.combine(factors)
    candidates = score.rank().filter(score >= 8.0)
    setups     = score.classify(candidates)
    decisions  = risk.gate(setups, portfolio_state)
    sim_broker.execute(decisions, slippage_model, commission_model)
    sim_broker.mark_to_market(date)
    portfolio_state = update(portfolio_state, sim_broker.fills)
    metrics.snapshot(date, portfolio_state)
```

### Time Splits (Walk-Forward)

```
2018──2019──2020 │ 2021 ── 2022 ── 2023 ── 2024 │ 2025-05 → 2026-05
   train (3y)    │ test (1y) test (1y) test (1y)│      HOLDOUT
                                                 │  never seen by tuner
```

Tuning rules:

- Factor weights are discrete: `{0, 0.5, 1.0, 1.5, 2.0}` only — 5⁷ ≈ 78,000 combos, traversed via Bayesian search ≤ 200 evaluations.
- Any parameter (thresholds, Kelly fraction, stops) is tuned **only on train**.
- Each `test` year is run **once** during walk-forward; numbers are not used to re-tune.
- The **holdout** segment is run **once at the end**; its numbers must not feed back into any decision.

### Anti-Overfitting Hard Rules

1. **Discrete weights** {0, 0.5, 1.0, 1.5, 2.0}.
2. **50% rule** — walk-forward test Sharpe drop > 50% from train → reject parameter set.
3. **Deflated Sharpe** (López de Prado) penalty applied for the number of combos tried.
4. **Random shuffle test** — for the realized equity curve, permute its daily returns 200×; for each permutation, fit an OLS trend on the log equity series and compute the slope's t-statistic. The real strategy's t-statistic must exceed the 95th percentile of the permutation distribution. (Sharpe alone is order-invariant under permutation and would not detect timing skill; the OLS-trend t-statistic is order-sensitive and so distinguishes a real edge from noise.)
5. **Captured-the-event check** — the validation set must hit ≥ 5 of 8 historical squeeze events (entered within 5 trading days of event start).
6. **Holdout veto** — any holdout metric below `train − 1×SD` → entire strategy goes back for redesign.

### Cost Model (deliberately conservative)

- Commissions: $0.005/share × 2 (round trip), IBKR Tiered Pro
- Slippage (stock): 5 bps / side at price ≥ $10; 15 bps / side at $5–$10; +10 bps during the first 5 minutes of the session
- Slippage (options): cross from mid to mid + ¼ × spread (no assumption of midpoint fill)
- Borrow: not applicable (no shorting)
- Halts: positions held during a halt are marked at the first post-restart print
- Overnight gaps: positions are marked at next-session open price

### Validation Case Set (exactly 8 events)

Recent (2024–2026, primary — 5 events):

- **CAR (Apr 2026)** — Avis: heavy call buying + short-squeeze chatter (CAR-type, freshest)
- **HTZ (Apr 2025)** — Hertz: ~85% single-day squeeze on Q4 + 2026 guidance (CAR-type)
- **OKLO (Jan–Feb 2025)** — clean energy + policy tailwind + high SI (Mixed)
- **GME (May 2024)** — Roaring Kitty return (GME-type)
- **TUP (May 2024)** — meme rally (GME-type)

Anchors (older, baseline — 3 events):

- **CAR (Nov 2021)** — original CAR-type benchmark (2 Nov 2021, +108% close-to-close; this entry originally said Aug 2022, but the cached bars show no event that month — max +5.4%)
- **GME (Jan 2021)** — original GME-type benchmark
- **BBBY (2022)** — Mixed / Ryan Cohen catalyst

Distribution: 3 CAR-type + 3 GME-type + 2 Mixed. The 5/8 hit threshold means the strategy must capture at least one event from each setup family.

`DJT (2024 multiple)` and `AMC (2021)` were considered but excluded — DJT had several overlapping events that complicate single-event accounting, and AMC's 2021 episode is dominated by the same retail wave that drives GME 2021 and so is not independently informative.

`FFIE (May 2024)` is deliberately **out** because its market cap fell below the $200M universe floor; the system should detect it but the universe filter must reject it. This is used as a negative-control case for universe correctness, not as part of the 8-event hit count.

### Three Promotion Gates (all conditions required)

| Gate | From → To | Conditions |
| --- | --- | --- |
| **1** | Backtest → Paper | Holdout Sharpe ≥ 1.0; Sortino ≥ 1.5; MaxDD ≤ 25%; win-rate ≥ 30%; avg payoff ≥ 1.5; captured-the-event hits ≥ 5/8; random-shuffle p < 0.05; deflated Sharpe ≥ threshold (default `0.0`; tighten to `0.3+` only when `n_trials ≤ 30`) |
| **2** | Paper → Small Live ($2–5K) | 30 calendar days of paper; Sharpe ≥ 0.5; 0 kill-switch triggers; 0 broker/data outages > 1h; deviation from backtest expectation < 50% |
| **3** | Small Live → Normal Size | 60 calendar days of small live; Sharpe ≥ 0.5; 0 operational incidents; total P&L ≥ 0 (net positive or flat) |

### Reported Metrics

- Annualized return, Sharpe, Sortino, Calmar, Max DD, time-in-market
- Win rate, average payoff, average holding period
- Per-setup-type breakdown (CAR / GME / Mixed)
- Per-event tracking on the 8 validation cases (entry, exit, return, contributing factors)
- Monthly equity curve and drawdown curve

## 6. Risk & Execution

### Pre-Trade Risk Gates (in order; any failure rejects the trade)

1. `squeeze_score ≥ 8.0`
2. `setup_class ≠ "Weak"`
3. Kill-switch is not active
4. New positions opened today < 3
5. Concurrent open positions < 6
6. Ticker not already held
7. Post-trade single-position weight ≤ 8% of equity
8. Post-trade gross exposure ≤ 90% (preserves 10% cash buffer; no leverage)
9. Ticker `ADV20$ ≥ 100 ×` planned position size (liquidity to exit cleanly)
10. Ticker is not currently halted
11. Days since IPO ≥ 30
12. Ticker is in the current universe
13. Post-trade 90-day correlation with existing portfolio ≤ 0.7
14. Earnings within next 3 trading days → size **halved** (not rejected, but de-risked for event)

### Position Sizing

```
# Per-setup-type rolling 6-month observed win_rate (p_obs) and payoff (b_obs)
# from live + paper trades; computed separately for CAR / GME / Mixed.
kelly_raw  = (p_used × b_used - (1 - p_used)) / b_used
kelly_used = clip(0.20 × kelly_raw, 0, 0.08)
position_$ = equity × kelly_used

# Bayesian shrinkage when sample is thin (n < 30 trades).
# Priors come from Gate 1 backtest holdout, NOT from fixed numbers,
# so the system bootstraps from realistic estimates rather than
# pathological pessimism that would forbid all trading at n = 0.
prior_win_rate = p_holdout[setup_type]    # set at Phase 3 start, frozen
prior_payoff   = b_holdout[setup_type]    # set at Phase 3 start, frozen
weight   = n / (n + 30)
p_used   = weight × p_obs + (1 − weight) × prior_win_rate
b_used   = weight × b_obs + (1 − weight) × prior_payoff
```

**For GME-type entries**, `position_$` is split: 70% to stock, 30% to OTM-call premium (strike ≈ 1.1× spot, 30–60 days to expiry). Stock stops apply to the stock leg; option stops (Section 6 stop stack) apply independently to the call leg.

### Stop Stack (stocks)

1. **Hard stop -12%** set immediately after entry
2. **Trailing stop** — CAR: -20% from peak; GME: -25% from peak
3. **Time stop** — 21 trading days, regardless of P&L
4. **Signal decay** — `squeeze_score` decay ≥ 50% from entry → halve position; ≥ 75% → exit
5. **Catalyst fizzle** — earnings-driven entry: if no follow-through within 5 days → exit

### Stop Stack (options)

1. **Premium stop -50%** of entry premium → close
2. **Take-profit +100%** → close half
3. **Days-to-expiry ≤ 14** → roll or close (avoid theta acceleration)
4. Follow the underlying stock's signal-decay rules

### Entry Execution (TWAP slicing)

```
09:30–09:35  no orders (avoid open-auction noise)
09:35–09:55  TWAP 6–8 slices, ~150–180s apart
              limit price = mid + 0.5 × spread (passive, not chasing)
              after 5 slices unfilled → switch to mid + 1 × spread on remaining
              after 80% of slices unfilled → switch to marketable limits
              hard cap: at 09:55, sweep remaining size with marketable limits
```

### Exit Execution

- Stop / time-stop triggers → market order, immediately
- Take-profit / position trim → VWAP into 4 slices over 5 minutes (don't over-take initiative)

### Kill-Switch

**Trigger conditions** (any one):

- Rolling monthly drawdown ≤ -10%
- 3 consecutive trading days with cumulative P&L ≤ -5%
- Single position gaps through stop ≥ -25%
- Broker connection lost > 5 minutes during market hours
- A critical data-source failure persists > 2 hours

**Behaviour:**

- Stop opening new positions immediately
- **Existing positions continue to be managed by their stops** (no panic-flatten)
- Push alert (Telegram + Slack + email)
- Auto-resume after 7 calendar days OR explicit manual reset

**Manual emergency flatten** (separate, not automatic): `squeeze-hunter emergency-flatten --confirm` → market-flatten everything immediately.

### Halt Handling

- Held name halted: no actions until reopen; on reopen, re-evaluate `squeeze_score` and follow signal-decay rules.
- Candidate name halted: skip entry today; re-evaluate next day.

### Order State Machine

```
PENDING → ROUTED → PARTIAL → FILLED
        ↘ REJECTED  ↘ CANCELLED  ↘ EXPIRED
```

Reconciliation:

- Every 60 seconds during market hours: internal state vs broker positions
- EOD: full reconciliation; any drift triggers an alert
- All operations idempotent: orders carry a client-side `order_id` that is deduped on submit

## 7. Operations & Rollout

### Phased Rollout

| Phase | Content | Duration |
| --- | --- | --- |
| **0. Bootstrap** | uv project skeleton (src layout), ruff/ty/pytest, pre-commit; docker-compose with postgres + ib-gateway + prometheus + grafana; ib-async hello-world (paper); Postgres schema + first Alembic migration | ~1 week |
| **1. Data + Signals + Score** | DataProvider Protocol + 5 free providers; 7 signals + z-score combiner + setup classifier; historical backfill 2018–2026 → parquet; full-universe daily scan working end-to-end | ~2–3 weeks |
| **2. Backtest + Walk-Forward** | `BacktestProvider`, bar-based loop, slippage / commission / option spread / halt / gap simulation; walk-forward over 2018–2024 train/test, 2025-05+ holdout; the 6 anti-overfitting rules; **Gate 1 evaluation** | ~1–2 weeks |
| **3. Paper Trading** | `IBKRBroker` against paper account; full schedule + execution + monitoring online; **Gate 2 evaluation** | 30 calendar days |
| **4. Small Live ($2–5K)** | Switch to IBKR live account; same risk gates; **Gate 3 evaluation** | 60 calendar days |
| **5. Normal Size / Continuous Improvement** | Scale by equity; monthly backtest re-run watching for decay (auto-alert); quarterly review of paid-data ROI | ongoing |

Total time to Gate-3 pass (normal size unlocked): approximately **5–6 months**, of which 90 days is paper + small-live observation.

### Deployment Topology

**MVP (local, always-on machine):**

```
docker-compose.yml
├── postgres:14
├── ib-gateway          (Docker, IBC for auto-login)
├── prometheus
├── grafana
└── squeeze-hunter      (the app, with /health and /metrics)

Volumes:
├── ~/squeeze-hunter/data/parquet/    bars, options, sentiment
├── ~/squeeze-hunter/data/postgres/   state
├── ~/squeeze-hunter/logs/            JSON logs, 30-day retention
└── ~/squeeze-hunter/backups/         daily pg_dump + parquet
```

**Cloud upgrade path (after Phase 4):** DigitalOcean droplet in US-East ($40–80/month) — same docker-compose, block storage for persistence, Tailscale for remote access.

### Logging

- `structlog` → JSON lines
- Fields: `ts`, `level`, `component`, `ticker?`, `correlation_id?`, `msg`, plus structured kwargs
- Components: `data` / `signals` / `score` / `risk` / `execution` / `broker` / `monitor`
- Daily rotation, 30-day retention

### Prometheus Metrics

App-level:

```
sh_signals_computed_total{factor}
sh_score_distribution_quantile{q}
sh_candidates_total{setup_type}
sh_orders_submitted_total{side, status}
sh_orders_filled_total{side}
sh_position_count
sh_gross_exposure_pct
sh_equity_usd
sh_daily_pnl_usd
sh_drawdown_pct
sh_kill_switch_active{reason}
```

Infra-level:

```
sh_data_provider_latency_seconds{provider, op}
sh_data_provider_errors_total{provider, error}
sh_broker_connected
sh_db_connected
```

### Alert Tiering

- **Telegram** (immediate, mobile): kill-switch trigger, broker disconnect > 1 min, single position gap-through-stop, any ERROR-level log
- **Slack** (work hours): nightly candidate list, daily P&L summary (post-close), Gate evaluations
- **Email** (low frequency): weekly / monthly review, important config changes

### Health Checks & Backups

- `GET :8080/health` returns DB / IBKR / data-freshness / kill-switch state
- `GET :8080/metrics` Prometheus exporter
- Daily `pg_dump | gzip → ~/backups/YYYY-MM-DD.sql.gz`
- Weekly cold backup: tar `parquet + sql` → encrypt → off-machine (rclone to B2 / S3 / NAS)
- **Disaster-recovery dry run** scheduled at the end of Phase 1 — wipe local, restore from backup, verify

### Engineering Discipline

- Python 3.12+ with `uv` (lockfile pinned)
- `ruff` (format + lint), `ty` (type check), enforced via pre-commit
- `pytest` with line coverage ≥ 70% overall, ≥ 90% for `signals` / `risk` / `score` / `execution`
- Per-commit: format / lint / typecheck / unit tests + a micro backtest end-to-end smoke test
- All magic numbers live in YAML config — none inlined in code
- Secrets in `.env` (IBKR creds, Telegram token, Slack webhook, …) — never in git
- Conventional Commits

## 8. Open Questions / Future Work

These are deliberately deferred from v1; capturing them so they're not forgotten:

- **Paid-data ROI evaluation.** Quarterly review whether Polygon / Ortex / Quiver are worth the spend. Decision criterion: paid-source delta on holdout Sharpe ≥ +0.3 to justify subscription.
- **ML score replacement.** Keep `combiner.score()` interface stable; once we have ≥ 12 months of out-of-sample trade outcomes, evaluate XGBoost / LightGBM as a drop-in. Class imbalance remains a hard problem.
- **Microstructure factor for hard-to-borrow.** Once a paid source for CTB rate is plugged in, weight 1.5 (placeholder slot). Without it, do not pretend.
- **Multi-account / multi-strategy.** Out of scope for v1. Architecture allows it but config and risk gates would need extension.
- **Streaming compute.** If signal-decay reaction time matters, replace 60s polling with WebSocket-driven event loop. Not warranted unless paper trading shows we are missing fast unwinds.
- **Gate failure response.** If Gate 1 fails repeatedly, do we (a) tighten universe / score threshold, (b) drop a setup type, or (c) abandon the strategy? Decision criterion to be set when we see actual failure-mode data.
- **Phase 0–2 implementation deviations.** The first implementation pass surfaced minor corrections that are now in code but not in the original spec wording: (a) `BacktestProvider.fetch_bars` now clamps cached OHLC to the `low ≤ open,close ≤ high` invariant before constructing `Bar` records (handles upstream Yahoo data quirks); (b) `score.classifier.classify_setups` tolerates missing factor columns by treating them as zero, so partial-coverage scans don't crash; (c) the Bollinger-breakout signal evaluates the squeeze condition on the pre-breakout window, not the breakout day's own (post-expansion) band-width. Each is logged in the corresponding commit on `main`.
- **Round-12 deviations (Sep 2026).** (a) The classifier's CAR/GME other-axis cutoff is 3.0, not 2.0 (R9.8), and both cutoffs now come from `score.setup_thresholds`. (b) The CAR Kelly payoff prior is 3.5, not 3.0, so raw Kelly stays positive. (c) `f4_wsb_mention` carries weight 0.0 until a Reddit baseline cache exists, so total weight is 8.5. (d) Exits are marketable limits (0.995 × price), not market orders; the signal-decay halve is a one-shot per position. (e) Deferred to Phase 4, not implemented: the 70/30 stock + OTM-call split and option-leg stops, the catalyst-fizzle stop, VWAP take-profit slices, Postgres as a runtime state store / position persistence across restarts, and the walk-forward weight search with its 50%-drop and holdout-veto rules. (f) The universe is the static `config/universe.txt` list; the backtest applies the price floor and the ADV20 liquidity gate from real bars, but market cap, listing age (placeholder 365 d), halts and pairwise correlations have no data source yet. (g) Gate 1 captured-events are counted over the union of all out-of-sample windows (test windows + holdout). Event dates in `backtest.validation_events` are the largest close-to-close move in the month this section names; the CAR anchor is 2 Nov 2021 (see §5). (h) The backtest provider clock sits at 23:59:59 UTC of each trading day so bars stamped 04:00/05:00 UTC are visible on their own label. (i) `/metrics` and `/health` are served by a stdlib HTTP server on `monitor.http_port`; a killswitch trip pushes a Telegram/Slack alert when the env vars are set.

## 9. Validation Sources

The validation case set was confirmed against public reporting:

- FFIE 2024 short squeeze rally — investorplace.com
- TUP 2024 short squeeze — investorplace.com
- GME 2024 Roaring Kitty squeeze — cgaa.org
- BYND 2024 short squeeze — investorplace.com
- OKLO short squeeze 2025 — barchart.com
- HTZ 2025 April short squeeze — seekingalpha.com
- CAR April 2026 short squeeze chatter — 247wallst.com

Original anchors (GME 2021, CAR 2022, BBBY 2022, AMC 2021) are well-documented in mainstream financial press.
