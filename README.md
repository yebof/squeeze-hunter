# squeeze-hunter

Quantitative research and trading system for US short-squeeze events. It uses
public data to identify stocks where a short-cover move may be building, and
wraps every trade in a conservative, rule-based risk framework (position caps,
a layered stop stack, a global killswitch) so that losses stay bounded. Long
stock and long options only; the system never shorts.

## Purpose and disclaimer

**This project exists for education and research.** It studies three questions:
whether short-squeeze dynamics can be detected from public data, what the real
risk of an event-driven, high-volatility strategy looks like, and whether a
conservative rule-based risk framework (fractional Kelly sizing, pre-trade
gates, a layered stop stack, a global killswitch) keeps losses bounded. It is a
learning and risk-discovery tool, not a product that promises returns.

- **Not investment advice.** Nothing in this repository — code, configuration,
  backtest output, scan output or documentation — is a recommendation or
  solicitation to buy or sell any security. The author is not a registered
  investment adviser or broker-dealer and manages no one else's money.
- **Trading is risky; you can lose all of your capital.** Short-squeeze names
  are extremely volatile. Backtested performance is not indicative of future
  results, and backtests are limited by data errors, survivorship bias and
  slippage assumptions.
- **Your own account, your own responsibility.** Users must comply with the
  laws of their jurisdiction, exchange rules, their broker's (IBKR) API terms
  and the terms of every data source used (Yahoo Finance, FINRA, Reddit,
  Finnhub), and handle their own taxes.
- **No market manipulation.** The system only reads public data. It does not
  post, promote or coordinate anything. Using it for any form of market
  manipulation or other unlawful activity is prohibited.
- **Provided "as is", without warranty of any kind; the author accepts no
  liability.** The default mode is paper trading; live trading requires an
  explicit `--confirm-real-money` flag.

## Status (September 2026)

- Phases 0–3 are complete and tagged: data layer, signals and scoring, backtest
  engine, paper-trading runtime.
- Twelve rounds of code review and bug fixes are merged on `main`; 355 tests
  pass.
- **The next steps are operator-driven:** backfill FINRA short interest and
  earnings data, run the walk-forward backtest, read the Gate 1 verdict. Gate 1
  unlocks 30 days of paper trading (Gate 2); only after that does small live
  capital (Phase 4) come into question.
- Only daily bars are on disk right now. The short-interest factors (f1, f2)
  and the earnings-reaction factor (f3) are dead until those backfills run.

## Strategy thesis

Two kinds of squeeze are scored by one model:

| Type | Driver | Examples | Main signals |
| --- | --- | --- | --- |
| CAR-type | Mechanical short cover on a fundamental catalyst (earnings, guidance) | HTZ Apr 2025, CAR Nov 2021 | High short interest + post-earnings price/volume reaction |
| GME-type | Option gamma + retail sentiment | GME Jan 2021, GME May 2024 | Community mention counts + call open-interest velocity |

Scores are linear weighted sums of cross-sectional z-scores; setup labels come
from a rule-based classifier; weights are fixed from backtest. The design
principles, in order: conservative bias, few high-quality data sources, simple
and stable methods, one code path for backtest / paper / live, and one config
file for every tunable.

## Pipeline

```
parquet history ──> 7 factors ──> cross-sectional z ──> weighted score ──> setup label (CAR / GME / Mixed / Weak)
       │                                                                          │
       │                        Kelly sizing ──> 14 pre-trade gates ──> entry at the next open
       │                                                                          │
       v                                                                          v
 backtest: SimulatorBroker                     paper / live: IBKR + lifecycle daemon (every 60 s)
                                                    stop stack ──> killswitch ──> /metrics, /health, alerts
```

The `IBroker` and `DataProvider` Protocols guarantee that backtest, paper and
live share the same signal, risk and stop code.

## Module map

| Path | Responsibility |
| --- | --- |
| `data/` | Domain schemas, the `DataProvider` Protocol, the parquet cache, five providers (parquet replay, Yahoo, FINRA, Reddit, Finnhub) |
| `ingest/` | One-shot backfills: bars, FINRA short interest (merged with Yahoo float), Finnhub earnings calendar |
| `signals/` | The seven factor functions, cross-sectional z-scoring, concurrent orchestration |
| `score/` | Weighted combiner and rule-based setup classifier |
| `universe.py` | Universe filter (not yet wired into the pipeline; see limitations) |
| `risk/` | Kelly sizing, pre-trade gates, stop stack, killswitch |
| `execution/` | Lifecycle daemon (stops, pending-exit reconciliation); OMS + TWAP slicer (wired in Phase 4) |
| `broker/` | `IBroker` Protocol with IBKR live, IBKR paper and a deterministic simulator |
| `backtest/` | Trading-day runner, cost model, walk-forward split, metrics, Gate 1 verdict |
| `runtime.py` | `RuntimeContext` (sim / paper / live), portfolio telemetry, killswitch state machine |
| `scheduler.py` | The seven APScheduler jobs |
| `monitor/` | Prometheus registry, health snapshot, `/metrics` + `/health` server, Telegram / Slack alerts |
| `store/`, `alembic/` | Postgres ORM and migrations (schema only; unused at runtime) |
| `cli.py` | typer entry points |

## Signals and scoring

| Factor | Meaning | Source | Weight | State |
| --- | --- | --- | --- | --- |
| f1 `si_pct` | Short interest as a share of float | FINRA + Yahoo float | 2.0 | needs `ingest finra` |
| f2 `days_to_cover` | Short shares / 20-day volume | FINRA | 1.0 | needs `ingest finra` |
| f3 `earnings_reaction` | Gap × volume within 5 trading days of a report, linear decay | Finnhub + bars | 2.0 | needs `ingest earnings` |
| f4 `wsb_mention` | Reddit mention z-score | Reddit | 0.0 | weight stays 0 until a baseline cache exists |
| f5 `call_oi_velocity` | 5-trading-day change in ATM call open interest | option chains | 1.5 | returns 0 until an options ingest job exists |
| f6 `bollinger_breakout` | Band squeeze followed by a close above the upper band | bars | 1.0 | live |
| f7 `volume_spike` | Today's volume / 20-day average | bars | 1.0 | live |

- Each factor is z-scored across the universe (NaN / Inf excluded, clipped to
  ±3), then summed with the YAML weights. The score threshold (8.0) is applied
  by the pre-trade gates.
- Classifier: A = z(f1) + z(f3), B = z(f4) + z(f5). CAR if A ≥ 4 and B < 3;
  GME if B ≥ 4 and A < 3; Mixed if both ≥ 3; otherwise Weak (never traded).
  Cutoffs come from `score.setup_thresholds`.

## Risk layer

**Sizing.** Fractional Kelly with `kelly_fraction` 0.20, an 8% position cap, at
most 6 positions, 3 new positions per day and 90% gross exposure. Priors are
per setup type — CAR (0.25, 3.5), GME (0.15, 8.0), Mixed (0.20, 5.5) — and
Weak sizes to zero.

**Pre-trade gates (14 plus an equity guard).** Non-positive equity, score below
threshold, Weak setup, killswitch active, daily new-position cap, max
positions, already held, earnings within 3 trading days (halves size instead
of rejecting), position cap, gross-exposure cap, insufficient liquidity (ADV20
dollar volume below 100× the position), halted, listed for fewer than 30 days,
outside the universe, correlation too high.

**Stop stack (in order).** Hard stop at −12% → trailing stop per setup (CAR
20%, GME 25%, Mixed 22% from the peak, armed only once the peak exceeds cost)
→ 21-trading-day time stop → signal decay (halve once at ≥ 50% decay, exit at
≥ 75%). Live exits use a marketable limit 0.5% below the current price, never a
market order.

**Killswitch (five arms).** 30-day rolling drawdown ≤ −10%, three-day
cumulative loss ≤ −5%, any position gapping ≤ −25%, broker disconnected for
≥ 300 s of session time, critical data stale for ≥ 2 hours. A trip blocks new
entries for 7 calendar days while existing positions keep running their stops;
it pushes a Telegram / Slack alert and can be reset manually.

## Backtest and the three gates

The backtest iterates over NYSE sessions (weekends, exchange holidays and
special closures excluded, via `pandas_market_calendars`). The scan uses information available after the close; entries fill
at the next session's open with open-window slippage; stops are evaluated
against the day's low and filled at the low (conservative). FINRA short
interest becomes visible only after its publication lag (settlement date + 8
business days) to avoid look-ahead.

Walk-forward validation splits history into a training window, several test
windows and a holdout. Gate 1 is evaluated on the holdout, except that the
"captured historical events" criterion counts entries across the union of all
out-of-sample windows.

| Gate | From → to | Conditions |
| --- | --- | --- |
| 1 | Backtest → paper | Holdout Sharpe ≥ 1.0, Sortino ≥ 1.5, max drawdown ≤ 25%, hit rate ≥ 30%, average payoff ≥ 1.5, ≥ 5 of 8 historical squeeze events captured, random-shuffle p < 0.05, deflated Sharpe above threshold (≥ 0.3 when n_trials ≤ 30) |
| 2 | Paper → small live | 30 calendar days of paper, Sharpe ≥ 0.5, zero killswitch trips, zero broker/data outages over 1 h, deviation from backtest expectation < 50% |
| 3 | Small live → normal size | 60 calendar days, Sharpe ≥ 0.5, zero operational incidents, total P&L ≥ 0 |

The eight validation events live in `config/settings.example.yml` under
`backtest.validation_events`.

## Runtime and scheduling

`RuntimeContext` runs in one of three modes: `sim` (local simulator), `paper`
(IBKR paper account on port 7497) or `live` (IBKR real account). Scheduled
jobs, all in US Eastern time:

| Job | When | State |
| --- | --- | --- |
| `ingest_eod` | 17:00 | Phase 4 |
| `nightly_scan` | 22:00 | wired: runs the scan, publishes candidates, refreshes held positions' scores |
| `premarket_data` | 04:00 | Phase 4 |
| `premarket_verify` | 08:00 | wired |
| `intraday_loop` | every 60 s | wired: stops, mark-to-market, killswitch (09:30–16:00 only) |
| `moc_decision` | 15:55 | Phase 4 |
| `eod_close` | 16:30 | wired: advances bars held |

**Phase 3 does not auto-enter positions.** The nightly scan publishes
candidates to `last_candidates` for manual review; the automatic entry path is
Phase 4 work. The lifecycle daemon manages exits for positions that exist.

Once `paper` or `live` is running, `monitor.http_port` (default 8080) serves
`/metrics` (Prometheus text) and `/health` (JSON; 503 while unhealthy or while
the killswitch is active). `docker/compose.yml` brings up Postgres, Prometheus,
Grafana and an IB Gateway.

## Data and storage

- The parquet cache under `data/parquet/` (gitignored) holds
  `bars/<TICKER>.parquet`, `short_interest/all.parquet` and
  `earnings/all.parquet`.
- Backtest, paper and live scans all read the same parquet cache; keeping it
  current in paper / live mode needs a separate ingest job (Phase 4).
- Postgres and Alembic define a schema, but no runtime module reads or writes
  it yet.

## Quick start

```bash
# 1) install
uv sync --all-extras

# 2) bring up infra (postgres + prometheus + grafana + ib-gateway)
docker compose -f docker/compose.yml up -d

# 3) apply DB schema
SH_DB_URL=postgresql+psycopg://squeeze:squeeze@localhost:5432/squeeze \
  uv run alembic upgrade head

# 4) configure — the CLI loads .env automatically
cp .env.example .env   # then edit IBKR / Finnhub / Reddit / Telegram / Slack credentials

# 5) sanity-check the IBKR connection
uv run squeeze-hunter hello AAPL
```

## Credentials and external services

Copy `.env.example` to `.env` in the repository root and fill in what you
need. The CLI loads the nearest `.env` automatically and never overrides
variables already set in your shell. `.env` is gitignored; never commit it.

| Variable | Needed for | Where to get it |
| --- | --- | --- |
| `FINNHUB_KEY` | `ingest earnings` — the earnings calendar behind f3 | Sign up at [finnhub.io](https://finnhub.io), copy the API key from the dashboard. The free tier is enough for a 20-name universe. |
| `IBKR_HOST`, `IBKR_PORT`, `IBKR_CLIENT_ID`, `IBKR_ACCOUNT` | `hello`, `paper`, `live`, `emergency-flatten` | An [Interactive Brokers](https://www.interactivebrokers.com) account with a paper-trading sub-account (IDs start with `DU`). Run TWS or IB Gateway on the same machine and enable the API (Configure → API → Settings → *Enable ActiveX and Socket Clients*; add `127.0.0.1` to trusted IPs). Paper mode requires port **7497** — `PaperBroker` refuses any other port, so if you use IB Gateway set its API port to 7497. Live mode requires `IBKR_PORT` to be set explicitly (7496 for TWS live, 4001 for IB Gateway live) and refuses 7497. Leave `IBKR_ACCOUNT` empty unless you have several sub-accounts; a value this login does not manage is refused at connect. `IBKR_CLIENT_ID` is any small integer not used by another API client. |
| `IBKR_USERID`, `IBKR_PASSWORD` | Only the `ib-gateway` container in `docker/compose.yml` | Your IBKR paper-account login. Not read by the Python code. |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Killswitch alerts (high severity) | Create a bot with [@BotFather](https://t.me/BotFather) (`/newbot`) to get the token; send the bot a message, then read your chat id from `https://api.telegram.org/bot<TOKEN>/getUpdates`. Optional. |
| `SLACK_WEBHOOK_URL` | Low-severity alerts | Slack → *Apps* → *Incoming Webhooks* → add to a channel, copy the webhook URL. Optional. |
| `SH_DB_URL` | `alembic upgrade head` only | Leave the default for the local docker Postgres. The runtime does not read the database yet. |
| `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` | Reserved for the Reddit mention ingest behind f4, which is not implemented yet | [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) → create a *script* app. Nothing reads these today. |

Services that need no key: Yahoo Finance bars and float (via `yfinance`) and
the FINRA short-interest files (public monthly downloads from
`cdn.finra.org`; if that host answers 403 from your network, try another
network or the [FINRA API](https://developer.finra.org)). Every `SH_*`
variable of the form `SH_<SECTION>__<KEY>` overrides the matching YAML setting.

## CLI

```
squeeze-hunter --help

  hello              Connect to IBKR (paper) and print a quote.
  scan               Run a single-day scan against the parquet cache.
  backtest           Run walk-forward backtest and produce a Gate 1 verdict.
  ingest             Historical backfill commands (bars / FINRA / earnings).
  paper              Run the paper-trading loop indefinitely.
  live               Run the live-trading loop. Requires --confirm-real-money.
  emergency-flatten  Market-flatten every open position. Requires --confirm.
```

## End-to-end validation

```bash
# 1) backfill historical data (one-time, slow)
uv run squeeze-hunter ingest bars --start 2018-01-01 --end 2026-05-10
uv run squeeze-hunter ingest finra      # exits non-zero if no FINRA file downloads
uv run squeeze-hunter ingest earnings   # needs FINNHUB_KEY

# 2) sanity-check a known squeeze date
uv run squeeze-hunter scan --date 2025-04-21
# HTZ should rank in the top 3 with setup_type=CAR

# 3) walk-forward backtest with Gate 1
# train_end must be <= the first test window's start (2021-01-01) or the
# walk-forward validator rejects the run as train/test leakage.
uv run squeeze-hunter backtest \
  --train-start 2018-01-01 --train-end 2020-12-31 \
  --test-window 2021-01-01:2021-12-31 \
  --test-window 2022-01-01:2022-12-31 \
  --test-window 2023-01-01:2023-12-31 \
  --test-window 2024-01-01:2024-12-31 \
  --test-window 2025-01-01:2025-04-30 \
  --holdout 2025-05-01:2026-05-01 \
  --n-trials 1
# The Gate 1 "captured >= 5 of 8 events" check counts entries across ALL test
# windows + holdout, so leave no gaps between windows.

# 4) read data/backtests/gate1_report.txt
```

If Gate 1 passes, start paper trading: `uv run squeeze-hunter paper`.
After 30 days of clean paper trading (Gate 2), promote to small live trading.

## Configuration

Every tunable lives in `config/settings.example.yml`: factor weights and the
score threshold, classifier cutoffs, Kelly and position limits, stops, universe
floors, the FINRA publication lag, the Gate 1 validation events and the monitor
port. Environment variables override YAML with the `SH_<SECTION>__<KEY>` form
(for example `SH_RISK__POSITION_CAP=0.05`). A misspelled key inside any section
fails loudly instead of being ignored.

## Engineering

- **Tooling:** `uv` (deps + lockfile), `ruff` (format + lint), `ty` (type
  check), `pytest`, `structlog`. Pre-commit hooks run ruff, ruff-format and ty
  on every commit; pytest runs on pre-push
  (`uv run pre-commit install && uv run pre-commit install --hook-type pre-push`).
- **Tests:** 355 unit + integration tests (`uv run pytest`), fully offline.
  Every reviewed bug fix has a regression test named after the finding.
- **Rules:** write the failing test first; catch only the exceptions you expect
  (`ConnectionError`, `TimeoutError`, `OSError`) and let programming errors
  propagate to `tick_safe`; keep internal timestamps in UTC; iterate backtests
  over trading days, not calendar days; never block the event loop with
  synchronous I/O.
- **Secrets:** `.env`, gitignored.

## Known limitations and next steps

- The universe is the hand-maintained 20-name list in `config/universe.txt`;
  `universe.build_universe` is not wired in. In the backtest the liquidity and
  price-floor gates use real bars, while the market-cap, listing-age, halt and
  correlation gates still receive placeholder inputs.
- f4 has weight 0 and f5 returns 0 without an options-chain ingest, so GME-type
  setups are barely detectable today; a passing Gate 1 may rest on CAR-type
  trades alone (the report prints a coverage warning).
- Not yet implemented from the spec: the 70/30 stock + call split and option-leg
  stops, the catalyst-fizzle stop, VWAP take-profit slices, and position
  persistence across restarts (state is in memory; Phase 4).
- Yahoo's float is today's float applied to every historical short-interest
  record (share counts are split-adjusted at ingest, the float itself is not
  reconstructed historically).
- Known external blockers: `cdn.finra.org` currently answers 403 from the
  development machine (try another network or the FINRA API), and the earnings
  backfill needs a Finnhub key.

Order of work: restore FINRA access → backfill short interest and earnings →
run the Gate 1 backtest → 30 days of paper trading → only then discuss live.

## Documentation

- **Design spec (source of truth):**
  `docs/superpowers/specs/2026-05-10-squeeze-hunter-design.md` — section 8
  lists every known deviation between spec and code.
- **Implementation plans:** `docs/superpowers/plans/`.
- **Runbooks:** `docs/runbooks/disaster-recovery.md`.
- **Conventions for AI coding assistants:** `CLAUDE.md`, `AGENTS.md`.

## Phase milestones

| Tag                       | Meaning                                                      |
| ------------------------- | ------------------------------------------------------------ |
| `phase-0-bootstrap`       | uv project, ruff/ty/pytest, ORM, docker, IBKR hello-world    |
| `phase-1-scan`            | 7 signals + score + setup classifier + scan CLI + ingest     |
| `phase-2-backtest`        | walk-forward + Gate 1 verdict CLI                            |
| `phase-3-paper-ready`     | live broker, OMS, TWAP, lifecycle, killswitch, monitor       |
| `phase-4-live-ramp`       | (after first round-trip on small live capital)               |
| `phase-5-scale`           | (after first equity-tier promotion)                          |

## Contributing and security

Pull requests are welcome; CI runs ruff, ty and the full test suite. Report
anything that could move money, leak credentials or bypass the risk controls
through GitHub's private vulnerability reporting (see `SECURITY.md`), not a
public issue.

## License

Apache License 2.0 — see `LICENSE`. The license's warranty disclaimer and
limitation of liability (sections 7 and 8) apply to everything in this
repository, in addition to the disclaimer above.
