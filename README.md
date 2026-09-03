# squeeze-hunter

Quantitative trading system for US short-squeeze events. Targets both GME-type
(gamma + retail-driven) and CAR-type (mechanical short-cover on fundamental
catalysts) setups via a unified scoring model. Long stock + long options only;
no shorting.

**Status:** Phase 0–3 complete. The system is ready for end-to-end backtesting
and paper trading. Live trading begins after Gate 1 (backtest) and Gate 2
(30-day paper) pass.

## Documentation

- **Design spec:** `docs/superpowers/specs/2026-05-10-squeeze-hunter-design.md`
  — overview, architecture, signal model, data layer, backtest methodology,
  risk and execution, operations.
- **Implementation plans:**
  - `docs/superpowers/plans/2026-05-10-squeeze-hunter-phase-0-2.md` — bootstrap,
    data + signals + score, backtest engine.
  - `docs/superpowers/plans/2026-05-11-squeeze-hunter-phase-3-5.md` — paper
    trading scaffolding, small-live ramp, scale-up.
- **Runbooks:** `docs/runbooks/disaster-recovery.md`.

## Quick start

```bash
# 1) install
uv sync --all-extras

# 2) bring up infra (postgres + prometheus + grafana + ib-gateway)
docker compose -f docker/compose.yml up -d

# 3) apply DB schema
SH_DB_URL=postgresql+psycopg://squeeze:squeeze@localhost:5432/squeeze \
  uv run alembic upgrade head

# 4) configure
cp .env.example .env   # then edit IBKR / Finnhub / Reddit / Telegram credentials

# 5) sanity-check the IBKR connection
uv run squeeze-hunter hello AAPL
```

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
uv run squeeze-hunter ingest finra
uv run squeeze-hunter ingest earnings

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
  --holdout 2025-05-01:2026-05-01 \
  --n-trials 1

# 4) read data/backtests/gate1_report.txt
```

If Gate 1 passes, start paper trading: `uv run squeeze-hunter paper`.
After 30 days of clean paper trading (Gate 2), promote to small live trading.

## Architecture

Modular monolith in Python 3.12+ with `src` layout. Three core abstractions
let the same code path serve backtest, paper, and live:

- **`IBroker` Protocol** — `IBKRBroker` / `PaperBroker` / `SimulatorBroker`.
- **`DataProvider` Protocol** — `BacktestProvider` (parquet replay), `YahooProvider`,
  `FinraProvider`, `RedditProvider`, `FinnhubProvider`.
- **Pure-function signals** — `(tickers, provider, clock) -> Factor`.

Postgres stores state and reference; parquet stores time-series history.

```
src/squeeze_hunter/
├── data/        # schemas, Protocol, parquet cache, 5 providers
├── universe.py  # filter & rebuild universe
├── signals/     # 7 factors + cross-sectional z-score + orchestrator
├── score/       # weighted combiner + setup classifier
├── risk/        # Kelly + 14 gates + stops + killswitch
├── execution/   # OMS + TWAP slicer + lifecycle daemon
├── broker/      # IBroker Protocol + 3 implementations
├── backtest/    # bar-based runner + walk-forward + metrics + Gate 1
├── monitor/     # Prometheus exporter + health + Telegram/Slack alerts
├── store/       # Postgres ORM + Alembic
├── runtime.py   # RuntimeContext (paper/live/sim) + PortfolioTelemetry
├── scheduler.py # APScheduler 7-job graph
└── cli.py       # typer entry points
```

## Engineering

- **Tooling:** `uv` (deps + lockfile), `ruff` (format + lint), `ty` (type check),
  `pytest`, `structlog`. Pre-commit hooks run ruff, ruff-format, and ty on every
  commit; pytest runs on pre-push.
- **Tests:** 308 unit + integration tests (run with `uv run pytest`). Each
  reviewed bug fix has a regression test that would have caught the original
  bug. Aim for ≥ 90% line coverage on `signals/`, `risk/`, `score/`, `execution/`.
- **Config:** all magic numbers live in `config/settings.example.yml`. Override
  via env vars (`SH_*` prefix, `__` for nesting).
- **Secrets:** `.env`, gitignored.

## Phase milestones

| Tag                       | Meaning                                                      |
| ------------------------- | ------------------------------------------------------------ |
| `phase-0-bootstrap`       | uv project, ruff/ty/pytest, ORM, docker, IBKR hello-world    |
| `phase-1-scan`            | 7 signals + score + setup classifier + scan CLI + ingest     |
| `phase-2-backtest`        | walk-forward + Gate 1 verdict CLI                            |
| `phase-3-paper-ready`     | live broker, OMS, TWAP, lifecycle, killswitch, monitor       |
| `phase-4-live-ramp`       | (after first round-trip on small live capital)               |
| `phase-5-scale`           | (after first equity-tier promotion)                          |

## License

Private. All rights reserved.
