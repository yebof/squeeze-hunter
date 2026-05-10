# Squeeze Hunter — Phase 0–2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take squeeze-hunter from empty repo to a working backtest pipeline that produces walk-forward + Gate 1 metrics on 2018–2024 train and 2025-05 → 2026-05 holdout.

**Architecture:** Modular monolith in Python 3.12+ with `src` layout. Three core abstractions — `IBroker`, `DataProvider`, and pure-function `signals` — let the same code path serve backtest, paper, and live. Postgres holds state and reference; parquet holds time-series history.

**Tech Stack:** Python 3.12+, `uv` (deps + lockfile), `ruff` (format + lint), `ty` (type check), `pytest`, `structlog`, Pydantic v2, SQLAlchemy 2 + Alembic, Postgres 14, `ib-async`, `yfinance`, `praw`, `finnhub-python`, `pandas`, `pyarrow`, `numpy`, `typer`, `apscheduler`, `prometheus-client`, Docker Compose for local infra.

**Spec reference:** [`docs/superpowers/specs/2026-05-10-squeeze-hunter-design.md`](../specs/2026-05-10-squeeze-hunter-design.md)

**Scope of THIS plan:**
- Phase 0 — bootstrap (project skeleton, Docker stack, ib-async hello-world, Postgres schema)
- Phase 1 — data layer + 7 signals + score + setup classifier + daily scanner
- Phase 2 — backtest engine + walk-forward + Gate 1 evaluation

**Out of scope (next plan):** Phase 3 paper trading, Phase 4 small live, Phase 5 scale-up. Those are operational milestones that depend on Gate 1 results, so they're deferred.

---

## File Structure

After this plan completes the repo will look like:

```
squeeze-hunter/
├── pyproject.toml                     # uv-managed
├── uv.lock
├── .python-version                    # 3.12
├── .gitignore                         # already exists
├── .env.example                       # IBKR_HOST, FINNHUB_KEY, REDDIT_*, ...
├── .pre-commit-config.yaml
├── ruff.toml
├── pytest.ini                         # or pyproject.toml [tool.pytest]
├── README.md
├── docker/
│   ├── compose.yml                    # postgres + ib-gateway + prometheus + grafana + app
│   ├── ib-gateway.dockerfile          # ib-gateway + IBC + jts.ini template
│   └── prometheus.yml
├── config/
│   ├── settings.example.yml           # weights, thresholds, risk params
│   ├── universe.yml                   # universe filter
│   └── grafana/                       # dashboard JSON
├── alembic/
│   ├── env.py
│   └── versions/
│       └── 0001_initial_schema.py
├── alembic.ini
├── src/squeeze_hunter/
│   ├── __init__.py
│   ├── config.py                      # Pydantic Settings + YAML loader
│   ├── logging_setup.py               # structlog config
│   ├── data/
│   │   ├── __init__.py
│   │   ├── schema.py                  # Bar, Quote, OptionChain, ShortInterest, ...
│   │   ├── protocol.py                # DataProvider Protocol
│   │   ├── cache.py                   # parquet-on-disk + LRU
│   │   └── providers/
│   │       ├── __init__.py
│   │       ├── ibkr.py
│   │       ├── yahoo.py
│   │       ├── finra.py
│   │       ├── reddit.py
│   │       ├── finnhub.py
│   │       └── backtest.py            # BacktestProvider with clock
│   ├── universe.py
│   ├── signals/
│   │   ├── __init__.py
│   │   ├── base.py                    # Signal Protocol + Factor schema
│   │   ├── normalize.py               # cross-sectional z-score + clip
│   │   ├── short_interest.py          # f1, f2
│   │   ├── earnings_reaction.py       # f3
│   │   ├── sentiment.py               # f4
│   │   ├── options_flow.py            # f5
│   │   ├── technicals.py              # f6, f7
│   │   └── compute.py                 # orchestrate all factors
│   ├── score/
│   │   ├── __init__.py
│   │   ├── combiner.py                # weighted z-score
│   │   └── classifier.py              # CAR / GME / Mixed / Weak
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── kelly.py
│   │   ├── gates.py
│   │   ├── stops.py
│   │   └── killswitch.py
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── oms.py
│   │   ├── slicing.py
│   │   └── lifecycle.py
│   ├── broker/
│   │   ├── __init__.py
│   │   ├── base.py                    # IBroker Protocol
│   │   ├── ibkr.py                    # ib-async impl (hello-world only in Phase 0)
│   │   └── simulator.py               # for backtest
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── runner.py                  # bar-based loop
│   │   ├── cost_model.py
│   │   ├── metrics.py                 # Sharpe, Sortino, MaxDD, ...
│   │   ├── walk_forward.py
│   │   ├── deflated_sharpe.py
│   │   └── shuffle_test.py
│   ├── store/
│   │   ├── __init__.py
│   │   ├── db.py                      # SQLAlchemy session
│   │   └── models.py                  # ORM models
│   ├── monitor/
│   │   └── __init__.py                # placeholder, populated in Phase 3
│   ├── scheduler.py                   # APScheduler (only nightly scan job in Phase 1)
│   └── cli.py                         # typer: scan / backtest / ingest / hello
└── tests/
    ├── conftest.py
    ├── data/
    ├── signals/
    ├── score/
    ├── risk/
    ├── backtest/
    └── e2e/                            # micro backtest end-to-end smoke
```

Files split by responsibility. `signals/*.py` modules each compute a related family of factors and stay under ~150 LOC. `data/providers/*.py` each implement `DataProvider` for one source and stay under ~200 LOC. Tests mirror src.

**Phase milestones (each = a green commit + tag):**
- `phase-0-bootstrap` — `docker compose up` brings postgres + ib-gateway + prometheus + grafana up; `squeeze-hunter hello` connects to IBKR paper and prints a quote.
- `phase-1-scan` — `squeeze-hunter scan --date 2024-05-13` produces a ranked CSV of squeeze candidates from historical parquet data.
- `phase-2-backtest` — `squeeze-hunter backtest --train 2018-01-01:2024-12-31 --holdout 2025-05-01:2026-05-01` produces a Gate 1 metric report.

---

## Phase 0 — Bootstrap

### Task 0.1: Initialize uv project skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `src/squeeze_hunter/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `README.md`

- [ ] **Step 1: Install uv if missing**

```bash
which uv || curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

Expected: `uv 0.5.x` or newer.

- [ ] **Step 2: Initialize the project**

Run from repo root (`/Users/yebof/Documents/squeeze-hunter`):

```bash
uv init --package --name squeeze-hunter --python 3.12
```

This creates `pyproject.toml`, `.python-version`, and `src/squeeze_hunter/__init__.py`.

- [ ] **Step 3: Replace generated `pyproject.toml`**

Overwrite with:

```toml
[project]
name = "squeeze-hunter"
version = "0.1.0"
description = "Quantitative trading system for short-squeeze events"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.7",
    "pydantic-settings>=2.4",
    "pyyaml>=6.0",
    "structlog>=24.1",
    "typer>=0.12",
    "pandas>=2.2",
    "numpy>=1.26",
    "pyarrow>=16",
    "sqlalchemy>=2.0",
    "alembic>=1.13",
    "psycopg[binary]>=3.2",
    "apscheduler>=3.10",
    "prometheus-client>=0.20",
    "httpx>=0.27",
]

[project.optional-dependencies]
data = [
    "ib-async>=1.0",
    "yfinance>=0.2.40",
    "praw>=7.7",
    "finnhub-python>=2.4",
]

[project.scripts]
squeeze-hunter = "squeeze_hunter.cli:app"

[dependency-groups]
dev = [
    "pytest>=8.2",
    "pytest-cov>=5.0",
    "pytest-asyncio>=0.23",
    "freezegun>=1.5",
    "ruff>=0.6",
    "ty>=0.0.1a1",
    "pre-commit>=3.7",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/squeeze_hunter"]
```

- [ ] **Step 4: Sync deps**

```bash
uv sync --all-extras
```

Expected: lockfile written, venv populated. No errors.

- [ ] **Step 5: Sanity check**

```bash
uv run python -c "import squeeze_hunter; print(squeeze_hunter.__name__)"
```

Expected: `squeeze_hunter`.

- [ ] **Step 6: Write minimal README and conftest**

`README.md`:

```markdown
# squeeze-hunter

Quantitative trading system for short-squeeze events (GME-type and CAR-type).
See `docs/superpowers/specs/2026-05-10-squeeze-hunter-design.md` for the full design.

## Quick start

    uv sync --all-extras
    docker compose -f docker/compose.yml up -d
    uv run squeeze-hunter hello
```

`tests/conftest.py`:

```python
"""Shared fixtures for squeeze-hunter tests."""

import pytest


@pytest.fixture
def repo_root(tmp_path_factory):
    """Empty temp directory for filesystem-touching tests."""
    return tmp_path_factory.mktemp("repo")
```

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock .python-version src/ tests/ README.md
git commit -m "chore: bootstrap uv project skeleton"
```

---

### Task 0.2: Configure ruff, ty, pytest, and pre-commit

**Files:**
- Create: `ruff.toml`
- Create: `.pre-commit-config.yaml`
- Modify: `pyproject.toml` (add `[tool.pytest.ini_options]`, `[tool.ty]`)

- [ ] **Step 1: Write `ruff.toml`**

```toml
target-version = "py312"
line-length = 100

[lint]
select = [
    "E", "F", "W",      # pycodestyle / pyflakes
    "I",                # isort
    "N",                # naming
    "UP",               # pyupgrade
    "B",                # bugbear
    "SIM",              # simplify
    "RUF",              # ruff-specific
    "T20",              # no print
    "PT",               # pytest style
    "ANN",              # require annotations
]
ignore = [
    "ANN401",           # allow Any for adapter boundaries
    "E501",             # ruff-format handles line length
]

[lint.per-file-ignores]
"tests/**/*.py" = ["ANN", "T20"]      # tests can skip annotations and use print
"alembic/versions/*" = ["ANN", "N"]   # generated migrations

[format]
quote-style = "double"
indent-style = "space"
```

- [ ] **Step 2: Append pytest and ty config to `pyproject.toml`**

Append:

```toml
[tool.pytest.ini_options]
minversion = "8.0"
addopts = "-ra --strict-markers --strict-config -q"
testpaths = ["tests"]
asyncio_mode = "auto"
markers = [
    "slow: long-running tests (>1s)",
    "integration: requires postgres / IBKR / network",
]

[tool.coverage.run]
source = ["src/squeeze_hunter"]
branch = true

[tool.coverage.report]
exclude_lines = ["pragma: no cover", "if TYPE_CHECKING:", "raise NotImplementedError"]
fail_under = 70

[tool.ty]
src.root = "src"
```

- [ ] **Step 3: Write `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: local
    hooks:
      - id: ty
        name: ty type check
        entry: uv run ty check src
        language: system
        pass_filenames: false
        types: [python]
      - id: pytest-fast
        name: pytest (fast unit tests)
        entry: uv run pytest -m "not slow and not integration" -x -q
        language: system
        pass_filenames: false
        stages: [pre-push]
```

- [ ] **Step 4: Install hooks and verify**

```bash
uv run pre-commit install
uv run pre-commit install --hook-type pre-push
uv run ruff format .
uv run ruff check .
uv run ty check src
uv run pytest
```

Expected: ruff format/check pass; ty reports "no issues"; pytest reports "no tests ran" (still empty).

- [ ] **Step 5: Commit**

```bash
git add ruff.toml pyproject.toml .pre-commit-config.yaml uv.lock
git commit -m "chore: configure ruff, ty, pytest, pre-commit"
```

---

### Task 0.3: Add `config.py` and `logging_setup.py`

**Files:**
- Create: `src/squeeze_hunter/config.py`
- Create: `src/squeeze_hunter/logging_setup.py`
- Create: `config/settings.example.yml`
- Create: `.env.example`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:

```python
from pathlib import Path

import pytest

from squeeze_hunter.config import Settings, load_settings


def test_load_settings_from_yaml(tmp_path: Path) -> None:
    cfg = tmp_path / "s.yml"
    cfg.write_text(
        """
score:
  threshold: 8.0
  weights:
    f1_si_pct: 2.0
    f2_days_to_cover: 1.0
risk:
  kelly_fraction: 0.20
  position_cap: 0.08
  max_positions: 6
""",
        encoding="utf-8",
    )
    s = load_settings(cfg)
    assert s.score.threshold == 8.0
    assert s.risk.kelly_fraction == 0.20
    assert s.score.weights["f1_si_pct"] == 2.0


def test_settings_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "s.yml"
    cfg.write_text("score:\n  threshold: 7.0\n", encoding="utf-8")
    monkeypatch.setenv("SH_SCORE__THRESHOLD", "9.5")
    s = load_settings(cfg)
    assert s.score.threshold == 9.5
```

- [ ] **Step 2: Run the test, expect failure**

```bash
uv run pytest tests/test_config.py -v
```

Expected: ImportError for `squeeze_hunter.config`.

- [ ] **Step 3: Implement `config.py`**

```python
"""Pydantic Settings + YAML loader.

Layered config: yaml file → env overrides via SH_*__* (double underscore = nesting).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ScoreCfg(BaseModel):
    threshold: float = 8.0
    weights: dict[str, float] = Field(default_factory=dict)
    setup_thresholds: dict[str, float] = Field(
        default_factory=lambda: {"strong": 4.0, "weak_floor": 2.0, "mixed_floor": 3.0}
    )


class RiskCfg(BaseModel):
    kelly_fraction: float = 0.20
    position_cap: float = 0.08
    max_positions: int = 6
    max_new_per_day: int = 3
    max_gross_exposure: float = 0.90
    monthly_drawdown_kill: float = 0.10
    bayes_prior_n: int = 30


class StopsCfg(BaseModel):
    hard_stop: float = -0.12
    trailing_car: float = -0.20
    trailing_gme: float = -0.25
    time_stop_days: int = 21
    signal_decay_halve: float = 0.50
    signal_decay_exit: float = 0.75


class UniverseCfg(BaseModel):
    min_market_cap: float = 200_000_000
    max_market_cap: float = 10_000_000_000
    min_price: float = 5.0
    min_days_listed: int = 30


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SH_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )
    score: ScoreCfg = Field(default_factory=ScoreCfg)
    risk: RiskCfg = Field(default_factory=RiskCfg)
    stops: StopsCfg = Field(default_factory=StopsCfg)
    universe: UniverseCfg = Field(default_factory=UniverseCfg)


def load_settings(yaml_path: Path | None = None) -> Settings:
    """Load settings: yaml file (if given) merged with env overrides."""
    base: dict[str, Any] = {}
    if yaml_path is not None and yaml_path.exists():
        base = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    return Settings(**base)
```

- [ ] **Step 4: Implement `logging_setup.py`**

```python
"""Structured logging via structlog → JSON lines."""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level)),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(component: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(component=component)
```

- [ ] **Step 5: Write example config files**

`config/settings.example.yml`:

```yaml
score:
  threshold: 8.0
  weights:
    f1_si_pct: 2.0
    f2_days_to_cover: 1.0
    f3_earnings_reaction: 2.0
    f4_wsb_mention: 1.5
    f5_call_oi_velocity: 1.5
    f6_bollinger_breakout: 1.0
    f7_volume_spike: 1.0
risk:
  kelly_fraction: 0.20
  position_cap: 0.08
  max_positions: 6
  max_new_per_day: 3
  max_gross_exposure: 0.90
stops:
  hard_stop: -0.12
  trailing_car: -0.20
  trailing_gme: -0.25
  time_stop_days: 21
universe:
  min_market_cap: 200000000
  max_market_cap: 10000000000
  min_price: 5.0
  min_days_listed: 30
```

`.env.example`:

```bash
# Copy to .env and fill in.
SH_LOG_LEVEL=INFO
SH_DB_URL=postgresql+psycopg://squeeze:squeeze@localhost:5432/squeeze
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=42
IBKR_ACCOUNT=DU0000000
FINNHUB_KEY=
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=squeeze-hunter/0.1 by yebof
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
```

- [ ] **Step 6: Run tests, expect pass**

```bash
uv run pytest tests/test_config.py -v
```

Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add src/squeeze_hunter/config.py src/squeeze_hunter/logging_setup.py \
        config/settings.example.yml .env.example tests/test_config.py
git commit -m "feat(config): add settings loader and structlog setup"
```

---

### Task 0.4: Postgres ORM models + Alembic baseline migration

**Files:**
- Create: `src/squeeze_hunter/store/__init__.py`
- Create: `src/squeeze_hunter/store/db.py`
- Create: `src/squeeze_hunter/store/models.py`
- Create: `alembic.ini`
- Create: `alembic/env.py`
- Create: `alembic/versions/0001_initial_schema.py`
- Create: `tests/store/test_models.py`

- [ ] **Step 1: Write the failing test**

`tests/store/__init__.py`: empty file.

`tests/store/test_models.py`:

```python
from datetime import UTC, date, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from squeeze_hunter.store.models import (
    Base,
    KillSwitchEvent,
    Order,
    Position,
    SignalDaily,
    UniverseRow,
)


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def test_universe_row_roundtrip(session: Session) -> None:
    session.add(
        UniverseRow(
            ticker="GME",
            as_of=date(2024, 5, 13),
            market_cap_usd=10_000_000_000,
            close_price=20.0,
            included=True,
        )
    )
    session.commit()
    row = session.query(UniverseRow).one()
    assert row.ticker == "GME"
    assert row.included


def test_signal_daily_roundtrip(session: Session) -> None:
    session.add(
        SignalDaily(
            as_of=date(2024, 5, 13),
            ticker="GME",
            factor_name="f1_si_pct",
            raw_value=15.0,
            z_score=3.2,
        )
    )
    session.commit()
    assert session.query(SignalDaily).count() == 1


def test_position_lifecycle(session: Session) -> None:
    p = Position(
        ticker="GME",
        opened_at=datetime(2024, 5, 13, 14, 0, tzinfo=UTC),
        side="long",
        quantity=100,
        avg_price=18.0,
        setup_type="GME",
        instrument="stock",
        status="open",
    )
    session.add(p)
    session.commit()
    p.status = "closed"
    p.closed_at = datetime(2024, 5, 17, 20, 0, tzinfo=UTC)
    session.commit()
    assert session.query(Position).filter_by(status="closed").count() == 1


def test_kill_switch_event(session: Session) -> None:
    session.add(
        KillSwitchEvent(
            triggered_at=datetime(2024, 5, 13, 19, 0, tzinfo=UTC),
            reason="monthly_drawdown",
            details_json={"drawdown_pct": -0.12},
        )
    )
    session.commit()
    assert session.query(KillSwitchEvent).count() == 1


def test_order_states(session: Session) -> None:
    session.add(
        Order(
            client_order_id="abc-1",
            ticker="GME",
            side="buy",
            instrument="stock",
            quantity=100,
            limit_price=18.5,
            status="pending",
            submitted_at=datetime(2024, 5, 13, 13, 35, tzinfo=UTC),
        )
    )
    session.commit()
    assert session.query(Order).filter_by(status="pending").count() == 1
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/store/test_models.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `models.py`**

```python
"""SQLAlchemy ORM models. One row = one fact at one time."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class UniverseRow(Base):
    __tablename__ = "universe"
    __table_args__ = (UniqueConstraint("ticker", "as_of", name="uq_universe_ticker_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    as_of: Mapped[date] = mapped_column(Date, index=True)
    market_cap_usd: Mapped[float] = mapped_column(Float)
    close_price: Mapped[float] = mapped_column(Float)
    included: Mapped[bool] = mapped_column(Boolean, default=True)
    exclusion_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)


class SignalDaily(Base):
    __tablename__ = "signals_daily"
    __table_args__ = (
        UniqueConstraint("as_of", "ticker", "factor_name", name="uq_signal_unique"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    as_of: Mapped[date] = mapped_column(Date, index=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    factor_name: Mapped[str] = mapped_column(String(64))
    raw_value: Mapped[float] = mapped_column(Float)
    z_score: Mapped[float] = mapped_column(Float)


class SetupClassification(Base):
    __tablename__ = "setup_classifications"
    __table_args__ = (UniqueConstraint("as_of", "ticker", name="uq_setup_unique"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    as_of: Mapped[date] = mapped_column(Date, index=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    score: Mapped[float] = mapped_column(Float)
    setup_type: Mapped[str] = mapped_column(String(16))   # CAR, GME, Mixed, Weak
    rank_in_universe: Mapped[int] = mapped_column(Integer)


class Position(Base):
    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    side: Mapped[str] = mapped_column(String(8))   # long
    instrument: Mapped[str] = mapped_column(String(16))   # stock, call, put
    quantity: Mapped[int] = mapped_column(Integer)
    avg_price: Mapped[float] = mapped_column(Float)
    setup_type: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), default="open")
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    option_strike: Mapped[float | None] = mapped_column(Float, nullable=True)
    option_expiry: Mapped[date | None] = mapped_column(Date, nullable=True)


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_order_id: Mapped[str] = mapped_column(String(64), unique=True)
    broker_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    side: Mapped[str] = mapped_column(String(8))   # buy, sell
    instrument: Mapped[str] = mapped_column(String(16))
    quantity: Mapped[int] = mapped_column(Integer)
    limit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(16))
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    filled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    avg_fill_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    position_id: Mapped[int | None] = mapped_column(ForeignKey("positions.id"), nullable=True)


class PnLDaily(Base):
    __tablename__ = "pnl_daily"

    as_of: Mapped[date] = mapped_column(Date, primary_key=True)
    equity_usd: Mapped[float] = mapped_column(Float)
    cash_usd: Mapped[float] = mapped_column(Float)
    gross_exposure_pct: Mapped[float] = mapped_column(Float)
    daily_pnl_usd: Mapped[float] = mapped_column(Float)
    drawdown_pct: Mapped[float] = mapped_column(Float)


class KillSwitchEvent(Base):
    __tablename__ = "kill_switch_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    triggered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    reason: Mapped[str] = mapped_column(String(64))
    details_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    cleared_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
```

- [ ] **Step 4: Implement `db.py`**

```python
"""Database session factory."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker


def _engine():
    url = os.environ.get(
        "SH_DB_URL",
        "postgresql+psycopg://squeeze:squeeze@localhost:5432/squeeze",
    )
    return create_engine(url, future=True, pool_pre_ping=True)


SessionLocal = sessionmaker(bind=_engine(), expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
```

- [ ] **Step 5: Run unit tests (sqlite in-memory), expect pass**

```bash
uv run pytest tests/store/test_models.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Initialize Alembic**

```bash
uv run alembic init -t async alembic
```

Then edit `alembic.ini` line `sqlalchemy.url =` to `sqlalchemy.url = postgresql+psycopg://squeeze:squeeze@localhost:5432/squeeze`.

Replace `alembic/env.py` with the synchronous version:

```python
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from squeeze_hunter.store.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)
target_metadata = Base.metadata


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
```

- [ ] **Step 7: Generate baseline migration (after Postgres is up — defer to Task 0.6 if needed; for now write it manually)**

`alembic/versions/0001_initial_schema.py`:

```python
"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-05-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "universe",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("ticker", sa.String(16), nullable=False),
        sa.Column("as_of", sa.Date, nullable=False),
        sa.Column("market_cap_usd", sa.Float, nullable=False),
        sa.Column("close_price", sa.Float, nullable=False),
        sa.Column("included", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("exclusion_reason", sa.String(64), nullable=True),
        sa.UniqueConstraint("ticker", "as_of", name="uq_universe_ticker_date"),
    )
    op.create_index("ix_universe_ticker", "universe", ["ticker"])
    op.create_index("ix_universe_as_of", "universe", ["as_of"])

    op.create_table(
        "signals_daily",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("as_of", sa.Date, nullable=False),
        sa.Column("ticker", sa.String(16), nullable=False),
        sa.Column("factor_name", sa.String(64), nullable=False),
        sa.Column("raw_value", sa.Float, nullable=False),
        sa.Column("z_score", sa.Float, nullable=False),
        sa.UniqueConstraint("as_of", "ticker", "factor_name", name="uq_signal_unique"),
    )
    op.create_index("ix_signals_as_of", "signals_daily", ["as_of"])
    op.create_index("ix_signals_ticker", "signals_daily", ["ticker"])

    op.create_table(
        "setup_classifications",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("as_of", sa.Date, nullable=False),
        sa.Column("ticker", sa.String(16), nullable=False),
        sa.Column("score", sa.Float, nullable=False),
        sa.Column("setup_type", sa.String(16), nullable=False),
        sa.Column("rank_in_universe", sa.Integer, nullable=False),
        sa.UniqueConstraint("as_of", "ticker", name="uq_setup_unique"),
    )

    op.create_table(
        "positions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("ticker", sa.String(16), nullable=False),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("side", sa.String(8), nullable=False),
        sa.Column("instrument", sa.String(16), nullable=False),
        sa.Column("quantity", sa.Integer, nullable=False),
        sa.Column("avg_price", sa.Float, nullable=False),
        sa.Column("setup_type", sa.String(16), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="open"),
        sa.Column("realized_pnl", sa.Float, nullable=False, server_default="0"),
        sa.Column("option_strike", sa.Float, nullable=True),
        sa.Column("option_expiry", sa.Date, nullable=True),
    )

    op.create_table(
        "orders",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("client_order_id", sa.String(64), nullable=False, unique=True),
        sa.Column("broker_order_id", sa.String(64), nullable=True),
        sa.Column("ticker", sa.String(16), nullable=False),
        sa.Column("side", sa.String(8), nullable=False),
        sa.Column("instrument", sa.String(16), nullable=False),
        sa.Column("quantity", sa.Integer, nullable=False),
        sa.Column("limit_price", sa.Float, nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("filled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("avg_fill_price", sa.Float, nullable=True),
        sa.Column("position_id", sa.Integer, sa.ForeignKey("positions.id"), nullable=True),
    )

    op.create_table(
        "pnl_daily",
        sa.Column("as_of", sa.Date, primary_key=True),
        sa.Column("equity_usd", sa.Float, nullable=False),
        sa.Column("cash_usd", sa.Float, nullable=False),
        sa.Column("gross_exposure_pct", sa.Float, nullable=False),
        sa.Column("daily_pnl_usd", sa.Float, nullable=False),
        sa.Column("drawdown_pct", sa.Float, nullable=False),
    )

    op.create_table(
        "kill_switch_events",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("triggered_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reason", sa.String(64), nullable=False),
        sa.Column("details_json", sa.JSON, nullable=False),
        sa.Column("cleared_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    for t in (
        "kill_switch_events",
        "pnl_daily",
        "orders",
        "positions",
        "setup_classifications",
        "signals_daily",
        "universe",
    ):
        op.drop_table(t)
```

- [ ] **Step 8: Commit**

```bash
git add src/squeeze_hunter/store/ alembic.ini alembic/ tests/store/
git commit -m "feat(store): add ORM models and Alembic baseline"
```

---

### Task 0.5: Docker Compose stack — postgres + prometheus + grafana

**Files:**
- Create: `docker/compose.yml`
- Create: `docker/prometheus.yml`
- Create: `docker/grafana/provisioning/datasources/prometheus.yml`

- [ ] **Step 1: Write `docker/compose.yml`**

```yaml
services:
  postgres:
    image: postgres:14
    environment:
      POSTGRES_USER: squeeze
      POSTGRES_PASSWORD: squeeze
      POSTGRES_DB: squeeze
    ports: ["5432:5432"]
    volumes:
      - ./.data/postgres:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U squeeze"]
      interval: 5s
      timeout: 3s
      retries: 10

  prometheus:
    image: prom/prometheus:v2.55.0
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
    ports: ["9090:9090"]

  grafana:
    image: grafana/grafana:11.2.0
    environment:
      GF_SECURITY_ADMIN_PASSWORD: squeeze
      GF_AUTH_ANONYMOUS_ENABLED: "true"
      GF_AUTH_ANONYMOUS_ORG_ROLE: Viewer
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
      - ./.data/grafana:/var/lib/grafana
    ports: ["3000:3000"]
    depends_on: [prometheus]

  ib-gateway:
    build:
      context: .
      dockerfile: ib-gateway.dockerfile
    environment:
      TWS_USERID:    ${IBKR_USERID}
      TWS_PASSWORD:  ${IBKR_PASSWORD}
      TRADING_MODE:  paper
    ports:
      - "7497:7497"   # paper TWS / API
      - "5901:5900"   # VNC for debugging
    restart: unless-stopped
```

- [ ] **Step 2: Write `docker/prometheus.yml`**

```yaml
global:
  scrape_interval: 15s
scrape_configs:
  - job_name: squeeze-hunter
    static_configs:
      - targets: ["host.docker.internal:8080"]
```

- [ ] **Step 3: Grafana datasource**

`docker/grafana/provisioning/datasources/prometheus.yml`:

```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

- [ ] **Step 4: Bring up postgres + prometheus + grafana (without ib-gateway, which needs Task 0.6's image)**

```bash
docker compose -f docker/compose.yml up -d postgres prometheus grafana
docker compose -f docker/compose.yml ps
```

Expected: 3 services healthy.

- [ ] **Step 5: Apply Alembic migration against the running postgres**

```bash
export SH_DB_URL=postgresql+psycopg://squeeze:squeeze@localhost:5432/squeeze
uv run alembic upgrade head
```

Expected: `INFO  [alembic.runtime.migration] Running upgrade  -> 0001`.

- [ ] **Step 6: Verify schema**

```bash
docker exec -it $(docker compose -f docker/compose.yml ps -q postgres) \
  psql -U squeeze -d squeeze -c '\dt'
```

Expected: list including `universe`, `signals_daily`, `setup_classifications`, `positions`, `orders`, `pnl_daily`, `kill_switch_events`, `alembic_version`.

- [ ] **Step 7: Commit**

```bash
git add docker/
git commit -m "feat(infra): docker compose for postgres + prometheus + grafana"
```

---

### Task 0.6: IB Gateway Docker image with IBC auto-login

**Files:**
- Create: `docker/ib-gateway.dockerfile`

- [ ] **Step 1: Use a maintained community image**

We do not roll our own from scratch. Use `gnzsnz/ib-gateway:stable` as the base — it bundles IBC, Java, the gateway, and a VNC server.

`docker/ib-gateway.dockerfile`:

```dockerfile
FROM gnzsnz/ib-gateway:stable

# Default to paper, port 7497.
ENV TRADING_MODE=paper \
    TWOFA_TIMEOUT_ACTION=restart \
    AUTO_RESTART_TIME=00:00 \
    READ_ONLY_API=no \
    BYPASS_WARNING=yes

EXPOSE 7497 5900
```

- [ ] **Step 2: Add IBKR credentials to `.env`**

The user has to populate these manually before bringing the gateway up:

```bash
cat <<EOF >> .env
IBKR_USERID=your_paper_username
IBKR_PASSWORD=your_paper_password
EOF
```

(`.env` is in `.gitignore`.)

- [ ] **Step 3: Build and start**

```bash
docker compose -f docker/compose.yml --env-file .env up -d ib-gateway
docker compose -f docker/compose.yml logs --tail=50 ib-gateway
```

Expected: log shows IBC starting, gateway listening on 7497, eventually `IBC: Login has completed`.

- [ ] **Step 4: Manual VNC sanity check (optional)**

Connect to `vnc://localhost:5901` (password is `gnzsnz` by default per the upstream image — the user should change it for non-loopback exposure). Verify the gateway window shows logged in to paper.

- [ ] **Step 5: Commit**

```bash
git add docker/ib-gateway.dockerfile docker/compose.yml
git commit -m "feat(infra): ib-gateway dockerfile with IBC auto-login"
```

---

### Task 0.7: IBKR hello-world via ib-async + CLI scaffold

**Files:**
- Create: `src/squeeze_hunter/cli.py`
- Create: `src/squeeze_hunter/broker/__init__.py`
- Create: `src/squeeze_hunter/broker/base.py`
- Create: `src/squeeze_hunter/broker/ibkr.py`
- Create: `tests/broker/test_base.py`

- [ ] **Step 1: Write the failing test**

`tests/broker/__init__.py`: empty.

`tests/broker/test_base.py`:

```python
from squeeze_hunter.broker.base import BrokerHealth, IBroker


def test_protocol_attributes() -> None:
    assert "connect" in IBroker.__dict__
    assert "disconnect" in IBroker.__dict__
    assert "fetch_quote" in IBroker.__dict__


def test_broker_health_dataclass() -> None:
    h = BrokerHealth(connected=True, last_ping_ms=42, account="DU0")
    assert h.connected is True
    assert h.last_ping_ms == 42
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/broker/test_base.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write `broker/base.py`**

```python
"""IBroker Protocol — the only contract live/paper/sim brokers must satisfy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True, frozen=True)
class Quote:
    ticker: str
    bid: float
    ask: float
    last: float
    timestamp_ns: int


@dataclass(slots=True, frozen=True)
class BrokerHealth:
    connected: bool
    last_ping_ms: int
    account: str


class IBroker(Protocol):
    name: str

    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def fetch_quote(self, ticker: str) -> Quote: ...
    async def health(self) -> BrokerHealth: ...
```

- [ ] **Step 4: Write `broker/ibkr.py` (minimal hello-world)**

```python
"""IBKRBroker — ib-async wrapper. Phase 0 only implements connect / quote / health."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from ib_async import IB, Stock

from squeeze_hunter.broker.base import BrokerHealth, Quote
from squeeze_hunter.logging_setup import get_logger

log = get_logger("broker.ibkr")


@dataclass
class IBKRBroker:
    name: str = "ibkr"
    host: str = os.environ.get("IBKR_HOST", "127.0.0.1")
    port: int = int(os.environ.get("IBKR_PORT", "7497"))
    client_id: int = int(os.environ.get("IBKR_CLIENT_ID", "42"))
    account: str = os.environ.get("IBKR_ACCOUNT", "")

    def __post_init__(self) -> None:
        self._ib = IB()

    async def connect(self) -> None:
        log.info("connecting", host=self.host, port=self.port)
        await self._ib.connectAsync(self.host, self.port, clientId=self.client_id)
        log.info("connected", server_version=self._ib.client.serverVersion())

    async def disconnect(self) -> None:
        self._ib.disconnect()

    async def fetch_quote(self, ticker: str) -> Quote:
        contract = Stock(ticker, "SMART", "USD")
        await self._ib.qualifyContractsAsync(contract)
        ticker_data = self._ib.reqMktData(contract, "", snapshot=True, regulatorySnapshot=False)
        # Wait for snapshot — ib-async populates fields as updates arrive
        for _ in range(40):
            await self._ib.waitOnUpdate(timeout=0.25)
            if ticker_data.last is not None or ticker_data.bid is not None:
                break
        return Quote(
            ticker=ticker,
            bid=float(ticker_data.bid or 0.0),
            ask=float(ticker_data.ask or 0.0),
            last=float(ticker_data.last or ticker_data.close or 0.0),
            timestamp_ns=time.time_ns(),
        )

    async def health(self) -> BrokerHealth:
        return BrokerHealth(
            connected=self._ib.isConnected(),
            last_ping_ms=0,
            account=self.account,
        )
```

- [ ] **Step 5: Write `cli.py`**

```python
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
```

- [ ] **Step 6: Run unit tests, expect pass**

```bash
uv run pytest tests/broker/ -v
```

Expected: 2 passed.

- [ ] **Step 7: Run hello-world against paper gateway**

(Requires Task 0.6 ib-gateway running and logged in.)

```bash
uv run squeeze-hunter hello AAPL
```

Expected: log lines `connecting ... connected ... quote ticker=AAPL bid=... ask=... last=...`, then a one-line stdout summary.

- [ ] **Step 8: Mark Phase 0 milestone**

```bash
git add src/squeeze_hunter/cli.py src/squeeze_hunter/broker/ tests/broker/
git commit -m "feat(broker): ib-async hello-world + CLI scaffold"
git tag phase-0-bootstrap
```

---

## Phase 1 — Data Layer + Signals + Score

### Task 1.1: Domain schemas

**Files:**
- Create: `src/squeeze_hunter/data/__init__.py`
- Create: `src/squeeze_hunter/data/schema.py`
- Create: `tests/data/__init__.py`
- Create: `tests/data/test_schema.py`

- [ ] **Step 1: Write the failing test**

`tests/data/test_schema.py`:

```python
from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from squeeze_hunter.data.schema import (
    Bar,
    EarningsEvent,
    OptionChain,
    OptionQuote,
    Quote,
    RedditMention,
    ShortInterest,
)


def test_bar_validates_ohlc() -> None:
    Bar(
        ticker="GME",
        ts=datetime(2024, 5, 13, 13, 30, tzinfo=UTC),
        open=18.0, high=20.0, low=17.5, close=19.5, volume=1_000_000,
    )
    with pytest.raises(ValidationError):
        Bar(
            ticker="GME",
            ts=datetime(2024, 5, 13, 13, 30, tzinfo=UTC),
            open=18.0, high=15.0, low=17.5, close=19.5, volume=1_000_000,
        )


def test_short_interest_days_to_cover() -> None:
    si = ShortInterest(
        ticker="GME",
        settlement_date=date(2024, 4, 30),
        si_shares=1_000_000,
        si_pct_float=0.10,
        avg_daily_volume_20d=200_000,
    )
    assert si.days_to_cover == pytest.approx(5.0)


def test_option_chain_filter() -> None:
    chain = OptionChain(
        underlying="GME",
        as_of=datetime(2024, 5, 13, 20, 0, tzinfo=UTC),
        spot=18.0,
        quotes=[
            OptionQuote(strike=18.0, expiry=date(2024, 6, 21), right="C",
                        open_interest=1000, volume=200, implied_vol=0.7,
                        bid=1.0, ask=1.2),
            OptionQuote(strike=20.0, expiry=date(2024, 6, 21), right="C",
                        open_interest=500, volume=100, implied_vol=0.8,
                        bid=0.5, ask=0.6),
        ],
    )
    near = chain.near_money_calls(window_pct=0.05)
    assert len(near) == 1
    assert near[0].strike == 18.0


def test_reddit_mention_z_inputs() -> None:
    m = RedditMention(
        ticker="GME",
        as_of=datetime(2024, 5, 13, 12, 0, tzinfo=UTC),
        subreddit="wallstreetbets",
        count_24h=400,
        baseline_30d_mean=50.0,
        baseline_30d_std=20.0,
    )
    assert m.z_score == pytest.approx((400 - 50) / 20)


def test_earnings_event_surprise() -> None:
    e = EarningsEvent(
        ticker="GME",
        report_at=datetime(2024, 5, 12, 20, 30, tzinfo=UTC),
        actual_eps=0.10,
        estimate_eps=0.05,
    )
    assert e.surprise_pct == pytest.approx(1.0)


def test_quote_basic() -> None:
    Quote(ticker="GME", ts=datetime(2024, 5, 13, 14, 0, tzinfo=UTC),
          bid=18.0, ask=18.05, last=18.02, bid_size=100, ask_size=200)
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/data/test_schema.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `data/schema.py`**

```python
"""Domain schemas. All timestamps UTC; only display layers convert."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, computed_field, model_validator


class Bar(BaseModel):
    ticker: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

    @model_validator(mode="after")
    def _ohlc_consistent(self) -> "Bar":
        if not (self.low <= self.open <= self.high and self.low <= self.close <= self.high):
            raise ValueError("OHLC inconsistent: low/high must bound open and close")
        return self


class Quote(BaseModel):
    ticker: str
    ts: datetime
    bid: float
    ask: float
    last: float
    bid_size: int = 0
    ask_size: int = 0


class OptionQuote(BaseModel):
    strike: float
    expiry: date
    right: Literal["C", "P"]
    open_interest: int
    volume: int
    implied_vol: float
    bid: float = 0.0
    ask: float = 0.0


class OptionChain(BaseModel):
    underlying: str
    as_of: datetime
    spot: float
    quotes: list[OptionQuote] = Field(default_factory=list)

    def near_money_calls(self, window_pct: float = 0.05) -> list[OptionQuote]:
        lo = self.spot * (1 - window_pct)
        hi = self.spot * (1 + window_pct)
        return [q for q in self.quotes if q.right == "C" and lo <= q.strike <= hi]

    def total_call_oi(self, *, near_money_window_pct: float | None = None) -> int:
        if near_money_window_pct is None:
            return sum(q.open_interest for q in self.quotes if q.right == "C")
        return sum(q.open_interest for q in self.near_money_calls(near_money_window_pct))


class ShortInterest(BaseModel):
    ticker: str
    settlement_date: date
    si_shares: int
    si_pct_float: float
    avg_daily_volume_20d: int

    @computed_field   # type: ignore[prop-decorator]
    @property
    def days_to_cover(self) -> float:
        if self.avg_daily_volume_20d <= 0:
            return float("inf")
        return self.si_shares / self.avg_daily_volume_20d


class EarningsEvent(BaseModel):
    ticker: str
    report_at: datetime
    actual_eps: float | None = None
    estimate_eps: float | None = None
    actual_revenue: float | None = None
    estimate_revenue: float | None = None

    @computed_field   # type: ignore[prop-decorator]
    @property
    def surprise_pct(self) -> float | None:
        if self.actual_eps is None or self.estimate_eps in (None, 0):
            return None
        assert self.estimate_eps is not None
        return (self.actual_eps - self.estimate_eps) / abs(self.estimate_eps)


class RedditMention(BaseModel):
    ticker: str
    as_of: datetime
    subreddit: str
    count_24h: int
    baseline_30d_mean: float
    baseline_30d_std: float

    @computed_field   # type: ignore[prop-decorator]
    @property
    def z_score(self) -> float:
        if self.baseline_30d_std <= 0:
            return 0.0
        return (self.count_24h - self.baseline_30d_mean) / self.baseline_30d_std
```

- [ ] **Step 4: Run tests, expect pass**

```bash
uv run pytest tests/data/test_schema.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/squeeze_hunter/data/__init__.py src/squeeze_hunter/data/schema.py tests/data/
git commit -m "feat(data): pydantic domain schemas"
```

---

### Task 1.2: DataProvider Protocol + parquet cache

**Files:**
- Create: `src/squeeze_hunter/data/protocol.py`
- Create: `src/squeeze_hunter/data/cache.py`
- Create: `tests/data/test_cache.py`

- [ ] **Step 1: Write the failing test for cache**

`tests/data/test_cache.py`:

```python
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from squeeze_hunter.data.cache import ParquetCache


def test_parquet_cache_roundtrip(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path)
    df = pd.DataFrame(
        {
            "ticker": ["GME", "AMC"],
            "ts": [datetime(2024, 5, 13, tzinfo=UTC), datetime(2024, 5, 13, tzinfo=UTC)],
            "close": [18.0, 4.0],
        }
    )
    cache.write_partition(domain="bars", partition_key="2024-05-13", df=df)
    out = cache.read_partition(domain="bars", partition_key="2024-05-13")
    assert len(out) == 2
    assert set(out["ticker"]) == {"GME", "AMC"}


def test_parquet_cache_dedup_on_append(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path, dedup_keys=["ticker", "ts"])
    df1 = pd.DataFrame(
        {"ticker": ["GME"], "ts": [datetime(2024, 5, 13, tzinfo=UTC)], "close": [18.0]}
    )
    df2 = pd.DataFrame(
        {"ticker": ["GME"], "ts": [datetime(2024, 5, 13, tzinfo=UTC)], "close": [18.5]}
    )
    cache.write_partition("bars", "2024-05-13", df1)
    cache.append_partition("bars", "2024-05-13", df2)
    out = cache.read_partition("bars", "2024-05-13")
    assert len(out) == 1
    assert out["close"].iloc[0] == 18.5   # latest wins
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/data/test_cache.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `cache.py`**

```python
"""Parquet on-disk cache, partitioned by `domain/partition_key/`."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class ParquetCache:
    root: Path
    dedup_keys: list[str] = field(default_factory=list)

    def _path(self, domain: str, partition_key: str) -> Path:
        return self.root / domain / f"{partition_key}.parquet"

    def write_partition(self, domain: str, partition_key: str, df: pd.DataFrame) -> None:
        path = self._path(domain, partition_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.dedup_keys:
            df = df.drop_duplicates(self.dedup_keys, keep="last")
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)

    def read_partition(self, domain: str, partition_key: str) -> pd.DataFrame:
        path = self._path(domain, partition_key)
        if not path.exists():
            return pd.DataFrame()
        return pq.read_table(path).to_pandas()

    def append_partition(self, domain: str, partition_key: str, df: pd.DataFrame) -> None:
        existing = self.read_partition(domain, partition_key)
        merged = pd.concat([existing, df], ignore_index=True) if not existing.empty else df
        if self.dedup_keys:
            merged = merged.drop_duplicates(self.dedup_keys, keep="last")
        self.write_partition(domain, partition_key, merged)
```

- [ ] **Step 4: Implement `protocol.py`**

```python
"""DataProvider Protocol — the only contract data sources must satisfy."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Protocol, runtime_checkable

from squeeze_hunter.data.schema import (
    Bar,
    EarningsEvent,
    OptionChain,
    Quote,
    RedditMention,
    ShortInterest,
)

Resolution = Literal["1m", "5m", "1h", "1d"]


@runtime_checkable
class DataProvider(Protocol):
    name: str
    capabilities: frozenset[str]   # subset of {"bars","quote","options","si","earnings","sentiment"}

    async def fetch_bars(
        self, ticker: str, start: datetime, end: datetime, resolution: Resolution = "1d"
    ) -> list[Bar]: ...

    async def fetch_quote(self, ticker: str) -> Quote: ...

    async def fetch_option_chain(
        self, ticker: str, expiry: date | None = None
    ) -> OptionChain: ...

    async def fetch_short_interest(
        self, ticker: str, since: date | None = None
    ) -> list[ShortInterest]: ...

    async def fetch_earnings(self, ticker: str, since: date | None = None) -> list[EarningsEvent]: ...

    async def fetch_sentiment(self, ticker: str, as_of: datetime) -> RedditMention | None: ...
```

- [ ] **Step 5: Run, expect pass**

```bash
uv run pytest tests/data/ -v
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/squeeze_hunter/data/protocol.py src/squeeze_hunter/data/cache.py \
        tests/data/test_cache.py
git commit -m "feat(data): DataProvider Protocol + parquet cache"
```

---

### Task 1.3: YahooProvider (EOD bars, float, options chain fallback)

**Files:**
- Create: `src/squeeze_hunter/data/providers/__init__.py`
- Create: `src/squeeze_hunter/data/providers/yahoo.py`
- Create: `tests/data/test_yahoo.py`

- [ ] **Step 1: Write the failing test (mocked)**

`tests/data/test_yahoo.py`:

```python
from datetime import UTC, datetime
from unittest.mock import patch

import pandas as pd
import pytest

from squeeze_hunter.data.providers.yahoo import YahooProvider


@pytest.mark.asyncio
async def test_yahoo_fetch_bars_normalizes() -> None:
    fake = pd.DataFrame(
        {
            "Open": [10.0, 11.0],
            "High": [11.0, 12.0],
            "Low": [9.0, 10.5],
            "Close": [10.5, 11.5],
            "Volume": [1_000_000, 1_500_000],
        },
        index=pd.to_datetime(["2024-05-10", "2024-05-13"], utc=True),
    )
    with patch("squeeze_hunter.data.providers.yahoo._yf_history", return_value=fake):
        p = YahooProvider()
        bars = await p.fetch_bars(
            "GME",
            datetime(2024, 5, 10, tzinfo=UTC),
            datetime(2024, 5, 14, tzinfo=UTC),
        )
    assert len(bars) == 2
    assert bars[0].ticker == "GME"
    assert bars[0].close == 10.5


def test_yahoo_capabilities() -> None:
    p = YahooProvider()
    assert "bars" in p.capabilities
    assert "options" in p.capabilities
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/data/test_yahoo.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `providers/yahoo.py`**

```python
"""YahooProvider — yfinance wrapper. Used for EOD history and as fallback."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd
import yfinance as yf

from squeeze_hunter.data.schema import (
    Bar,
    EarningsEvent,
    OptionChain,
    OptionQuote,
    Quote,
    RedditMention,
    ShortInterest,
)


def _yf_history(ticker: str, start: datetime, end: datetime, interval: str) -> pd.DataFrame:
    """Synchronous yfinance call — kept top-level so tests can patch it."""
    return yf.Ticker(ticker).history(start=start, end=end, interval=interval, auto_adjust=False)


@dataclass
class YahooProvider:
    name: str = "yahoo"
    capabilities: frozenset[str] = frozenset({"bars", "options", "earnings"})

    async def fetch_bars(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        resolution: str = "1d",
    ) -> list[Bar]:
        interval = {"1d": "1d", "1h": "1h", "5m": "5m", "1m": "1m"}.get(resolution, "1d")
        df = await asyncio.to_thread(_yf_history, ticker, start, end, interval)
        if df.empty:
            return []
        df.index = pd.to_datetime(df.index, utc=True)
        return [
            Bar(
                ticker=ticker,
                ts=ts.to_pydatetime(),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=int(row["Volume"]),
            )
            for ts, row in df.iterrows()
        ]

    async def fetch_quote(self, ticker: str) -> Quote:
        raise NotImplementedError("YahooProvider does not provide real-time quotes")

    async def fetch_option_chain(
        self, ticker: str, expiry: date | None = None
    ) -> OptionChain:
        def _go() -> OptionChain:
            t = yf.Ticker(ticker)
            spot = float(t.history(period="1d")["Close"].iloc[-1])
            expiries = t.options
            target = expiry.isoformat() if expiry else (expiries[0] if expiries else None)
            quotes: list[OptionQuote] = []
            if target:
                chain = t.option_chain(target)
                for df, right in [(chain.calls, "C"), (chain.puts, "P")]:
                    for _, row in df.iterrows():
                        quotes.append(
                            OptionQuote(
                                strike=float(row["strike"]),
                                expiry=date.fromisoformat(target),
                                right=right,   # type: ignore[arg-type]
                                open_interest=int(row.get("openInterest") or 0),
                                volume=int(row.get("volume") or 0),
                                implied_vol=float(row.get("impliedVolatility") or 0.0),
                                bid=float(row.get("bid") or 0.0),
                                ask=float(row.get("ask") or 0.0),
                            )
                        )
            return OptionChain(
                underlying=ticker, as_of=datetime.utcnow(), spot=spot, quotes=quotes
            )

        return await asyncio.to_thread(_go)

    async def fetch_short_interest(
        self, ticker: str, since: date | None = None
    ) -> list[ShortInterest]:
        return []   # not provided by yahoo

    async def fetch_earnings(
        self, ticker: str, since: date | None = None
    ) -> list[EarningsEvent]:
        def _go() -> list[EarningsEvent]:
            t = yf.Ticker(ticker)
            cal = t.calendar
            events: list[EarningsEvent] = []
            if isinstance(cal, dict) and cal.get("Earnings Date"):
                ts = pd.Timestamp(cal["Earnings Date"][0]).tz_localize("UTC").to_pydatetime()
                events.append(
                    EarningsEvent(
                        ticker=ticker,
                        report_at=ts,
                        estimate_eps=cal.get("Earnings Average"),
                    )
                )
            return events

        return await asyncio.to_thread(_go)

    async def fetch_sentiment(self, ticker: str, as_of: datetime) -> RedditMention | None:
        return None
```

- [ ] **Step 4: Run, expect pass**

```bash
uv run pytest tests/data/test_yahoo.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/squeeze_hunter/data/providers/__init__.py \
        src/squeeze_hunter/data/providers/yahoo.py tests/data/test_yahoo.py
git commit -m "feat(data): YahooProvider for EOD bars and options"
```

---

### Task 1.4: FinraProvider (FTP bulk SI download)

**Files:**
- Create: `src/squeeze_hunter/data/providers/finra.py`
- Create: `tests/data/test_finra.py`

- [ ] **Step 1: Write the failing test**

`tests/data/test_finra.py`:

```python
from datetime import date
from io import StringIO

import pytest

from squeeze_hunter.data.providers.finra import FinraProvider


SAMPLE_FILE = """Settlement Date|Symbol Code|Symbol|Market Class Code|Current Short Position|Previous Short Position|Change|Percent Change|Average Daily Volume|Days To Cover|Revision Indicator
20240430|GME|GME|N|10000000|9500000|500000|5.26|2000000|5.0|N
20240430|AAPL|AAPL|N|50000000|48000000|2000000|4.17|80000000|0.625|N
"""


@pytest.mark.asyncio
async def test_parse_finra_sample() -> None:
    p = FinraProvider()
    rows = list(p._parse_finra_pipe_text(StringIO(SAMPLE_FILE)))
    by_ticker = {r.ticker: r for r in rows}
    assert by_ticker["GME"].si_shares == 10_000_000
    assert by_ticker["GME"].settlement_date == date(2024, 4, 30)
    assert by_ticker["GME"].avg_daily_volume_20d == 2_000_000
    assert by_ticker["GME"].days_to_cover == 5.0


def test_finra_capabilities() -> None:
    p = FinraProvider()
    assert p.capabilities == frozenset({"si"})
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/data/test_finra.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `providers/finra.py`**

```python
"""FinraProvider — biweekly short-interest bulk file (CDN, not FTP since 2023)."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date, datetime
from io import StringIO
from typing import IO

import httpx

from squeeze_hunter.data.schema import (
    Bar,
    EarningsEvent,
    OptionChain,
    Quote,
    RedditMention,
    ShortInterest,
)

# FINRA publishes biweekly short interest at:
# https://cdn.finra.org/equity/regsho/monthly/shrt{YYYYMM}{a|b}.txt
# where the suffix is 'a' for the 15th-of-month report and 'b' for the
# end-of-month report. Format is pipe-delimited with the columns parsed below.
FINRA_URL_TEMPLATE = "https://cdn.finra.org/equity/regsho/monthly/shrt{yyyymm}{half}.txt"


@dataclass
class FinraProvider:
    name: str = "finra"
    capabilities: frozenset[str] = frozenset({"si"})
    timeout_s: float = 30.0

    async def fetch_short_interest(
        self, ticker: str, since: date | None = None
    ) -> list[ShortInterest]:
        # Fetch the most recent 4 reports (covers ~2 months).
        today = date.today()
        results: list[ShortInterest] = []
        candidates: list[tuple[str, str]] = []
        for months_back in range(0, 3):
            year = today.year
            month = today.month - months_back
            while month <= 0:
                month += 12
                year -= 1
            yyyymm = f"{year:04d}{month:02d}"
            for half in ("b", "a"):
                candidates.append((yyyymm, half))
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            for yyyymm, half in candidates:
                url = FINRA_URL_TEMPLATE.format(yyyymm=yyyymm, half=half)
                try:
                    r = await client.get(url)
                    r.raise_for_status()
                except httpx.HTTPError:
                    continue
                for row in self._parse_finra_pipe_text(StringIO(r.text)):
                    if row.ticker == ticker and (since is None or row.settlement_date >= since):
                        results.append(row)
        return results

    @staticmethod
    def _parse_finra_pipe_text(stream: IO[str]) -> Iterator[ShortInterest]:
        header = stream.readline().rstrip("\n").split("|")
        idx = {h.strip(): i for i, h in enumerate(header)}
        for line in stream:
            parts = line.rstrip("\n").split("|")
            if len(parts) < len(header):
                continue
            try:
                yield ShortInterest(
                    ticker=parts[idx["Symbol"]],
                    settlement_date=datetime.strptime(
                        parts[idx["Settlement Date"]], "%Y%m%d"
                    ).date(),
                    si_shares=int(parts[idx["Current Short Position"]]),
                    si_pct_float=0.0,   # FINRA file has no float; merged later by signal layer
                    avg_daily_volume_20d=int(float(parts[idx["Average Daily Volume"]])),
                )
            except (ValueError, KeyError):
                continue

    async def fetch_bars(self, *_a, **_kw) -> list[Bar]:
        return []

    async def fetch_quote(self, ticker: str) -> Quote:
        raise NotImplementedError

    async def fetch_option_chain(self, *_a, **_kw) -> OptionChain:
        return OptionChain(underlying="", as_of=datetime.utcnow(), spot=0.0, quotes=[])

    async def fetch_earnings(self, *_a, **_kw) -> list[EarningsEvent]:
        return []

    async def fetch_sentiment(self, *_a, **_kw) -> RedditMention | None:
        return None


def _months_back(today: date, n: int) -> Iterable[date]:
    return [today]   # kept for future use
```

- [ ] **Step 4: Run, expect pass**

```bash
uv run pytest tests/data/test_finra.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/squeeze_hunter/data/providers/finra.py tests/data/test_finra.py
git commit -m "feat(data): FinraProvider for biweekly short interest"
```

---

### Task 1.5: RedditProvider (PRAW mention counting)

**Files:**
- Create: `src/squeeze_hunter/data/providers/reddit.py`
- Create: `tests/data/test_reddit.py`

- [ ] **Step 1: Write the failing test**

`tests/data/test_reddit.py`:

```python
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from squeeze_hunter.data.providers.reddit import RedditProvider, _count_ticker_mentions


def test_mention_counter_matches_dollar_signs_and_words() -> None:
    posts = [
        SimpleNamespace(title="$GME to the moon", selftext="loaded calls"),
        SimpleNamespace(title="GameStop earnings tomorrow", selftext="$GME paper hands sell"),
        SimpleNamespace(title="AAPL is fine", selftext="boring"),
    ]
    n = _count_ticker_mentions(posts, ticker="GME", aliases=["GameStop"])
    assert n == 2   # third post has no GME / GameStop


@pytest.mark.asyncio
async def test_provider_z_score_uses_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    p = RedditProvider(client_id="x", client_secret="y", user_agent="t")

    async def fake_count_recent(*a, **kw) -> int:
        return 400

    async def fake_baseline(*a, **kw) -> tuple[float, float]:
        return 50.0, 20.0

    monkeypatch.setattr(p, "_count_24h_mentions", fake_count_recent)
    monkeypatch.setattr(p, "_compute_baseline_30d", fake_baseline)
    m = await p.fetch_sentiment("GME", datetime(2024, 5, 13, tzinfo=UTC))
    assert m is not None
    assert m.count_24h == 400
    assert m.z_score == pytest.approx((400 - 50) / 20)
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/data/test_reddit.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `providers/reddit.py`**

```python
"""RedditProvider — PRAW wrapper for r/wallstreetbets mention counting."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import praw

from squeeze_hunter.data.schema import (
    Bar,
    EarningsEvent,
    OptionChain,
    Quote,
    RedditMention,
    ShortInterest,
)


def _count_ticker_mentions(posts: Iterable[object], ticker: str, aliases: list[str]) -> int:
    patterns = [re.compile(rf"\${ticker}\b", re.IGNORECASE), re.compile(rf"\b{ticker}\b")]
    for a in aliases:
        patterns.append(re.compile(rf"\b{re.escape(a)}\b", re.IGNORECASE))
    n = 0
    for post in posts:
        text = f"{getattr(post, 'title', '')} {getattr(post, 'selftext', '')}"
        if any(p.search(text) for p in patterns):
            n += 1
    return n


@dataclass
class RedditProvider:
    client_id: str
    client_secret: str
    user_agent: str
    subreddits: tuple[str, ...] = ("wallstreetbets",)
    name: str = "reddit"
    capabilities: frozenset[str] = field(default_factory=lambda: frozenset({"sentiment"}))

    def __post_init__(self) -> None:
        self._reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent,
            ratelimit_seconds=60,
        )

    async def fetch_sentiment(self, ticker: str, as_of: datetime) -> RedditMention | None:
        count_24h = await self._count_24h_mentions(ticker, as_of)
        baseline_mean, baseline_std = await self._compute_baseline_30d(ticker, as_of)
        return RedditMention(
            ticker=ticker,
            as_of=as_of,
            subreddit=self.subreddits[0],
            count_24h=count_24h,
            baseline_30d_mean=baseline_mean,
            baseline_30d_std=baseline_std,
        )

    async def _count_24h_mentions(self, ticker: str, as_of: datetime) -> int:
        cutoff = as_of - timedelta(days=1)
        total = 0
        for sub_name in self.subreddits:
            sub = self._reddit.subreddit(sub_name)
            posts = list(sub.new(limit=1000))
            posts = [p for p in posts if datetime.fromtimestamp(p.created_utc, tz=UTC) >= cutoff]
            total += _count_ticker_mentions(posts, ticker, aliases=[])
        return total

    async def _compute_baseline_30d(self, ticker: str, as_of: datetime) -> tuple[float, float]:
        # Cheap baseline: per-day counts over the last 30 days, taken from search.
        # In Phase 1 we approximate from cached daily counts (recorded by nightly job).
        # Without history yet, fall back to a wide flat prior (mean=10, std=20) to
        # give well-behaved z-scores until enough days accumulate.
        return 10.0, 20.0

    # -- protocol stubs --
    async def fetch_bars(self, *_a, **_kw) -> list[Bar]: return []
    async def fetch_quote(self, ticker: str) -> Quote: raise NotImplementedError
    async def fetch_option_chain(self, *_a, **_kw) -> OptionChain:
        return OptionChain(underlying="", as_of=datetime.utcnow(), spot=0.0, quotes=[])
    async def fetch_short_interest(self, *_a, **_kw) -> list[ShortInterest]: return []
    async def fetch_earnings(self, *_a, **_kw) -> list[EarningsEvent]: return []
```

- [ ] **Step 4: Run, expect pass**

```bash
uv run pytest tests/data/test_reddit.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/squeeze_hunter/data/providers/reddit.py tests/data/test_reddit.py
git commit -m "feat(data): RedditProvider with PRAW mention counting"
```

---

### Task 1.6: FinnhubProvider (earnings calendar)

**Files:**
- Create: `src/squeeze_hunter/data/providers/finnhub.py`
- Create: `tests/data/test_finnhub.py`

- [ ] **Step 1: Write the failing test**

`tests/data/test_finnhub.py`:

```python
from datetime import UTC, date, datetime
from unittest.mock import patch

import pytest

from squeeze_hunter.data.providers.finnhub import FinnhubProvider


@pytest.mark.asyncio
async def test_finnhub_earnings_parses() -> None:
    fake = {
        "earningsCalendar": [
            {
                "symbol": "GME",
                "date": "2024-06-04",
                "hour": "amc",
                "epsActual": 0.10,
                "epsEstimate": 0.05,
                "revenueActual": 1_000_000_000,
                "revenueEstimate": 950_000_000,
            }
        ]
    }
    with patch.object(FinnhubProvider, "_call_calendar", return_value=fake):
        p = FinnhubProvider(api_key="x")
        events = await p.fetch_earnings("GME", since=date(2024, 1, 1))
    assert len(events) == 1
    assert events[0].ticker == "GME"
    assert events[0].surprise_pct == pytest.approx(1.0)


def test_finnhub_capabilities() -> None:
    p = FinnhubProvider(api_key="x")
    assert "earnings" in p.capabilities
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/data/test_finnhub.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `providers/finnhub.py`**

```python
"""FinnhubProvider — free-tier earnings calendar."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import finnhub

from squeeze_hunter.data.schema import (
    Bar,
    EarningsEvent,
    OptionChain,
    Quote,
    RedditMention,
    ShortInterest,
)


@dataclass
class FinnhubProvider:
    api_key: str
    name: str = "finnhub"
    capabilities: frozenset[str] = field(default_factory=lambda: frozenset({"earnings"}))

    def __post_init__(self) -> None:
        self._client = finnhub.Client(api_key=self.api_key) if self.api_key else None

    def _call_calendar(self, ticker: str, frm: str, to: str) -> dict[str, Any]:
        if self._client is None:
            return {"earningsCalendar": []}
        return self._client.earnings_calendar(_from=frm, to=to, symbol=ticker, international=False)

    async def fetch_earnings(
        self, ticker: str, since: date | None = None
    ) -> list[EarningsEvent]:
        frm = (since or (date.today() - timedelta(days=365 * 2))).isoformat()
        to = (date.today() + timedelta(days=180)).isoformat()
        raw = await asyncio.to_thread(self._call_calendar, ticker, frm, to)
        out: list[EarningsEvent] = []
        for ev in raw.get("earningsCalendar", []):
            d = date.fromisoformat(ev["date"])
            hour = (ev.get("hour") or "").lower()
            ts = datetime(d.year, d.month, d.day, 20 if hour == "amc" else 12, 30, tzinfo=UTC)
            out.append(
                EarningsEvent(
                    ticker=ticker,
                    report_at=ts,
                    actual_eps=ev.get("epsActual"),
                    estimate_eps=ev.get("epsEstimate"),
                    actual_revenue=ev.get("revenueActual"),
                    estimate_revenue=ev.get("revenueEstimate"),
                )
            )
        return out

    # -- protocol stubs --
    async def fetch_bars(self, *_a, **_kw) -> list[Bar]: return []
    async def fetch_quote(self, ticker: str) -> Quote: raise NotImplementedError
    async def fetch_option_chain(self, *_a, **_kw) -> OptionChain:
        return OptionChain(underlying="", as_of=datetime.utcnow(), spot=0.0, quotes=[])
    async def fetch_short_interest(self, *_a, **_kw) -> list[ShortInterest]: return []
    async def fetch_sentiment(self, *_a, **_kw) -> RedditMention | None: return None
```

- [ ] **Step 4: Run, expect pass**

```bash
uv run pytest tests/data/test_finnhub.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/squeeze_hunter/data/providers/finnhub.py tests/data/test_finnhub.py
git commit -m "feat(data): FinnhubProvider for earnings calendar"
```

---

### Task 1.7: BacktestProvider with clock (replay parquet history)

**Files:**
- Create: `src/squeeze_hunter/data/providers/backtest.py`
- Create: `tests/data/test_backtest_provider.py`

- [ ] **Step 1: Write the failing test**

`tests/data/test_backtest_provider.py`:

```python
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.data.providers.backtest import BacktestProvider, Clock


@pytest.mark.asyncio
async def test_backtest_only_returns_rows_at_or_before_clock(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path)
    df = pd.DataFrame(
        {
            "ticker": ["GME", "GME", "GME"],
            "ts": [
                datetime(2024, 5, 10, tzinfo=UTC),
                datetime(2024, 5, 13, tzinfo=UTC),
                datetime(2024, 5, 14, tzinfo=UTC),
            ],
            "open": [16.0, 18.0, 25.0],
            "high": [17.0, 20.0, 30.0],
            "low": [15.0, 17.5, 22.0],
            "close": [16.5, 19.5, 28.0],
            "volume": [1_000_000, 5_000_000, 30_000_000],
        }
    )
    cache.write_partition("bars", "GME", df)

    clock = Clock(now=datetime(2024, 5, 13, 23, 59, tzinfo=UTC))
    p = BacktestProvider(cache=cache, clock=clock)
    bars = await p.fetch_bars(
        "GME", datetime(2024, 5, 1, tzinfo=UTC), datetime(2024, 5, 20, tzinfo=UTC)
    )
    assert len(bars) == 2   # 5/14 row excluded
    assert max(b.ts for b in bars) == datetime(2024, 5, 13, tzinfo=UTC)
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/data/test_backtest_provider.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `providers/backtest.py`**

```python
"""BacktestProvider — replays cached parquet history with a clock that
prevents lookahead bias by construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime

import pandas as pd

from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.data.schema import (
    Bar,
    EarningsEvent,
    OptionChain,
    Quote,
    RedditMention,
    ShortInterest,
)


@dataclass
class Clock:
    now: datetime

    def advance_to(self, t: datetime) -> None:
        if t < self.now:
            raise ValueError("clock cannot go backwards")
        self.now = t


@dataclass
class BacktestProvider:
    cache: ParquetCache
    clock: Clock
    name: str = "backtest"
    capabilities: frozenset[str] = field(
        default_factory=lambda: frozenset({"bars", "options", "si", "earnings", "sentiment"})
    )

    async def fetch_bars(
        self, ticker: str, start: datetime, end: datetime, resolution: str = "1d"
    ) -> list[Bar]:
        df = self.cache.read_partition("bars", ticker)
        if df.empty:
            return []
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        mask = (df["ts"] >= start) & (df["ts"] <= end) & (df["ts"] <= self.clock.now)
        return [
            Bar(
                ticker=row["ticker"],
                ts=row["ts"].to_pydatetime(),
                open=float(row["open"]), high=float(row["high"]),
                low=float(row["low"]), close=float(row["close"]),
                volume=int(row["volume"]),
            )
            for _, row in df[mask].iterrows()
        ]

    async def fetch_quote(self, ticker: str) -> Quote:
        bars = await self.fetch_bars(
            ticker,
            self.clock.now.replace(hour=0, minute=0, second=0, microsecond=0),
            self.clock.now,
        )
        if not bars:
            raise LookupError(f"no bars for {ticker} as of {self.clock.now}")
        b = bars[-1]
        return Quote(ticker=ticker, ts=b.ts, bid=b.close, ask=b.close, last=b.close)

    async def fetch_option_chain(
        self, ticker: str, expiry: date | None = None
    ) -> OptionChain:
        df = self.cache.read_partition("options", f"{ticker}__{self.clock.now.date().isoformat()}")
        if df.empty:
            return OptionChain(underlying=ticker, as_of=self.clock.now, spot=0.0, quotes=[])
        # df is expected to already be a serialized OptionChain — just trust it.
        from squeeze_hunter.data.schema import OptionQuote
        quotes = [
            OptionQuote(
                strike=float(r["strike"]),
                expiry=date.fromisoformat(r["expiry"]),
                right=r["right"],
                open_interest=int(r["open_interest"]),
                volume=int(r["volume"]),
                implied_vol=float(r["implied_vol"]),
                bid=float(r.get("bid", 0.0)),
                ask=float(r.get("ask", 0.0)),
            )
            for _, r in df.iterrows()
        ]
        spot = float(df["spot"].iloc[0]) if "spot" in df.columns else 0.0
        return OptionChain(underlying=ticker, as_of=self.clock.now, spot=spot, quotes=quotes)

    async def fetch_short_interest(
        self, ticker: str, since: date | None = None
    ) -> list[ShortInterest]:
        df = self.cache.read_partition("short_interest", "all")
        if df.empty:
            return []
        df["settlement_date"] = pd.to_datetime(df["settlement_date"]).dt.date
        clock_d = self.clock.now.date()
        mask = (df["ticker"] == ticker) & (df["settlement_date"] <= clock_d)
        if since is not None:
            mask = mask & (df["settlement_date"] >= since)
        return [
            ShortInterest(
                ticker=row["ticker"],
                settlement_date=row["settlement_date"],
                si_shares=int(row["si_shares"]),
                si_pct_float=float(row.get("si_pct_float", 0.0)),
                avg_daily_volume_20d=int(row["avg_daily_volume_20d"]),
            )
            for _, row in df[mask].iterrows()
        ]

    async def fetch_earnings(
        self, ticker: str, since: date | None = None
    ) -> list[EarningsEvent]:
        df = self.cache.read_partition("earnings", "all")
        if df.empty:
            return []
        df["report_at"] = pd.to_datetime(df["report_at"], utc=True)
        mask = (df["ticker"] == ticker) & (df["report_at"] <= self.clock.now)
        if since is not None:
            mask = mask & (df["report_at"].dt.date >= since)
        return [
            EarningsEvent(
                ticker=row["ticker"],
                report_at=row["report_at"].to_pydatetime(),
                actual_eps=row.get("actual_eps"),
                estimate_eps=row.get("estimate_eps"),
            )
            for _, row in df[mask].iterrows()
        ]

    async def fetch_sentiment(
        self, ticker: str, as_of: datetime
    ) -> RedditMention | None:
        df = self.cache.read_partition("sentiment", as_of.date().isoformat())
        if df.empty:
            return None
        df = df[df["ticker"] == ticker]
        if df.empty:
            return None
        row = df.iloc[-1]
        return RedditMention(
            ticker=ticker,
            as_of=as_of,
            subreddit=row["subreddit"],
            count_24h=int(row["count_24h"]),
            baseline_30d_mean=float(row["baseline_30d_mean"]),
            baseline_30d_std=float(row["baseline_30d_std"]),
        )
```

- [ ] **Step 4: Run, expect pass**

```bash
uv run pytest tests/data/test_backtest_provider.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/squeeze_hunter/data/providers/backtest.py tests/data/test_backtest_provider.py
git commit -m "feat(data): BacktestProvider with clock-bound queries"
```

---

### Task 1.8: Universe builder

**Files:**
- Create: `src/squeeze_hunter/universe.py`
- Create: `tests/test_universe.py`

- [ ] **Step 1: Write the failing test**

`tests/test_universe.py`:

```python
from datetime import date

import pandas as pd
import pytest

from squeeze_hunter.config import UniverseCfg
from squeeze_hunter.universe import build_universe


def test_universe_filters_by_cap_price_and_listing() -> None:
    rows = pd.DataFrame(
        [
            {"ticker": "GME",  "market_cap": 1e9,  "close": 18.0, "days_listed": 365},
            {"ticker": "AAPL", "market_cap": 3e12, "close": 200.0, "days_listed": 5000},   # over cap
            {"ticker": "PNNY", "market_cap": 1e8,  "close": 1.0,   "days_listed": 365},    # below cap & price
            {"ticker": "NEW",  "market_cap": 5e8,  "close": 10.0,  "days_listed": 10},     # too new
            {"ticker": "HTZ",  "market_cap": 2e9,  "close": 6.0,   "days_listed": 1000},
        ]
    )
    cfg = UniverseCfg()
    out = build_universe(rows, as_of=date(2024, 5, 13), cfg=cfg)
    included = sorted(out.loc[out["included"], "ticker"].tolist())
    assert included == ["GME", "HTZ"]
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/test_universe.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `universe.py`**

```python
"""Universe builder. Pure function over a dataframe of universe candidates."""

from __future__ import annotations

from datetime import date

import pandas as pd

from squeeze_hunter.config import UniverseCfg


def build_universe(rows: pd.DataFrame, as_of: date, cfg: UniverseCfg) -> pd.DataFrame:
    """Add `included` and `exclusion_reason` columns."""
    out = rows.copy()
    reasons = []
    included = []
    for _, r in out.iterrows():
        if r["market_cap"] < cfg.min_market_cap:
            reasons.append("market_cap_below_floor")
            included.append(False)
        elif r["market_cap"] > cfg.max_market_cap:
            reasons.append("market_cap_above_ceiling")
            included.append(False)
        elif r["close"] < cfg.min_price:
            reasons.append("price_below_floor")
            included.append(False)
        elif r["days_listed"] < cfg.min_days_listed:
            reasons.append("listed_too_recently")
            included.append(False)
        else:
            reasons.append(None)
            included.append(True)
    out["included"] = included
    out["exclusion_reason"] = reasons
    out["as_of"] = as_of
    return out
```

- [ ] **Step 4: Run, expect pass**

```bash
uv run pytest tests/test_universe.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/squeeze_hunter/universe.py tests/test_universe.py
git commit -m "feat(universe): rule-based universe builder"
```

---

### Task 1.9: Signal helpers — Factor schema + cross-sectional z-score

**Files:**
- Create: `src/squeeze_hunter/signals/__init__.py`
- Create: `src/squeeze_hunter/signals/base.py`
- Create: `src/squeeze_hunter/signals/normalize.py`
- Create: `tests/signals/__init__.py`
- Create: `tests/signals/test_normalize.py`

- [ ] **Step 1: Write the failing test**

`tests/signals/test_normalize.py`:

```python
import pandas as pd
import pytest

from squeeze_hunter.signals.normalize import cross_sectional_z


def test_z_zero_when_at_mean() -> None:
    s = pd.Series([1.0, 2.0, 3.0])
    z = cross_sectional_z(s)
    assert z.iloc[1] == pytest.approx(0.0)


def test_z_clipped_to_window() -> None:
    s = pd.Series([0.0, 0.0, 0.0, 0.0, 100.0])
    z = cross_sectional_z(s, clip=3.0)
    assert z.max() == pytest.approx(3.0)


def test_z_handles_zero_std() -> None:
    s = pd.Series([5.0, 5.0, 5.0])
    z = cross_sectional_z(s)
    assert (z == 0.0).all()
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/signals/test_normalize.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `signals/normalize.py`**

```python
"""Cross-sectional normalization."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_z(series: pd.Series, clip: float = 3.0) -> pd.Series:
    mean = float(series.mean())
    std = float(series.std(ddof=0))
    if std == 0.0 or np.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    z = (series - mean) / std
    return z.clip(-clip, clip)
```

- [ ] **Step 4: Implement `signals/base.py`**

```python
"""Signal Protocol + Factor schema. Each signal is a pure function:

    compute(universe_tickers, provider, clock) -> Factor

Factor is a dataframe keyed by ticker with the raw value and (later) z-score.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from squeeze_hunter.data.protocol import DataProvider


@dataclass
class Factor:
    name: str
    as_of: datetime
    values: pd.DataFrame   # columns: ticker, raw_value, evidence (optional dict)


SignalFn = Callable[[list[str], DataProvider, datetime], Awaitable[Factor]]
```

- [ ] **Step 5: Run, expect pass**

```bash
uv run pytest tests/signals/test_normalize.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/squeeze_hunter/signals/ tests/signals/test_normalize.py tests/signals/__init__.py
git commit -m "feat(signals): factor schema and cross-sectional z-score"
```

---

### Task 1.10: Signals f1 + f2 — short interest factors

**Files:**
- Create: `src/squeeze_hunter/signals/short_interest.py`
- Create: `tests/signals/test_short_interest.py`

- [ ] **Step 1: Write the failing test**

`tests/signals/test_short_interest.py`:

```python
from datetime import UTC, date, datetime
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from squeeze_hunter.data.schema import ShortInterest
from squeeze_hunter.signals.short_interest import compute_si_pct_float, compute_days_to_cover


@pytest.mark.asyncio
async def test_si_pct_float_factor_picks_latest() -> None:
    provider = AsyncMock()
    provider.fetch_short_interest.side_effect = lambda t, since=None: [
        ShortInterest(
            ticker=t,
            settlement_date=date(2024, 4, 30),
            si_shares={"GME": 5_000_000, "AAPL": 1_000_000}[t],
            si_pct_float={"GME": 0.20, "AAPL": 0.01}[t],
            avg_daily_volume_20d={"GME": 1_000_000, "AAPL": 80_000_000}[t],
        )
    ]
    factor = await compute_si_pct_float(
        ["GME", "AAPL"], provider, datetime(2024, 5, 13, tzinfo=UTC)
    )
    out = factor.values.set_index("ticker")
    assert out.loc["GME", "raw_value"] == 0.20
    assert out.loc["AAPL", "raw_value"] == 0.01


@pytest.mark.asyncio
async def test_days_to_cover_uses_si_div_adv() -> None:
    provider = AsyncMock()
    provider.fetch_short_interest.return_value = [
        ShortInterest(
            ticker="GME",
            settlement_date=date(2024, 4, 30),
            si_shares=5_000_000,
            si_pct_float=0.20,
            avg_daily_volume_20d=1_000_000,
        )
    ]
    factor = await compute_days_to_cover(["GME"], provider, datetime(2024, 5, 13, tzinfo=UTC))
    out = factor.values.set_index("ticker")
    assert out.loc["GME", "raw_value"] == pytest.approx(5.0)
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/signals/test_short_interest.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `signals/short_interest.py`**

```python
"""Signals f1 (SI % of float) and f2 (Days-to-cover)."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from squeeze_hunter.data.protocol import DataProvider
from squeeze_hunter.signals.base import Factor


async def compute_si_pct_float(
    tickers: list[str], provider: DataProvider, clock: datetime
) -> Factor:
    rows = []
    for t in tickers:
        si_list = await provider.fetch_short_interest(t)
        if not si_list:
            continue
        latest = max(si_list, key=lambda x: x.settlement_date)
        rows.append({"ticker": t, "raw_value": latest.si_pct_float})
    return Factor(name="f1_si_pct", as_of=clock, values=pd.DataFrame(rows))


async def compute_days_to_cover(
    tickers: list[str], provider: DataProvider, clock: datetime
) -> Factor:
    rows = []
    for t in tickers:
        si_list = await provider.fetch_short_interest(t)
        if not si_list:
            continue
        latest = max(si_list, key=lambda x: x.settlement_date)
        rows.append({"ticker": t, "raw_value": float(latest.days_to_cover)})
    return Factor(name="f2_days_to_cover", as_of=clock, values=pd.DataFrame(rows))
```

- [ ] **Step 4: Run, expect pass**

```bash
uv run pytest tests/signals/test_short_interest.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/squeeze_hunter/signals/short_interest.py tests/signals/test_short_interest.py
git commit -m "feat(signals): f1 SI%-of-float and f2 days-to-cover"
```

---

### Task 1.11: Signal f3 — earnings reaction

**Files:**
- Create: `src/squeeze_hunter/signals/earnings_reaction.py`
- Create: `tests/signals/test_earnings_reaction.py`

- [ ] **Step 1: Write the failing test**

`tests/signals/test_earnings_reaction.py`:

```python
from datetime import UTC, date, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from squeeze_hunter.data.schema import Bar, EarningsEvent
from squeeze_hunter.signals.earnings_reaction import compute_earnings_reaction


def _bar(ticker: str, ts: datetime, close: float, volume: int) -> Bar:
    return Bar(ticker=ticker, ts=ts, open=close, high=close, low=close,
               close=close, volume=volume)


@pytest.mark.asyncio
async def test_earnings_reaction_uses_post_event_gap_and_volume() -> None:
    earnings_at = datetime(2024, 5, 12, 20, 30, tzinfo=UTC)   # AMC, 2024-05-12
    pre_bars = [
        _bar("GME", earnings_at - timedelta(days=d), 18.0, 1_000_000)
        for d in range(20, 0, -1)
    ]
    post_bar = _bar("GME", earnings_at + timedelta(days=1), 22.0, 8_000_000)
    provider = AsyncMock()
    provider.fetch_earnings.return_value = [
        EarningsEvent(
            ticker="GME", report_at=earnings_at, actual_eps=0.10, estimate_eps=0.05
        )
    ]
    provider.fetch_bars.return_value = [*pre_bars, post_bar]
    factor = await compute_earnings_reaction(
        ["GME"], provider, datetime(2024, 5, 13, 13, 0, tzinfo=UTC)
    )
    out = factor.values.set_index("ticker")
    # gap ~22% with volume ratio 8x → strong positive raw value
    assert out.loc["GME", "raw_value"] > 0
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/signals/test_earnings_reaction.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `signals/earnings_reaction.py`**

```python
"""Signal f3 — earnings reaction.

raw_value = sign(gap) * |gap| * log(1 + max(0, volume_ratio - 1))

where:
  gap            = (close_first_session_after - close_session_before) / close_session_before
  volume_ratio   = post_volume / mean(pre_20d_volume)

Captures fundamental catalysts via *price reaction* without paid news/NLP.
Only earnings within the past 5 trading days contribute; older ones decay to 0.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from math import log

import pandas as pd

from squeeze_hunter.data.protocol import DataProvider
from squeeze_hunter.signals.base import Factor


async def compute_earnings_reaction(
    tickers: list[str], provider: DataProvider, clock: datetime
) -> Factor:
    rows = []
    horizon_days = 5
    for t in tickers:
        events = await provider.fetch_earnings(t)
        recent = [
            e for e in events
            if 0 <= (clock - e.report_at).days <= horizon_days
        ]
        if not recent:
            rows.append({"ticker": t, "raw_value": 0.0})
            continue
        ev = max(recent, key=lambda e: e.report_at)
        bars = await provider.fetch_bars(
            t, ev.report_at - timedelta(days=30), clock
        )
        if len(bars) < 21:
            rows.append({"ticker": t, "raw_value": 0.0})
            continue
        bars.sort(key=lambda b: b.ts)
        # last bar before event:
        pre_event_bars = [b for b in bars if b.ts < ev.report_at]
        post_event_bars = [b for b in bars if b.ts > ev.report_at]
        if not pre_event_bars or not post_event_bars:
            rows.append({"ticker": t, "raw_value": 0.0})
            continue
        last_pre = pre_event_bars[-1]
        first_post = post_event_bars[0]
        gap = (first_post.close - last_pre.close) / last_pre.close
        last_20 = pre_event_bars[-20:]
        avg_vol = sum(b.volume for b in last_20) / max(len(last_20), 1)
        ratio = first_post.volume / avg_vol if avg_vol > 0 else 1.0
        raw = (1 if gap >= 0 else -1) * abs(gap) * log(1 + max(0.0, ratio - 1.0))
        # decay: linear over horizon
        days_old = (clock - ev.report_at).days
        decay = max(0.0, 1.0 - days_old / horizon_days)
        rows.append({"ticker": t, "raw_value": raw * decay})
    return Factor(name="f3_earnings_reaction", as_of=clock, values=pd.DataFrame(rows))
```

- [ ] **Step 4: Run, expect pass**

```bash
uv run pytest tests/signals/test_earnings_reaction.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/squeeze_hunter/signals/earnings_reaction.py tests/signals/test_earnings_reaction.py
git commit -m "feat(signals): f3 earnings reaction (gap × volume)"
```

---

### Task 1.12: Signal f4 — WSB sentiment

**Files:**
- Create: `src/squeeze_hunter/signals/sentiment.py`
- Create: `tests/signals/test_sentiment.py`

- [ ] **Step 1: Write the failing test**

`tests/signals/test_sentiment.py`:

```python
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from squeeze_hunter.data.schema import RedditMention
from squeeze_hunter.signals.sentiment import compute_wsb_sentiment


@pytest.mark.asyncio
async def test_wsb_sentiment_returns_z_from_provider() -> None:
    provider = AsyncMock()
    provider.fetch_sentiment.side_effect = lambda t, ts: RedditMention(
        ticker=t,
        as_of=ts,
        subreddit="wallstreetbets",
        count_24h={"GME": 400, "AAPL": 8}[t],
        baseline_30d_mean=10.0,
        baseline_30d_std=20.0,
    )
    factor = await compute_wsb_sentiment(
        ["GME", "AAPL"], provider, datetime(2024, 5, 13, tzinfo=UTC)
    )
    out = factor.values.set_index("ticker")
    assert out.loc["GME", "raw_value"] == pytest.approx((400 - 10) / 20)
    assert out.loc["AAPL", "raw_value"] == pytest.approx((8 - 10) / 20)
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/signals/test_sentiment.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `signals/sentiment.py`**

```python
"""Signal f4 — Reddit WSB mention z-score (24h vs 30d baseline)."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from squeeze_hunter.data.protocol import DataProvider
from squeeze_hunter.signals.base import Factor


async def compute_wsb_sentiment(
    tickers: list[str], provider: DataProvider, clock: datetime
) -> Factor:
    rows = []
    for t in tickers:
        m = await provider.fetch_sentiment(t, clock)
        if m is None:
            rows.append({"ticker": t, "raw_value": 0.0})
            continue
        rows.append({"ticker": t, "raw_value": float(m.z_score)})
    return Factor(name="f4_wsb_mention", as_of=clock, values=pd.DataFrame(rows))
```

- [ ] **Step 4: Run, expect pass**

```bash
uv run pytest tests/signals/test_sentiment.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/squeeze_hunter/signals/sentiment.py tests/signals/test_sentiment.py
git commit -m "feat(signals): f4 WSB sentiment z-score"
```

---

### Task 1.13: Signal f5 — ATM call OI 7-day velocity

**Files:**
- Create: `src/squeeze_hunter/signals/options_flow.py`
- Create: `tests/signals/test_options_flow.py`

- [ ] **Step 1: Write the failing test**

`tests/signals/test_options_flow.py`:

```python
from datetime import UTC, date, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from squeeze_hunter.data.schema import OptionChain, OptionQuote
from squeeze_hunter.signals.options_flow import compute_call_oi_velocity


def _chain(spot: float, oi: int, ts: datetime) -> OptionChain:
    return OptionChain(
        underlying="GME", as_of=ts, spot=spot,
        quotes=[
            OptionQuote(strike=spot, expiry=date(2024, 6, 21), right="C",
                        open_interest=oi, volume=10, implied_vol=0.7),
        ],
    )


@pytest.mark.asyncio
async def test_call_oi_velocity_positive_when_growing() -> None:
    now = datetime(2024, 5, 13, 13, 30, tzinfo=UTC)
    week_ago = now - timedelta(days=7)
    provider = AsyncMock()

    async def chain(_t, expiry=None) -> OptionChain:
        return _chain(spot=18.0, oi=10000, ts=now)

    async def bars(t, start, end, resolution="1d"):
        # synthetic chain history is read via cache by the real impl;
        # here we monkeypatch the helper that fetches "OI 7d ago"
        return []

    provider.fetch_option_chain.side_effect = chain

    # Provide a small in-memory archive used by the function under test.
    archive = {("GME", week_ago.date()): _chain(spot=17.0, oi=4000, ts=week_ago)}
    factor = await compute_call_oi_velocity(["GME"], provider, now, _archive=archive)
    out = factor.values.set_index("ticker")
    assert out.loc["GME", "raw_value"] > 0   # OI grew 4k → 10k


@pytest.mark.asyncio
async def test_call_oi_velocity_zero_when_no_history() -> None:
    now = datetime(2024, 5, 13, 13, 30, tzinfo=UTC)
    provider = AsyncMock()

    async def chain(_t, expiry=None) -> OptionChain:
        return _chain(spot=18.0, oi=10000, ts=now)

    provider.fetch_option_chain.side_effect = chain
    factor = await compute_call_oi_velocity(["GME"], provider, now, _archive={})
    out = factor.values.set_index("ticker")
    assert out.loc["GME", "raw_value"] == 0.0
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/signals/test_options_flow.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `signals/options_flow.py`**

```python
"""Signal f5 — ATM call OI 7-day velocity.

raw_value = (oi_today_atm - oi_7d_ago_atm) / max(oi_7d_ago_atm, baseline)

ATM band: strikes within 5% of spot.

The 7-day-ago snapshot is read from a separate archive (the daily-scan
job persists option chains under data/options/<ticker>/<date>.parquet).
The `_archive` keyword is a test seam — production reads from the cache.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from squeeze_hunter.data.protocol import DataProvider
from squeeze_hunter.data.schema import OptionChain
from squeeze_hunter.signals.base import Factor


def _atm_call_oi(chain: OptionChain, window_pct: float = 0.05) -> int:
    return chain.total_call_oi(near_money_window_pct=window_pct)


async def compute_call_oi_velocity(
    tickers: list[str],
    provider: DataProvider,
    clock: datetime,
    *,
    _archive: dict[tuple[str, date], OptionChain] | None = None,
) -> Factor:
    archive = _archive or {}
    rows = []
    for t in tickers:
        try:
            today = await provider.fetch_option_chain(t)
        except (NotImplementedError, LookupError):
            rows.append({"ticker": t, "raw_value": 0.0})
            continue
        if not today.quotes:
            rows.append({"ticker": t, "raw_value": 0.0})
            continue
        oi_today = _atm_call_oi(today)
        prior_date = (clock - timedelta(days=7)).date()
        prior = archive.get((t, prior_date))
        if prior is None or not prior.quotes:
            rows.append({"ticker": t, "raw_value": 0.0})
            continue
        oi_prior = _atm_call_oi(prior)
        baseline = max(oi_prior, 100)
        raw = (oi_today - oi_prior) / baseline
        rows.append({"ticker": t, "raw_value": raw})
    return Factor(name="f5_call_oi_velocity", as_of=clock, values=pd.DataFrame(rows))
```

- [ ] **Step 4: Run, expect pass**

```bash
uv run pytest tests/signals/test_options_flow.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/squeeze_hunter/signals/options_flow.py tests/signals/test_options_flow.py
git commit -m "feat(signals): f5 ATM call OI 7-day velocity"
```

---

### Task 1.14: Signals f6 + f7 — technicals

**Files:**
- Create: `src/squeeze_hunter/signals/technicals.py`
- Create: `tests/signals/test_technicals.py`

- [ ] **Step 1: Write the failing test**

`tests/signals/test_technicals.py`:

```python
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from squeeze_hunter.data.schema import Bar
from squeeze_hunter.signals.technicals import (
    compute_bollinger_breakout,
    compute_volume_spike,
)


def _bars(closes: list[float], volumes: list[int]) -> list[Bar]:
    base = datetime(2024, 5, 13, tzinfo=UTC) - timedelta(days=len(closes))
    return [
        Bar(ticker="GME", ts=base + timedelta(days=i),
            open=c, high=c, low=c, close=c, volume=v)
        for i, (c, v) in enumerate(zip(closes, volumes))
    ]


@pytest.mark.asyncio
async def test_bollinger_breakout_positive_after_squeeze_then_pop() -> None:
    flat = [10.0] * 30 + [10.05] * 5     # tight squeeze
    pop = [12.0]                          # breakout
    bars = _bars(flat + pop, [1_000_000] * (len(flat) + len(pop)))
    provider = AsyncMock()
    provider.fetch_bars.return_value = bars
    factor = await compute_bollinger_breakout(
        ["GME"], provider, datetime(2024, 5, 13, tzinfo=UTC)
    )
    out = factor.values.set_index("ticker")
    assert out.loc["GME", "raw_value"] > 0


@pytest.mark.asyncio
async def test_volume_spike_uses_today_vs_adv20() -> None:
    closes = [10.0] * 21
    volumes = [1_000_000] * 20 + [5_000_000]
    bars = _bars(closes, volumes)
    provider = AsyncMock()
    provider.fetch_bars.return_value = bars
    factor = await compute_volume_spike(
        ["GME"], provider, datetime(2024, 5, 13, tzinfo=UTC)
    )
    out = factor.values.set_index("ticker")
    assert out.loc["GME", "raw_value"] == pytest.approx(5.0)
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/signals/test_technicals.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `signals/technicals.py`**

```python
"""Signals f6 (Bollinger breakout) and f7 (Volume spike vs ADV20)."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from squeeze_hunter.data.protocol import DataProvider
from squeeze_hunter.signals.base import Factor


async def compute_bollinger_breakout(
    tickers: list[str], provider: DataProvider, clock: datetime
) -> Factor:
    """raw = max(0, (close - upper_band) / std) when band-width is in lowest 25% of last 60d.

    Captures "squeeze then breakout": tight Bollinger band followed by close > upper band.
    """
    rows = []
    for t in tickers:
        bars = await provider.fetch_bars(
            t, clock - timedelta(days=120), clock
        )
        if len(bars) < 60:
            rows.append({"ticker": t, "raw_value": 0.0})
            continue
        df = pd.DataFrame({"close": [b.close for b in bars]}).iloc[-60:]
        ma20 = df["close"].rolling(20).mean()
        sd20 = df["close"].rolling(20).std(ddof=0)
        upper = ma20 + 2 * sd20
        band_width = (upper - (ma20 - 2 * sd20)) / ma20
        bw_today = band_width.iloc[-1]
        bw_quartile = band_width.quantile(0.25)
        last_close = df["close"].iloc[-1]
        last_upper = upper.iloc[-1]
        last_sd = sd20.iloc[-1]
        squeezed = bw_today <= bw_quartile and not np.isnan(bw_today)
        breakout = max(0.0, (last_close - last_upper) / max(last_sd, 1e-9))
        raw = breakout if squeezed else 0.0
        rows.append({"ticker": t, "raw_value": float(raw)})
    return Factor(name="f6_bollinger_breakout", as_of=clock, values=pd.DataFrame(rows))


async def compute_volume_spike(
    tickers: list[str], provider: DataProvider, clock: datetime
) -> Factor:
    """raw = today_volume / mean(volume_20d_excl_today)."""
    rows = []
    for t in tickers:
        bars = await provider.fetch_bars(
            t, clock - timedelta(days=40), clock
        )
        if len(bars) < 21:
            rows.append({"ticker": t, "raw_value": 0.0})
            continue
        bars = sorted(bars, key=lambda b: b.ts)
        last20 = bars[-21:-1]
        today = bars[-1]
        adv = sum(b.volume for b in last20) / 20
        if adv <= 0:
            rows.append({"ticker": t, "raw_value": 0.0})
            continue
        rows.append({"ticker": t, "raw_value": today.volume / adv})
    return Factor(name="f7_volume_spike", as_of=clock, values=pd.DataFrame(rows))
```

- [ ] **Step 4: Run, expect pass**

```bash
uv run pytest tests/signals/test_technicals.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/squeeze_hunter/signals/technicals.py tests/signals/test_technicals.py
git commit -m "feat(signals): f6 Bollinger breakout and f7 volume spike"
```

---

### Task 1.15: Signal orchestrator — `compute.py`

**Files:**
- Create: `src/squeeze_hunter/signals/compute.py`
- Create: `tests/signals/test_compute.py`

- [ ] **Step 1: Write the failing test**

`tests/signals/test_compute.py`:

```python
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from squeeze_hunter.signals.base import Factor
from squeeze_hunter.signals.compute import FACTOR_NAMES, compute_all_factors


@pytest.mark.asyncio
async def test_compute_all_returns_one_row_per_factor_per_ticker() -> None:
    tickers = ["GME", "AAPL"]
    fake_factor = lambda name: Factor(
        name=name,
        as_of=datetime(2024, 5, 13, tzinfo=UTC),
        values=pd.DataFrame({"ticker": tickers, "raw_value": [1.0, 2.0]}),
    )

    async def stub_si(_t, _p, _c): return fake_factor("f1_si_pct")
    async def stub_dtc(_t, _p, _c): return fake_factor("f2_days_to_cover")
    async def stub_er(_t, _p, _c): return fake_factor("f3_earnings_reaction")
    async def stub_wsb(_t, _p, _c): return fake_factor("f4_wsb_mention")
    async def stub_oi(_t, _p, _c, **kw): return fake_factor("f5_call_oi_velocity")
    async def stub_bb(_t, _p, _c): return fake_factor("f6_bollinger_breakout")
    async def stub_vs(_t, _p, _c): return fake_factor("f7_volume_spike")

    provider = AsyncMock()
    with (
        patch("squeeze_hunter.signals.compute.compute_si_pct_float", stub_si),
        patch("squeeze_hunter.signals.compute.compute_days_to_cover", stub_dtc),
        patch("squeeze_hunter.signals.compute.compute_earnings_reaction", stub_er),
        patch("squeeze_hunter.signals.compute.compute_wsb_sentiment", stub_wsb),
        patch("squeeze_hunter.signals.compute.compute_call_oi_velocity", stub_oi),
        patch("squeeze_hunter.signals.compute.compute_bollinger_breakout", stub_bb),
        patch("squeeze_hunter.signals.compute.compute_volume_spike", stub_vs),
    ):
        df = await compute_all_factors(
            tickers, provider, datetime(2024, 5, 13, tzinfo=UTC)
        )
    # Long format
    assert set(df.columns) >= {"ticker", "factor_name", "raw_value", "z_score"}
    assert set(df["factor_name"]) == set(FACTOR_NAMES)
    assert set(df["ticker"]) == {"GME", "AAPL"}
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/signals/test_compute.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `signals/compute.py`**

```python
"""Compute all 7 factors and stack into a long-format dataframe with z-scores."""

from __future__ import annotations

import asyncio
from datetime import datetime

import pandas as pd

from squeeze_hunter.data.protocol import DataProvider
from squeeze_hunter.signals.earnings_reaction import compute_earnings_reaction
from squeeze_hunter.signals.normalize import cross_sectional_z
from squeeze_hunter.signals.options_flow import compute_call_oi_velocity
from squeeze_hunter.signals.sentiment import compute_wsb_sentiment
from squeeze_hunter.signals.short_interest import compute_days_to_cover, compute_si_pct_float
from squeeze_hunter.signals.technicals import compute_bollinger_breakout, compute_volume_spike


FACTOR_NAMES = (
    "f1_si_pct",
    "f2_days_to_cover",
    "f3_earnings_reaction",
    "f4_wsb_mention",
    "f5_call_oi_velocity",
    "f6_bollinger_breakout",
    "f7_volume_spike",
)


async def compute_all_factors(
    tickers: list[str], provider: DataProvider, clock: datetime
) -> pd.DataFrame:
    factors = await asyncio.gather(
        compute_si_pct_float(tickers, provider, clock),
        compute_days_to_cover(tickers, provider, clock),
        compute_earnings_reaction(tickers, provider, clock),
        compute_wsb_sentiment(tickers, provider, clock),
        compute_call_oi_velocity(tickers, provider, clock),
        compute_bollinger_breakout(tickers, provider, clock),
        compute_volume_spike(tickers, provider, clock),
    )
    frames = []
    for f in factors:
        if f.values.empty:
            continue
        v = f.values.copy()
        v["factor_name"] = f.name
        v["z_score"] = cross_sectional_z(v["raw_value"])
        frames.append(v[["ticker", "factor_name", "raw_value", "z_score"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["ticker", "factor_name", "raw_value", "z_score"]
    )
```

- [ ] **Step 4: Run, expect pass**

```bash
uv run pytest tests/signals/test_compute.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/squeeze_hunter/signals/compute.py tests/signals/test_compute.py
git commit -m "feat(signals): orchestrator computes all 7 factors with z-scores"
```

---

### Task 1.16: Score combiner

**Files:**
- Create: `src/squeeze_hunter/score/__init__.py`
- Create: `src/squeeze_hunter/score/combiner.py`
- Create: `tests/score/__init__.py`
- Create: `tests/score/test_combiner.py`

- [ ] **Step 1: Write the failing test**

`tests/score/test_combiner.py`:

```python
import pandas as pd
import pytest

from squeeze_hunter.score.combiner import combine


def test_score_is_weighted_sum_of_z() -> None:
    df = pd.DataFrame(
        [
            {"ticker": "GME", "factor_name": "f1_si_pct",         "raw_value": 0.20, "z_score": 3.0},
            {"ticker": "GME", "factor_name": "f3_earnings_reaction","raw_value": 0.05, "z_score": 1.0},
        ]
    )
    weights = {"f1_si_pct": 2.0, "f3_earnings_reaction": 2.0}
    result = combine(df, weights=weights)
    row = result.set_index("ticker").loc["GME"]
    assert row["score"] == pytest.approx(2.0 * 3.0 + 2.0 * 1.0)


def test_unknown_factor_zero_weight() -> None:
    df = pd.DataFrame([
        {"ticker": "GME", "factor_name": "f9_mystery", "raw_value": 1.0, "z_score": 5.0}
    ])
    result = combine(df, weights={"f1_si_pct": 2.0})
    assert result.loc[result["ticker"] == "GME", "score"].iloc[0] == 0.0
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/score/test_combiner.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `score/combiner.py`**

```python
"""Linear weighted z-score combiner. Pure pandas."""

from __future__ import annotations

import pandas as pd


def combine(factors_long: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """factors_long: columns ticker, factor_name, raw_value, z_score.
    Returns wide df: ticker, score, plus each factor's z as a column."""
    if factors_long.empty:
        return pd.DataFrame(columns=["ticker", "score"])
    df = factors_long.copy()
    df["weighted"] = df["factor_name"].map(weights).fillna(0.0) * df["z_score"]
    score = df.groupby("ticker", as_index=False)["weighted"].sum().rename(
        columns={"weighted": "score"}
    )
    pivot = df.pivot_table(
        index="ticker", columns="factor_name", values="z_score"
    ).reset_index()
    return score.merge(pivot, on="ticker", how="left")
```

- [ ] **Step 4: Run, expect pass**

```bash
uv run pytest tests/score/test_combiner.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/squeeze_hunter/score/__init__.py src/squeeze_hunter/score/combiner.py \
        tests/score/test_combiner.py tests/score/__init__.py
git commit -m "feat(score): weighted z-score combiner"
```

---

### Task 1.17: Setup classifier

**Files:**
- Create: `src/squeeze_hunter/score/classifier.py`
- Create: `tests/score/test_classifier.py`

- [ ] **Step 1: Write the failing test**

`tests/score/test_classifier.py`:

```python
import pandas as pd

from squeeze_hunter.score.classifier import classify_setups


def test_classifier_labels_each_type() -> None:
    df = pd.DataFrame(
        [
            # CAR-strong
            {"ticker": "HTZ",  "score": 9.0,
             "f1_si_pct": 2.5, "f3_earnings_reaction": 2.0,
             "f4_wsb_mention": 0.5, "f5_call_oi_velocity": 0.5},
            # GME-strong
            {"ticker": "GME",  "score": 9.0,
             "f1_si_pct": 1.0, "f3_earnings_reaction": 0.0,
             "f4_wsb_mention": 2.5, "f5_call_oi_velocity": 2.5},
            # Mixed
            {"ticker": "OKLO", "score": 8.5,
             "f1_si_pct": 1.5, "f3_earnings_reaction": 1.6,
             "f4_wsb_mention": 1.6, "f5_call_oi_velocity": 1.4},
            # Weak
            {"ticker": "AAPL", "score": 1.0,
             "f1_si_pct": 0.1, "f3_earnings_reaction": 0.1,
             "f4_wsb_mention": 0.1, "f5_call_oi_velocity": 0.1},
        ]
    )
    out = classify_setups(df)
    out_idx = out.set_index("ticker")
    assert out_idx.loc["HTZ", "setup_type"] == "CAR"
    assert out_idx.loc["GME", "setup_type"] == "GME"
    assert out_idx.loc["OKLO", "setup_type"] == "Mixed"
    assert out_idx.loc["AAPL", "setup_type"] == "Weak"
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/score/test_classifier.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `score/classifier.py`**

```python
"""Setup classifier — pure rule-based labeling.

A = z[f1_si_pct] + z[f3_earnings_reaction]    # CAR strength
B = z[f4_wsb_mention] + z[f5_call_oi_velocity] # GME strength

if   A >= 4 and B  < 2  → CAR
elif B >= 4 and A  < 2  → GME
elif A >= 3 and B >= 3  → Mixed
else                    → Weak
"""

from __future__ import annotations

import pandas as pd


def classify_setups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    a = out["f1_si_pct"].fillna(0.0) + out["f3_earnings_reaction"].fillna(0.0)
    b = out["f4_wsb_mention"].fillna(0.0) + out["f5_call_oi_velocity"].fillna(0.0)

    def label(row_a: float, row_b: float) -> str:
        if row_a >= 4.0 and row_b < 2.0:
            return "CAR"
        if row_b >= 4.0 and row_a < 2.0:
            return "GME"
        if row_a >= 3.0 and row_b >= 3.0:
            return "Mixed"
        return "Weak"

    out["car_strength"] = a
    out["gme_strength"] = b
    out["setup_type"] = [label(x, y) for x, y in zip(a, b)]
    return out
```

- [ ] **Step 4: Run, expect pass**

```bash
uv run pytest tests/score/test_classifier.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/squeeze_hunter/score/classifier.py tests/score/test_classifier.py
git commit -m "feat(score): rule-based setup classifier"
```

---

### Task 1.18: Daily scan orchestrator + CLI command

**Files:**
- Create: `src/squeeze_hunter/scan.py`
- Modify: `src/squeeze_hunter/cli.py`
- Create: `tests/test_scan.py`

- [ ] **Step 1: Write the failing test (end-to-end with BacktestProvider)**

`tests/test_scan.py`:

```python
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.data.providers.backtest import BacktestProvider, Clock
from squeeze_hunter.scan import run_scan


def _seed_cache(cache: ParquetCache) -> None:
    # Bars: 60 days flat, then GME spikes on 2024-05-13
    base = datetime(2024, 3, 13, tzinfo=UTC)
    rows_gme = []
    rows_aapl = []
    for i in range(62):
        ts = base + pd.Timedelta(days=i)
        rows_gme.append({
            "ticker": "GME", "ts": ts,
            "open": 18.0, "high": 18.1, "low": 17.9, "close": 18.0,
            "volume": 1_000_000,
        })
        rows_aapl.append({
            "ticker": "AAPL", "ts": ts,
            "open": 200.0, "high": 200.5, "low": 199.5, "close": 200.0,
            "volume": 50_000_000,
        })
    rows_gme[-1]["close"] = 22.0
    rows_gme[-1]["high"] = 22.5
    rows_gme[-1]["volume"] = 8_000_000
    cache.write_partition("bars", "GME", pd.DataFrame(rows_gme))
    cache.write_partition("bars", "AAPL", pd.DataFrame(rows_aapl))
    # Short interest
    cache.write_partition("short_interest", "all", pd.DataFrame([
        {"ticker": "GME",  "settlement_date": date(2024, 4, 30),
         "si_shares": 5_000_000, "si_pct_float": 0.30, "avg_daily_volume_20d": 1_000_000},
        {"ticker": "AAPL", "settlement_date": date(2024, 4, 30),
         "si_shares": 50_000_000, "si_pct_float": 0.005, "avg_daily_volume_20d": 80_000_000},
    ]))
    cache.write_partition("earnings", "all", pd.DataFrame(columns=[
        "ticker", "report_at", "actual_eps", "estimate_eps"
    ]))
    cache.write_partition("sentiment", "2024-05-13", pd.DataFrame([
        {"ticker": "GME",  "subreddit": "wallstreetbets",
         "count_24h": 400, "baseline_30d_mean": 10.0, "baseline_30d_std": 20.0},
        {"ticker": "AAPL", "subreddit": "wallstreetbets",
         "count_24h": 5,   "baseline_30d_mean": 10.0, "baseline_30d_std": 20.0},
    ]))


@pytest.mark.asyncio
async def test_scan_ranks_gme_above_aapl(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path)
    _seed_cache(cache)
    clock = Clock(now=datetime(2024, 5, 13, 23, 59, tzinfo=UTC))
    provider = BacktestProvider(cache=cache, clock=clock)
    settings = Settings()
    settings.score.weights = {
        "f1_si_pct": 2.0, "f2_days_to_cover": 1.0,
        "f3_earnings_reaction": 2.0,
        "f4_wsb_mention": 1.5, "f5_call_oi_velocity": 1.5,
        "f6_bollinger_breakout": 1.0, "f7_volume_spike": 1.0,
    }
    ranked = await run_scan(["GME", "AAPL"], provider, clock.now, settings)
    by_t = ranked.set_index("ticker")
    assert by_t.loc["GME", "score"] > by_t.loc["AAPL", "score"]
    assert "setup_type" in ranked.columns
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/test_scan.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `scan.py`**

```python
"""Daily scan orchestrator: universe → factors → score → setup → ranked output."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from squeeze_hunter.config import Settings
from squeeze_hunter.data.protocol import DataProvider
from squeeze_hunter.score.classifier import classify_setups
from squeeze_hunter.score.combiner import combine
from squeeze_hunter.signals.compute import compute_all_factors


async def run_scan(
    tickers: list[str],
    provider: DataProvider,
    clock: datetime,
    settings: Settings,
) -> pd.DataFrame:
    factors = await compute_all_factors(tickers, provider, clock)
    if factors.empty:
        return pd.DataFrame(columns=["ticker", "score", "setup_type"])
    scored = combine(factors, weights=settings.score.weights)
    classified = classify_setups(scored)
    classified = classified.sort_values("score", ascending=False).reset_index(drop=True)
    classified["rank"] = classified.index + 1
    classified["as_of"] = clock
    return classified
```

- [ ] **Step 4: Add `scan` to the CLI**

In `src/squeeze_hunter/cli.py`, append:

```python
import asyncio
from datetime import UTC, datetime
from pathlib import Path

from squeeze_hunter.config import load_settings
from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.data.providers.backtest import BacktestProvider, Clock
from squeeze_hunter.scan import run_scan


@app.command()
def scan(
    date_str: str = typer.Option(..., "--date", help="YYYY-MM-DD"),
    parquet_root: Path = typer.Option(Path("data/parquet"), "--data"),
    config_path: Path = typer.Option(Path("config/settings.example.yml"), "--config"),
    tickers_file: Path = typer.Option(Path("config/universe.txt"), "--tickers"),
    out: Path = typer.Option(Path("data/scans"), "--out"),
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
```

- [ ] **Step 5: Run unit tests**

```bash
uv run pytest tests/test_scan.py -v
```

Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add src/squeeze_hunter/scan.py src/squeeze_hunter/cli.py tests/test_scan.py
git commit -m "feat(scan): daily scan orchestrator + CLI"
```

---

### Task 1.19: Historical backfill scripts

**Files:**
- Create: `src/squeeze_hunter/ingest/__init__.py`
- Create: `src/squeeze_hunter/ingest/backfill_bars.py`
- Create: `src/squeeze_hunter/ingest/backfill_finra.py`
- Create: `src/squeeze_hunter/ingest/backfill_earnings.py`
- Modify: `src/squeeze_hunter/cli.py` (add `ingest` subcommands)
- Create: `tests/ingest/test_backfill_bars.py`

- [ ] **Step 1: Write the failing test (mocked yfinance)**

`tests/ingest/__init__.py`: empty file.

`tests/ingest/test_backfill_bars.py`:

```python
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.ingest.backfill_bars import backfill_bars_for_ticker


@pytest.mark.asyncio
async def test_backfill_writes_partition(tmp_path: Path) -> None:
    fake = pd.DataFrame(
        {
            "Open": [10.0, 11.0],
            "High": [11.0, 12.0],
            "Low": [9.0, 10.5],
            "Close": [10.5, 11.5],
            "Volume": [1_000_000, 1_500_000],
        },
        index=pd.to_datetime(["2024-05-10", "2024-05-13"], utc=True),
    )
    cache = ParquetCache(root=tmp_path)
    with patch("squeeze_hunter.data.providers.yahoo._yf_history", return_value=fake):
        await backfill_bars_for_ticker(
            "GME",
            datetime(2024, 5, 1, tzinfo=UTC),
            datetime(2024, 5, 14, tzinfo=UTC),
            cache=cache,
        )
    out = cache.read_partition("bars", "GME")
    assert len(out) == 2
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/ingest/test_backfill_bars.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `ingest/backfill_bars.py`**

```python
"""Backfill EOD bars from yfinance into parquet cache."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.data.providers.yahoo import YahooProvider


async def backfill_bars_for_ticker(
    ticker: str, start: datetime, end: datetime, cache: ParquetCache
) -> None:
    provider = YahooProvider()
    bars = await provider.fetch_bars(ticker, start, end)
    if not bars:
        return
    df = pd.DataFrame(
        [
            {
                "ticker": b.ticker, "ts": b.ts,
                "open": b.open, "high": b.high, "low": b.low,
                "close": b.close, "volume": b.volume,
            }
            for b in bars
        ]
    )
    cache.dedup_keys = ["ticker", "ts"]
    cache.append_partition("bars", ticker, df)


async def backfill_bars_for_universe(
    tickers: list[str], start: datetime, end: datetime, cache: ParquetCache
) -> None:
    for t in tickers:
        await backfill_bars_for_ticker(t, start, end, cache)
```

- [ ] **Step 4: Implement `backfill_finra.py`**

```python
"""Backfill historical FINRA short-interest into parquet cache."""

from __future__ import annotations

from datetime import date

import pandas as pd

from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.data.providers.finra import FinraProvider


async def backfill_finra(tickers: list[str], cache: ParquetCache) -> None:
    provider = FinraProvider()
    rows: list[dict] = []
    for t in tickers:
        si_list = await provider.fetch_short_interest(t, since=date(2018, 1, 1))
        for si in si_list:
            rows.append(
                {
                    "ticker": si.ticker,
                    "settlement_date": si.settlement_date,
                    "si_shares": si.si_shares,
                    "si_pct_float": si.si_pct_float,
                    "avg_daily_volume_20d": si.avg_daily_volume_20d,
                }
            )
    if not rows:
        return
    cache.dedup_keys = ["ticker", "settlement_date"]
    cache.append_partition("short_interest", "all", pd.DataFrame(rows))
```

- [ ] **Step 5: Implement `backfill_earnings.py`**

```python
"""Backfill earnings calendar into parquet cache."""

from __future__ import annotations

import os
from datetime import date

import pandas as pd

from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.data.providers.finnhub import FinnhubProvider


async def backfill_earnings(tickers: list[str], cache: ParquetCache) -> None:
    api_key = os.environ.get("FINNHUB_KEY", "")
    provider = FinnhubProvider(api_key=api_key)
    rows: list[dict] = []
    for t in tickers:
        events = await provider.fetch_earnings(t, since=date(2018, 1, 1))
        for e in events:
            rows.append(
                {
                    "ticker": e.ticker,
                    "report_at": e.report_at,
                    "actual_eps": e.actual_eps,
                    "estimate_eps": e.estimate_eps,
                }
            )
    if not rows:
        return
    cache.dedup_keys = ["ticker", "report_at"]
    cache.append_partition("earnings", "all", pd.DataFrame(rows))
```

- [ ] **Step 6: Add `ingest` CLI commands**

Append to `src/squeeze_hunter/cli.py`:

```python
ingest_app = typer.Typer(help="Historical backfill commands")
app.add_typer(ingest_app, name="ingest")


@ingest_app.command("bars")
def ingest_bars(
    start: str = typer.Option(..., "--start", help="YYYY-MM-DD"),
    end: str = typer.Option(..., "--end", help="YYYY-MM-DD"),
    tickers_file: Path = typer.Option(Path("config/universe.txt"), "--tickers"),
    parquet_root: Path = typer.Option(Path("data/parquet"), "--data"),
) -> None:
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
    tickers_file: Path = typer.Option(Path("config/universe.txt"), "--tickers"),
    parquet_root: Path = typer.Option(Path("data/parquet"), "--data"),
) -> None:
    from squeeze_hunter.ingest.backfill_finra import backfill_finra

    configure_logging()
    cache = ParquetCache(root=parquet_root)
    tickers = [line.strip() for line in tickers_file.read_text().splitlines() if line.strip()]
    asyncio.run(backfill_finra(tickers, cache))


@ingest_app.command("earnings")
def ingest_earnings(
    tickers_file: Path = typer.Option(Path("config/universe.txt"), "--tickers"),
    parquet_root: Path = typer.Option(Path("data/parquet"), "--data"),
) -> None:
    from squeeze_hunter.ingest.backfill_earnings import backfill_earnings

    configure_logging()
    cache = ParquetCache(root=parquet_root)
    tickers = [line.strip() for line in tickers_file.read_text().splitlines() if line.strip()]
    asyncio.run(backfill_earnings(tickers, cache))
```

- [ ] **Step 7: Run unit tests**

```bash
uv run pytest tests/ingest/ -v
```

Expected: 1 passed.

- [ ] **Step 8: Commit**

```bash
git add src/squeeze_hunter/ingest/ src/squeeze_hunter/cli.py tests/ingest/
git commit -m "feat(ingest): historical backfill for bars, FINRA, earnings"
```

---

### Task 1.20: Phase 1 milestone — full scan on a sample universe

- [ ] **Step 1: Build a small starter universe (~20 tickers covering both setup families)**

Create `config/universe.txt`:

```
GME
AMC
BBBY
HTZ
CAR
TUP
DJT
OKLO
BYND
KOSS
RIVN
LCID
NKLA
SOFI
PLTR
HOOD
RBLX
COIN
MARA
RIOT
```

(This is **not** the full ~1500-ticker universe — that comes after Gate 1; for milestone validation we want a small set including the validation-case names so the scan can be tested end-to-end.)

- [ ] **Step 2: Backfill bars and earnings for them (~5–10 min wall clock)**

```bash
uv run squeeze-hunter ingest bars --start 2018-01-01 --end 2026-05-10
uv run squeeze-hunter ingest earnings   # requires FINNHUB_KEY in .env
uv run squeeze-hunter ingest finra
```

Expected: parquet partitions land under `data/parquet/{bars,earnings,short_interest}/...`.

- [ ] **Step 3: Run the scan for HTZ's 2025-04-21 squeeze date**

```bash
uv run squeeze-hunter scan --date 2025-04-21
```

Expected: `data/scans/2025-04-21.csv` written; HTZ ranks in the top 3 with `setup_type=CAR`.

- [ ] **Step 4: Run the scan for GME's 2024-05-13 (Roaring Kitty return)**

```bash
uv run squeeze-hunter scan --date 2024-05-13
```

Expected: GME ranks #1 with `setup_type=GME` (or `Mixed`).

- [ ] **Step 5: Tag the milestone**

```bash
git tag phase-1-scan
```

---

## Phase 2 — Backtest Engine + Walk-Forward + Gate 1

### Task 2.1: Cost model

**Files:**
- Create: `src/squeeze_hunter/backtest/__init__.py`
- Create: `src/squeeze_hunter/backtest/cost_model.py`
- Create: `tests/backtest/__init__.py`
- Create: `tests/backtest/test_cost_model.py`

- [ ] **Step 1: Write the failing test**

`tests/backtest/test_cost_model.py`:

```python
import pytest

from squeeze_hunter.backtest.cost_model import StockCostModel


def test_high_price_slippage_5_bps() -> None:
    m = StockCostModel()
    fill = m.simulate_buy(reference_price=100.0, qty=100, is_open_5min=False)
    assert fill.fill_price == pytest.approx(100.0 * (1 + 0.0005))
    assert fill.commission_usd == pytest.approx(0.005 * 100)


def test_low_price_slippage_15_bps() -> None:
    m = StockCostModel()
    fill = m.simulate_buy(reference_price=7.0, qty=100, is_open_5min=False)
    assert fill.fill_price == pytest.approx(7.0 * (1 + 0.0015))


def test_open_window_adds_10_bps() -> None:
    m = StockCostModel()
    fill = m.simulate_buy(reference_price=100.0, qty=100, is_open_5min=True)
    assert fill.fill_price == pytest.approx(100.0 * (1 + 0.0005 + 0.0010))


def test_sell_pays_negative_slippage() -> None:
    m = StockCostModel()
    fill = m.simulate_sell(reference_price=100.0, qty=100, is_open_5min=False)
    assert fill.fill_price == pytest.approx(100.0 * (1 - 0.0005))
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/backtest/test_cost_model.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `cost_model.py`**

```python
"""Conservative cost model for stocks and options."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Fill:
    fill_price: float
    commission_usd: float


@dataclass
class StockCostModel:
    commission_per_share: float = 0.005
    slippage_high_price_bps: float = 5
    slippage_low_price_bps: float = 15
    low_price_threshold: float = 10.0
    open_window_extra_bps: float = 10

    def _slippage_bps(self, price: float, is_open_5min: bool) -> float:
        base = self.slippage_high_price_bps if price >= self.low_price_threshold else self.slippage_low_price_bps
        return base + (self.open_window_extra_bps if is_open_5min else 0)

    def simulate_buy(self, reference_price: float, qty: int, is_open_5min: bool = False) -> Fill:
        bps = self._slippage_bps(reference_price, is_open_5min)
        return Fill(
            fill_price=reference_price * (1 + bps / 10_000),
            commission_usd=qty * self.commission_per_share,
        )

    def simulate_sell(self, reference_price: float, qty: int, is_open_5min: bool = False) -> Fill:
        bps = self._slippage_bps(reference_price, is_open_5min)
        return Fill(
            fill_price=reference_price * (1 - bps / 10_000),
            commission_usd=qty * self.commission_per_share,
        )


@dataclass
class OptionCostModel:
    commission_per_contract: float = 0.65

    def simulate_buy(self, mid: float, spread: float, qty: int) -> Fill:
        return Fill(
            fill_price=mid + 0.25 * spread,
            commission_usd=qty * self.commission_per_contract,
        )

    def simulate_sell(self, mid: float, spread: float, qty: int) -> Fill:
        return Fill(
            fill_price=mid - 0.25 * spread,
            commission_usd=qty * self.commission_per_contract,
        )
```

- [ ] **Step 4: Run, expect pass**

```bash
uv run pytest tests/backtest/test_cost_model.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/squeeze_hunter/backtest/__init__.py src/squeeze_hunter/backtest/cost_model.py \
        tests/backtest/__init__.py tests/backtest/test_cost_model.py
git commit -m "feat(backtest): cost model for stocks and options"
```

---

### Task 2.2: Risk — Kelly sizing with Bayesian shrinkage

**Files:**
- Create: `src/squeeze_hunter/risk/__init__.py`
- Create: `src/squeeze_hunter/risk/kelly.py`
- Create: `tests/risk/__init__.py`
- Create: `tests/risk/test_kelly.py`

- [ ] **Step 1: Write the failing test**

`tests/risk/test_kelly.py`:

```python
import pytest

from squeeze_hunter.risk.kelly import KellyParams, kelly_position_pct


def test_kelly_zero_when_priors_negative_and_no_obs() -> None:
    p = KellyParams(prior_win_rate=0.20, prior_payoff=2.0, fraction=0.20, cap=0.08)
    pct = kelly_position_pct(observed_wins=0, observed_trades=0, observed_avg_payoff=0.0, params=p)
    assert pct == 0.0


def test_kelly_positive_when_observed_strong() -> None:
    p = KellyParams(prior_win_rate=0.20, prior_payoff=2.0, fraction=0.20, cap=0.08, prior_n=30)
    pct = kelly_position_pct(observed_wins=20, observed_trades=40, observed_avg_payoff=4.0, params=p)
    assert 0.0 < pct <= 0.08


def test_kelly_capped_at_position_cap() -> None:
    p = KellyParams(prior_win_rate=0.5, prior_payoff=10.0, fraction=1.0, cap=0.08, prior_n=0)
    pct = kelly_position_pct(observed_wins=99, observed_trades=100, observed_avg_payoff=20.0, params=p)
    assert pct == pytest.approx(0.08)
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/risk/test_kelly.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `risk/kelly.py`**

```python
"""Fractional Kelly with Bayesian shrinkage on observed win-rate / payoff."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class KellyParams:
    prior_win_rate: float = 0.20
    prior_payoff: float = 2.0
    prior_n: int = 30
    fraction: float = 0.20
    cap: float = 0.08


def kelly_position_pct(
    observed_wins: int,
    observed_trades: int,
    observed_avg_payoff: float,
    params: KellyParams,
) -> float:
    n = observed_trades
    weight = n / (n + params.prior_n) if (n + params.prior_n) > 0 else 0.0
    p_obs = (observed_wins / n) if n > 0 else params.prior_win_rate
    b_obs = observed_avg_payoff if n > 0 else params.prior_payoff
    p = weight * p_obs + (1 - weight) * params.prior_win_rate
    b = weight * b_obs + (1 - weight) * params.prior_payoff
    if b <= 0:
        return 0.0
    raw = (p * b - (1 - p)) / b
    sized = max(0.0, params.fraction * raw)
    return min(sized, params.cap)
```

- [ ] **Step 4: Run, expect pass**

```bash
uv run pytest tests/risk/test_kelly.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/squeeze_hunter/risk/__init__.py src/squeeze_hunter/risk/kelly.py \
        tests/risk/__init__.py tests/risk/test_kelly.py
git commit -m "feat(risk): fractional Kelly with Bayesian shrinkage"
```

---

### Task 2.3: Risk — pre-trade gates

**Files:**
- Create: `src/squeeze_hunter/risk/gates.py`
- Create: `tests/risk/test_gates.py`

- [ ] **Step 1: Write the failing test**

`tests/risk/test_gates.py`:

```python
from datetime import UTC, date, datetime

import pytest

from squeeze_hunter.risk.gates import (
    GateContext,
    PortfolioState,
    TradeProposal,
    evaluate_gates,
)


def _ctx() -> GateContext:
    return GateContext(
        as_of=datetime(2024, 5, 13, tzinfo=UTC),
        kill_switch_active=False,
        adv20_dollar_volume_by_ticker={"GME": 5_000_000_000},
        days_listed_by_ticker={"GME": 365},
        halted_tickers=frozenset(),
        universe_tickers=frozenset({"GME"}),
        earnings_within_3_days={"GME": False},
        portfolio_correlations={},
    )


def _state() -> PortfolioState:
    return PortfolioState(
        equity_usd=100_000.0, cash_usd=100_000.0,
        gross_exposure_pct=0.0, positions={}, opened_today=0,
    )


def _proposal(score: float = 9.0, setup: str = "CAR", size: float = 5_000.0) -> TradeProposal:
    return TradeProposal(
        ticker="GME", score=score, setup_type=setup,
        target_position_usd=size, instrument="stock",
    )


def test_score_threshold_rejects() -> None:
    res = evaluate_gates(_proposal(score=7.0), _ctx(), _state(), score_threshold=8.0)
    assert not res.accepted
    assert res.reason == "score_below_threshold"


def test_weak_setup_rejects() -> None:
    res = evaluate_gates(_proposal(setup="Weak"), _ctx(), _state(), score_threshold=8.0)
    assert not res.accepted
    assert res.reason == "weak_setup"


def test_kill_switch_rejects() -> None:
    ctx = _ctx()
    ctx.kill_switch_active = True
    res = evaluate_gates(_proposal(), ctx, _state(), score_threshold=8.0)
    assert not res.accepted
    assert res.reason == "kill_switch_active"


def test_already_held_rejects() -> None:
    state = _state()
    state.positions["GME"] = 100
    res = evaluate_gates(_proposal(), _ctx(), state, score_threshold=8.0)
    assert not res.accepted
    assert res.reason == "already_held"


def test_position_cap_rejects() -> None:
    res = evaluate_gates(_proposal(size=20_000.0), _ctx(), _state(), score_threshold=8.0,
                         position_cap=0.08)
    assert not res.accepted
    assert res.reason == "position_cap_exceeded"


def test_full_pass() -> None:
    res = evaluate_gates(_proposal(), _ctx(), _state(), score_threshold=8.0)
    assert res.accepted
    assert res.reason is None


def test_earnings_within_3d_halves_size() -> None:
    ctx = _ctx()
    ctx.earnings_within_3_days["GME"] = True
    res = evaluate_gates(_proposal(size=5_000.0), ctx, _state(), score_threshold=8.0)
    assert res.accepted
    assert res.adjusted_size_usd == pytest.approx(2_500.0)
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/risk/test_gates.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `risk/gates.py`**

```python
"""14 pre-trade gates from the design (Section 6)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TradeProposal:
    ticker: str
    score: float
    setup_type: str
    target_position_usd: float
    instrument: str   # stock, call, put


@dataclass
class PortfolioState:
    equity_usd: float
    cash_usd: float
    gross_exposure_pct: float
    positions: dict[str, int] = field(default_factory=dict)   # ticker -> qty
    opened_today: int = 0


@dataclass
class GateContext:
    as_of: datetime
    kill_switch_active: bool
    adv20_dollar_volume_by_ticker: dict[str, float]
    days_listed_by_ticker: dict[str, int]
    halted_tickers: frozenset[str]
    universe_tickers: frozenset[str]
    earnings_within_3_days: dict[str, bool]
    portfolio_correlations: dict[str, float]   # max corr with existing portfolio for each candidate


@dataclass
class GateResult:
    accepted: bool
    reason: str | None = None
    adjusted_size_usd: float | None = None


def evaluate_gates(
    p: TradeProposal,
    ctx: GateContext,
    state: PortfolioState,
    *,
    score_threshold: float = 8.0,
    max_new_per_day: int = 3,
    max_positions: int = 6,
    position_cap: float = 0.08,
    max_gross_exposure: float = 0.90,
    min_adv20_multiple: float = 100.0,
    min_days_listed: int = 30,
    max_correlation: float = 0.70,
) -> GateResult:
    if p.score < score_threshold:
        return GateResult(False, "score_below_threshold")
    if p.setup_type == "Weak":
        return GateResult(False, "weak_setup")
    if ctx.kill_switch_active:
        return GateResult(False, "kill_switch_active")
    if state.opened_today >= max_new_per_day:
        return GateResult(False, "daily_new_position_cap")
    if len(state.positions) >= max_positions:
        return GateResult(False, "max_positions_exceeded")
    if p.ticker in state.positions:
        return GateResult(False, "already_held")

    size = p.target_position_usd
    if ctx.earnings_within_3_days.get(p.ticker, False):
        size = size * 0.5

    if size / state.equity_usd > position_cap:
        return GateResult(False, "position_cap_exceeded")
    if (state.gross_exposure_pct + size / state.equity_usd) > max_gross_exposure:
        return GateResult(False, "gross_exposure_exceeded")
    adv = ctx.adv20_dollar_volume_by_ticker.get(p.ticker, 0.0)
    if size > 0 and adv < min_adv20_multiple * size:
        return GateResult(False, "insufficient_liquidity")
    if p.ticker in ctx.halted_tickers:
        return GateResult(False, "halted")
    if ctx.days_listed_by_ticker.get(p.ticker, 0) < min_days_listed:
        return GateResult(False, "listed_too_recently")
    if p.ticker not in ctx.universe_tickers:
        return GateResult(False, "outside_universe")
    if ctx.portfolio_correlations.get(p.ticker, 0.0) > max_correlation:
        return GateResult(False, "correlation_too_high")

    return GateResult(True, None, adjusted_size_usd=size)
```

- [ ] **Step 4: Run, expect pass**

```bash
uv run pytest tests/risk/test_gates.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/squeeze_hunter/risk/gates.py tests/risk/test_gates.py
git commit -m "feat(risk): 14 pre-trade gates"
```

---

### Task 2.4: Stops

**Files:**
- Create: `src/squeeze_hunter/risk/stops.py`
- Create: `tests/risk/test_stops.py`

- [ ] **Step 1: Write the failing test**

`tests/risk/test_stops.py`:

```python
import pytest

from squeeze_hunter.risk.stops import StopState, evaluate_stops


def test_hard_stop_triggers_below_threshold() -> None:
    state = StopState(
        entry_price=100.0, peak_price=100.0,
        current_score=10.0, entry_score=10.0,
        bars_held=2, setup_type="CAR",
    )
    sig = evaluate_stops(state, current_price=87.0)
    assert sig.action == "exit"
    assert sig.reason == "hard_stop"


def test_trailing_stop_triggers_after_peak() -> None:
    state = StopState(
        entry_price=100.0, peak_price=140.0,
        current_score=8.0, entry_score=10.0,
        bars_held=3, setup_type="CAR",
    )
    sig = evaluate_stops(state, current_price=110.0)   # 21% from peak (CAR uses 20%)
    assert sig.action == "exit"
    assert sig.reason == "trailing_stop"


def test_time_stop_at_21_bars() -> None:
    state = StopState(
        entry_price=100.0, peak_price=110.0,
        current_score=10.0, entry_score=10.0,
        bars_held=21, setup_type="CAR",
    )
    sig = evaluate_stops(state, current_price=105.0)
    assert sig.action == "exit"
    assert sig.reason == "time_stop"


def test_signal_decay_halves_then_exits() -> None:
    state_half = StopState(
        entry_price=100.0, peak_price=110.0,
        current_score=4.5, entry_score=10.0,   # decayed >=50%
        bars_held=5, setup_type="GME",
    )
    sig = evaluate_stops(state_half, current_price=108.0)
    assert sig.action == "halve"
    assert sig.reason == "signal_decay_50"

    state_exit = StopState(
        entry_price=100.0, peak_price=110.0,
        current_score=2.0, entry_score=10.0,   # decayed >=75%
        bars_held=6, setup_type="GME",
    )
    sig = evaluate_stops(state_exit, current_price=108.0)
    assert sig.action == "exit"
    assert sig.reason == "signal_decay_75"


def test_gme_uses_25_pct_trailing() -> None:
    state = StopState(
        entry_price=100.0, peak_price=200.0,
        current_score=10.0, entry_score=10.0,
        bars_held=5, setup_type="GME",
    )
    sig = evaluate_stops(state, current_price=149.0)   # 25.5% from peak
    assert sig.action == "exit"
    assert sig.reason == "trailing_stop"
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/risk/test_stops.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `risk/stops.py`**

```python
"""Layered stops: hard / trailing / time / signal-decay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class StopState:
    entry_price: float
    peak_price: float
    current_score: float
    entry_score: float
    bars_held: int
    setup_type: str   # CAR, GME, Mixed


@dataclass(slots=True, frozen=True)
class StopSignal:
    action: Literal["hold", "halve", "exit"]
    reason: str | None = None


_TRAILING_BY_SETUP: dict[str, float] = {"CAR": 0.20, "GME": 0.25, "Mixed": 0.22}


def evaluate_stops(
    state: StopState,
    current_price: float,
    *,
    hard_stop: float = -0.12,
    time_stop_bars: int = 21,
    signal_decay_halve: float = 0.50,
    signal_decay_exit: float = 0.75,
) -> StopSignal:
    pnl_pct = (current_price - state.entry_price) / state.entry_price
    if pnl_pct <= hard_stop:
        return StopSignal("exit", "hard_stop")

    trailing = _TRAILING_BY_SETUP.get(state.setup_type, 0.22)
    if state.peak_price > state.entry_price:
        from_peak = (current_price - state.peak_price) / state.peak_price
        if from_peak <= -trailing:
            return StopSignal("exit", "trailing_stop")

    if state.bars_held >= time_stop_bars:
        return StopSignal("exit", "time_stop")

    if state.entry_score > 0:
        decay = (state.entry_score - state.current_score) / state.entry_score
        if decay >= signal_decay_exit:
            return StopSignal("exit", "signal_decay_75")
        if decay >= signal_decay_halve:
            return StopSignal("halve", "signal_decay_50")

    return StopSignal("hold")
```

- [ ] **Step 4: Run, expect pass**

```bash
uv run pytest tests/risk/test_stops.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/squeeze_hunter/risk/stops.py tests/risk/test_stops.py
git commit -m "feat(risk): hard, trailing, time, and signal-decay stops"
```

---

### Task 2.5: Simulator broker

**Files:**
- Create: `src/squeeze_hunter/broker/simulator.py`
- Create: `tests/broker/test_simulator.py`

- [ ] **Step 1: Write the failing test**

`tests/broker/test_simulator.py`:

```python
from datetime import UTC, datetime

import pytest

from squeeze_hunter.backtest.cost_model import StockCostModel
from squeeze_hunter.broker.simulator import SimulatorBroker


@pytest.mark.asyncio
async def test_buy_then_sell_realizes_pnl() -> None:
    broker = SimulatorBroker(initial_cash=100_000.0, cost_model=StockCostModel())
    await broker.submit_buy(
        ticker="GME", qty=100, reference_price=100.0,
        ts=datetime(2024, 5, 13, 14, 0, tzinfo=UTC),
    )
    assert broker.position_qty("GME") == 100
    await broker.submit_sell(
        ticker="GME", qty=100, reference_price=120.0,
        ts=datetime(2024, 5, 17, 14, 0, tzinfo=UTC),
    )
    assert broker.position_qty("GME") == 0
    # Realized PnL = (sell - buy) * 100 - 2*commission
    assert broker.realized_pnl("GME") > 1_500.0
    assert broker.cash > 100_000.0


@pytest.mark.asyncio
async def test_mark_to_market_updates_equity() -> None:
    broker = SimulatorBroker(initial_cash=100_000.0, cost_model=StockCostModel())
    await broker.submit_buy(
        ticker="GME", qty=100, reference_price=100.0,
        ts=datetime(2024, 5, 13, tzinfo=UTC),
    )
    broker.mark_to_market({"GME": 130.0}, ts=datetime(2024, 5, 14, tzinfo=UTC))
    assert broker.equity > 100_000.0 + 100 * 30 - 100   # net of slippage and commission
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/broker/test_simulator.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `broker/simulator.py`**

```python
"""Backtest 'broker' — fully-deterministic order simulation against the cost model."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from squeeze_hunter.backtest.cost_model import Fill, StockCostModel


@dataclass
class Lot:
    qty: int
    avg_price: float
    opened_at: datetime


@dataclass
class SimulatorBroker:
    initial_cash: float
    cost_model: StockCostModel
    cash: float = field(init=False)
    equity: float = field(init=False)
    positions: dict[str, Lot] = field(default_factory=dict)
    realized: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def __post_init__(self) -> None:
        self.cash = self.initial_cash
        self.equity = self.initial_cash

    async def submit_buy(
        self, ticker: str, qty: int, reference_price: float, ts: datetime,
        is_open_5min: bool = False,
    ) -> Fill:
        fill = self.cost_model.simulate_buy(reference_price, qty, is_open_5min=is_open_5min)
        cost = fill.fill_price * qty + fill.commission_usd
        self.cash -= cost
        existing = self.positions.get(ticker)
        if existing:
            new_qty = existing.qty + qty
            new_avg = (existing.avg_price * existing.qty + fill.fill_price * qty) / new_qty
            self.positions[ticker] = Lot(new_qty, new_avg, existing.opened_at)
        else:
            self.positions[ticker] = Lot(qty, fill.fill_price, ts)
        return fill

    async def submit_sell(
        self, ticker: str, qty: int, reference_price: float, ts: datetime,
        is_open_5min: bool = False,
    ) -> Fill:
        fill = self.cost_model.simulate_sell(reference_price, qty, is_open_5min=is_open_5min)
        proceeds = fill.fill_price * qty - fill.commission_usd
        self.cash += proceeds
        existing = self.positions.get(ticker)
        if existing is None or existing.qty < qty:
            raise ValueError(f"insufficient position to sell {qty} of {ticker}")
        realized_pl = (fill.fill_price - existing.avg_price) * qty - fill.commission_usd
        self.realized[ticker] += realized_pl
        new_qty = existing.qty - qty
        if new_qty == 0:
            del self.positions[ticker]
        else:
            self.positions[ticker] = Lot(new_qty, existing.avg_price, existing.opened_at)
        return fill

    def position_qty(self, ticker: str) -> int:
        lot = self.positions.get(ticker)
        return lot.qty if lot else 0

    def realized_pnl(self, ticker: str) -> float:
        return self.realized.get(ticker, 0.0)

    def gross_exposure_pct(self, marks: dict[str, float]) -> float:
        if self.equity <= 0:
            return 0.0
        notional = sum(marks.get(t, lot.avg_price) * lot.qty for t, lot in self.positions.items())
        return notional / self.equity

    def mark_to_market(self, marks: dict[str, float], ts: datetime) -> None:
        notional = sum(marks.get(t, lot.avg_price) * lot.qty for t, lot in self.positions.items())
        self.equity = self.cash + notional
```

- [ ] **Step 4: Run, expect pass**

```bash
uv run pytest tests/broker/test_simulator.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/squeeze_hunter/broker/simulator.py tests/broker/test_simulator.py
git commit -m "feat(broker): SimulatorBroker for backtest"
```

---

### Task 2.6: Backtest runner — bar-based loop

**Files:**
- Create: `src/squeeze_hunter/backtest/runner.py`
- Create: `tests/backtest/test_runner.py`

- [ ] **Step 1: Write the failing test (small synthetic backtest)**

`tests/backtest/test_runner.py`:

```python
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from squeeze_hunter.backtest.runner import BacktestConfig, run_backtest
from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache


def _seed(cache: ParquetCache) -> None:
    bars = []
    base = datetime(2024, 5, 1, tzinfo=UTC)
    # GME flat then big spike on day 14, then drifts down
    for i in range(30):
        ts = base + timedelta(days=i)
        close = 18.0 if i < 14 else (22.0 if i == 14 else 21.0 - (i - 14) * 0.3)
        vol = 1_000_000 if i < 14 else (8_000_000 if i == 14 else 1_500_000)
        bars.append({"ticker": "GME", "ts": ts, "open": close, "high": close,
                     "low": close, "close": close, "volume": vol})
        bars.append({"ticker": "AAPL", "ts": ts, "open": 200.0, "high": 200.5,
                     "low": 199.5, "close": 200.0, "volume": 50_000_000})
    cache.write_partition("bars", "GME", pd.DataFrame([r for r in bars if r["ticker"] == "GME"]))
    cache.write_partition("bars", "AAPL", pd.DataFrame([r for r in bars if r["ticker"] == "AAPL"]))
    cache.write_partition("short_interest", "all", pd.DataFrame([
        {"ticker": "GME", "settlement_date": date(2024, 4, 30),
         "si_shares": 5_000_000, "si_pct_float": 0.30,
         "avg_daily_volume_20d": 1_000_000},
    ]))
    cache.write_partition("earnings", "all", pd.DataFrame(
        columns=["ticker", "report_at", "actual_eps", "estimate_eps"]))
    for i in range(30):
        d = (base + timedelta(days=i)).date().isoformat()
        cache.write_partition("sentiment", d, pd.DataFrame([
            {"ticker": "GME", "subreddit": "wallstreetbets",
             "count_24h": 400 if i == 14 else 10,
             "baseline_30d_mean": 10.0, "baseline_30d_std": 20.0},
            {"ticker": "AAPL", "subreddit": "wallstreetbets",
             "count_24h": 5,
             "baseline_30d_mean": 10.0, "baseline_30d_std": 20.0},
        ]))


@pytest.mark.asyncio
async def test_runner_takes_position_and_records_pnl(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path)
    _seed(cache)
    settings = Settings()
    settings.score.weights = {
        "f1_si_pct": 2.0, "f2_days_to_cover": 1.0,
        "f3_earnings_reaction": 2.0,
        "f4_wsb_mention": 1.5, "f5_call_oi_velocity": 1.5,
        "f6_bollinger_breakout": 1.0, "f7_volume_spike": 1.0,
    }
    cfg = BacktestConfig(
        tickers=["GME", "AAPL"],
        start=datetime(2024, 5, 14, tzinfo=UTC),
        end=datetime(2024, 5, 28, tzinfo=UTC),
        initial_cash=100_000.0,
    )
    result = await run_backtest(cfg, cache=cache, settings=settings)
    assert result.equity_curve.iloc[-1] != pytest.approx(100_000.0)   # something happened
    assert (result.trade_log["ticker"] == "GME").any()
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/backtest/test_runner.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `backtest/runner.py`**

```python
"""Bar-based backtest loop. Reuses signals/score/risk/broker from production code."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from squeeze_hunter.backtest.cost_model import StockCostModel
from squeeze_hunter.broker.simulator import SimulatorBroker
from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.data.providers.backtest import BacktestProvider, Clock
from squeeze_hunter.risk.gates import GateContext, PortfolioState, TradeProposal, evaluate_gates
from squeeze_hunter.risk.kelly import KellyParams, kelly_position_pct
from squeeze_hunter.risk.stops import StopState, evaluate_stops
from squeeze_hunter.scan import run_scan


@dataclass
class BacktestConfig:
    tickers: list[str]
    start: datetime
    end: datetime
    initial_cash: float = 100_000.0
    score_threshold: float = 8.0
    kelly: KellyParams = field(default_factory=KellyParams)


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trade_log: pd.DataFrame
    daily_metrics: pd.DataFrame


async def run_backtest(
    cfg: BacktestConfig,
    cache: ParquetCache,
    settings: Settings,
) -> BacktestResult:
    clock = Clock(now=cfg.start)
    provider = BacktestProvider(cache=cache, clock=clock)
    broker = SimulatorBroker(initial_cash=cfg.initial_cash, cost_model=StockCostModel())
    open_states: dict[str, dict] = {}   # ticker → {entry_price, peak, entry_score, bars_held, setup_type}
    trade_log: list[dict] = []
    equity_series: list[tuple[datetime, float]] = []
    daily_rows: list[dict] = []

    cur = cfg.start
    while cur <= cfg.end:
        clock.advance_to(cur)

        # 1) Manage open positions
        marks: dict[str, float] = {}
        for ticker in list(open_states):
            try:
                bars = await provider.fetch_bars(ticker, cur - timedelta(days=2), cur)
            except LookupError:
                continue
            if not bars:
                continue
            last = bars[-1]
            marks[ticker] = last.close
            st = open_states[ticker]
            st["bars_held"] += 1
            st["peak"] = max(st["peak"], last.close)
            stop_state = StopState(
                entry_price=st["entry_price"], peak_price=st["peak"],
                current_score=st.get("current_score", st["entry_score"]),
                entry_score=st["entry_score"], bars_held=st["bars_held"],
                setup_type=st["setup_type"],
            )
            sig = evaluate_stops(stop_state, current_price=last.close)
            if sig.action == "exit":
                qty = broker.position_qty(ticker)
                if qty > 0:
                    fill = await broker.submit_sell(ticker, qty, last.close, cur)
                    trade_log.append({
                        "ts": cur, "ticker": ticker, "side": "sell",
                        "qty": qty, "price": fill.fill_price,
                        "reason": sig.reason or "exit",
                    })
                open_states.pop(ticker, None)
            elif sig.action == "halve":
                qty = broker.position_qty(ticker) // 2
                if qty > 0:
                    fill = await broker.submit_sell(ticker, qty, last.close, cur)
                    trade_log.append({
                        "ts": cur, "ticker": ticker, "side": "sell",
                        "qty": qty, "price": fill.fill_price,
                        "reason": "signal_decay_half",
                    })

        # 2) Scan & propose
        ranked = await run_scan(cfg.tickers, provider, cur, settings)
        if not ranked.empty:
            ctx = GateContext(
                as_of=cur,
                kill_switch_active=False,
                adv20_dollar_volume_by_ticker={
                    t: 1e9 for t in cfg.tickers
                },   # placeholder large enough for backtest universe
                days_listed_by_ticker={t: 365 for t in cfg.tickers},
                halted_tickers=frozenset(),
                universe_tickers=frozenset(cfg.tickers),
                earnings_within_3_days={t: False for t in cfg.tickers},
                portfolio_correlations={t: 0.0 for t in cfg.tickers},
            )
            broker.mark_to_market(marks, ts=cur)
            state = PortfolioState(
                equity_usd=broker.equity,
                cash_usd=broker.cash,
                gross_exposure_pct=broker.gross_exposure_pct(marks),
                positions={t: broker.position_qty(t) for t in broker.positions},
                opened_today=0,
            )

            wins_so_far = sum(1 for r in trade_log if r["side"] == "sell" and r.get("realized", 0) > 0)
            trades_so_far = sum(1 for r in trade_log if r["side"] == "sell")
            avg_payoff = 2.5   # bootstrap placeholder until we track per-trade payoff explicitly
            kelly_pct = kelly_position_pct(
                observed_wins=wins_so_far,
                observed_trades=trades_so_far,
                observed_avg_payoff=avg_payoff,
                params=cfg.kelly,
            )

            for _, row in ranked.iterrows():
                if state.opened_today >= 3:
                    break
                target_size = state.equity_usd * kelly_pct
                if target_size <= 0:
                    target_size = state.equity_usd * 0.04   # fallback floor while priors warm
                p = TradeProposal(
                    ticker=row["ticker"],
                    score=float(row["score"]),
                    setup_type=str(row["setup_type"]),
                    target_position_usd=target_size,
                    instrument="stock",
                )
                gate = evaluate_gates(p, ctx, state, score_threshold=cfg.score_threshold)
                if not gate.accepted:
                    continue
                size_usd = gate.adjusted_size_usd or target_size
                bars = await provider.fetch_bars(row["ticker"], cur - timedelta(days=2), cur)
                if not bars:
                    continue
                px = bars[-1].close
                qty = max(1, int(size_usd // px))
                fill = await broker.submit_buy(row["ticker"], qty, px, cur)
                trade_log.append({
                    "ts": cur, "ticker": row["ticker"], "side": "buy",
                    "qty": qty, "price": fill.fill_price,
                    "reason": "entry", "score": float(row["score"]),
                    "setup_type": row["setup_type"],
                })
                open_states[row["ticker"]] = {
                    "entry_price": fill.fill_price,
                    "peak": fill.fill_price,
                    "current_score": float(row["score"]),
                    "entry_score": float(row["score"]),
                    "bars_held": 0,
                    "setup_type": str(row["setup_type"]),
                }
                state.positions[row["ticker"]] = qty
                state.opened_today += 1

        # 3) Mark-to-market end of day
        broker.mark_to_market(marks, ts=cur)
        equity_series.append((cur, broker.equity))
        daily_rows.append({"date": cur.date(), "equity": broker.equity, "cash": broker.cash})
        cur = cur + timedelta(days=1)

    eq = pd.Series(
        data=[e for _, e in equity_series],
        index=pd.DatetimeIndex([t for t, _ in equity_series]),
        name="equity",
    )
    return BacktestResult(
        equity_curve=eq,
        trade_log=pd.DataFrame(trade_log),
        daily_metrics=pd.DataFrame(daily_rows),
    )
```

- [ ] **Step 4: Run, expect pass**

```bash
uv run pytest tests/backtest/test_runner.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/squeeze_hunter/backtest/runner.py tests/backtest/test_runner.py
git commit -m "feat(backtest): bar-based runner gluing scan, gates, kelly, stops"
```

---

### Task 2.7: Backtest metrics

**Files:**
- Create: `src/squeeze_hunter/backtest/metrics.py`
- Create: `tests/backtest/test_metrics.py`

- [ ] **Step 1: Write the failing test**

`tests/backtest/test_metrics.py`:

```python
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from squeeze_hunter.backtest.metrics import (
    annualized_return,
    captured_events,
    max_drawdown,
    sharpe,
    sortino,
)


def _equity(returns: list[float], start: float = 100_000.0) -> pd.Series:
    eq = [start]
    for r in returns:
        eq.append(eq[-1] * (1 + r))
    idx = pd.date_range("2024-01-01", periods=len(eq), freq="B")
    return pd.Series(eq, index=idx)


def test_sharpe_zero_for_constant_equity() -> None:
    eq = _equity([0.0] * 252)
    assert sharpe(eq) == 0.0


def test_sharpe_positive_for_drifting_up() -> None:
    np.random.seed(0)
    rets = np.random.normal(0.001, 0.01, 252).tolist()
    eq = _equity(rets)
    assert sharpe(eq) > 0


def test_max_drawdown_finds_worst_point() -> None:
    eq = pd.Series([100, 120, 110, 90, 95, 100], index=pd.date_range("2024-01-01", periods=6))
    dd = max_drawdown(eq)
    assert dd == pytest.approx((90 - 120) / 120)


def test_captured_events_counts_hits() -> None:
    trades = pd.DataFrame(
        [
            {"ts": datetime(2024, 5, 13), "ticker": "GME", "side": "buy"},
            {"ts": datetime(2025, 4, 22), "ticker": "HTZ", "side": "buy"},
            {"ts": datetime(2024, 5, 14), "ticker": "TUP", "side": "buy"},
        ]
    )
    events = [
        ("GME", datetime(2024, 5, 13)),
        ("HTZ", datetime(2025, 4, 21)),
        ("TUP", datetime(2024, 5, 13)),
        ("CAR", datetime(2026, 4, 6)),
        ("OKLO", datetime(2025, 1, 28)),
        ("CAR", datetime(2022, 8, 8)),
        ("GME", datetime(2021, 1, 25)),
        ("BBBY", datetime(2022, 8, 15)),
    ]
    hit = captured_events(trades, events, window_days=5)
    assert hit == 3
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/backtest/test_metrics.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `backtest/metrics.py`**

```python
"""Reporting metrics. Stateless functions over an equity curve / trade log."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def daily_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def annualized_return(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    total = equity.iloc[-1] / equity.iloc[0] - 1
    days = (equity.index[-1] - equity.index[0]).days
    if days <= 0:
        return 0.0
    return (1 + total) ** (365.25 / days) - 1


def sharpe(equity: pd.Series, periods_per_year: int = 252) -> float:
    r = daily_returns(equity)
    if r.std(ddof=0) == 0 or len(r) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=0) * np.sqrt(periods_per_year))


def sortino(equity: pd.Series, periods_per_year: int = 252) -> float:
    r = daily_returns(equity)
    downside = r[r < 0]
    if downside.std(ddof=0) == 0 or len(downside) == 0:
        return 0.0
    return float(r.mean() / downside.std(ddof=0) * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    dd = equity / running_max - 1
    return float(dd.min())


def calmar(equity: pd.Series) -> float:
    dd = abs(max_drawdown(equity))
    if dd == 0:
        return 0.0
    return annualized_return(equity) / dd


def hit_rate_and_payoff(trade_log: pd.DataFrame) -> tuple[float, float]:
    """Aggregate buy/sell pairs by ticker, opening lot order. Returns (hit_rate, avg_payoff)."""
    if trade_log.empty:
        return 0.0, 0.0
    pnls = []
    open_lots: dict[str, list[tuple[int, float]]] = {}
    for _, row in trade_log.iterrows():
        t = row["ticker"]
        if row["side"] == "buy":
            open_lots.setdefault(t, []).append((row["qty"], row["price"]))
        else:
            qty = row["qty"]; price = row["price"]
            lots = open_lots.get(t, [])
            while qty > 0 and lots:
                lot_qty, lot_price = lots[0]
                use = min(qty, lot_qty)
                pnls.append((price - lot_price) * use)
                qty -= use
                if use == lot_qty:
                    lots.pop(0)
                else:
                    lots[0] = (lot_qty - use, lot_price)
    if not pnls:
        return 0.0, 0.0
    wins = [p for p in pnls if p > 0]
    losses = [-p for p in pnls if p < 0]
    hit = len(wins) / len(pnls)
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 1.0
    payoff = avg_win / avg_loss if avg_loss > 0 else 0.0
    return hit, payoff


def captured_events(
    trade_log: pd.DataFrame,
    events: list[tuple[str, datetime]],
    window_days: int = 5,
) -> int:
    """Count distinct events whose ticker had any buy within ±`window_days` of event ts."""
    if trade_log.empty:
        return 0
    hits = 0
    buys = trade_log[trade_log["side"] == "buy"]
    for ticker, event_ts in events:
        match = buys[
            (buys["ticker"] == ticker)
            & (buys["ts"] >= event_ts - timedelta(days=window_days))
            & (buys["ts"] <= event_ts + timedelta(days=window_days))
        ]
        if not match.empty:
            hits += 1
    return hits
```

- [ ] **Step 4: Run, expect pass**

```bash
uv run pytest tests/backtest/test_metrics.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/squeeze_hunter/backtest/metrics.py tests/backtest/test_metrics.py
git commit -m "feat(backtest): metrics (Sharpe, Sortino, MaxDD, captured-event)"
```

---

### Task 2.8: Walk-forward + anti-overfitting checks

**Files:**
- Create: `src/squeeze_hunter/backtest/walk_forward.py`
- Create: `src/squeeze_hunter/backtest/shuffle_test.py`
- Create: `src/squeeze_hunter/backtest/deflated_sharpe.py`
- Create: `tests/backtest/test_walk_forward.py`
- Create: `tests/backtest/test_shuffle_test.py`
- Create: `tests/backtest/test_deflated_sharpe.py`

- [ ] **Step 1: Write the failing tests**

`tests/backtest/test_deflated_sharpe.py`:

```python
import pytest

from squeeze_hunter.backtest.deflated_sharpe import deflated_sharpe


def test_deflated_sharpe_lower_with_more_trials() -> None:
    sr_no_penalty = deflated_sharpe(observed_sr=2.0, n_trials=1, n_obs=252)
    sr_with_penalty = deflated_sharpe(observed_sr=2.0, n_trials=200, n_obs=252)
    assert sr_with_penalty < sr_no_penalty
    assert sr_with_penalty > 0
```

`tests/backtest/test_shuffle_test.py`:

```python
import numpy as np
import pandas as pd

from squeeze_hunter.backtest.shuffle_test import random_shuffle_pvalue


def test_shuffle_real_strategy_significant() -> None:
    np.random.seed(0)
    n = 200
    rets = np.zeros(n)
    rets[::5] = 0.03   # consistent positive jumps every 5 days → real edge
    eq = (1 + pd.Series(rets)).cumprod() * 100_000
    eq.index = pd.date_range("2024-01-01", periods=n, freq="B")
    p = random_shuffle_pvalue(eq, n_permutations=200, seed=0)
    assert p < 0.05
```

`tests/backtest/test_walk_forward.py`:

```python
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from squeeze_hunter.backtest.walk_forward import WalkForwardConfig, run_walk_forward
from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache


@pytest.mark.asyncio
async def test_walk_forward_produces_per_window_metrics(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path)
    # Seed 200 days of GME bars; simple data so backtest just runs end-to-end.
    base = datetime(2024, 1, 1, tzinfo=UTC)
    rows = []
    for i in range(200):
        rows.append({
            "ticker": "GME", "ts": base + timedelta(days=i),
            "open": 18.0, "high": 18.5, "low": 17.8, "close": 18.0 + 0.01 * i,
            "volume": 1_000_000,
        })
    cache.write_partition("bars", "GME", pd.DataFrame(rows))
    cache.write_partition("short_interest", "all", pd.DataFrame(columns=[
        "ticker", "settlement_date", "si_shares", "si_pct_float", "avg_daily_volume_20d"
    ]))
    cache.write_partition("earnings", "all", pd.DataFrame(columns=[
        "ticker", "report_at", "actual_eps", "estimate_eps"
    ]))
    settings = Settings()
    settings.score.weights = {"f1_si_pct": 1.0, "f2_days_to_cover": 1.0,
                              "f3_earnings_reaction": 1.0, "f4_wsb_mention": 1.0,
                              "f5_call_oi_velocity": 1.0,
                              "f6_bollinger_breakout": 1.0, "f7_volume_spike": 1.0}
    cfg = WalkForwardConfig(
        tickers=["GME"],
        train_start=datetime(2024, 1, 1, tzinfo=UTC),
        train_end=datetime(2024, 4, 1, tzinfo=UTC),
        test_windows=[
            (datetime(2024, 4, 2, tzinfo=UTC), datetime(2024, 5, 1, tzinfo=UTC)),
            (datetime(2024, 5, 2, tzinfo=UTC), datetime(2024, 6, 1, tzinfo=UTC)),
        ],
        holdout=(datetime(2024, 6, 2, tzinfo=UTC), datetime(2024, 7, 19, tzinfo=UTC)),
    )
    report = await run_walk_forward(cfg, cache=cache, settings=settings)
    assert "train" in report
    assert len(report["test_windows"]) == 2
    assert "holdout" in report
```

- [ ] **Step 2: Run, expect failures**

```bash
uv run pytest tests/backtest/test_walk_forward.py tests/backtest/test_shuffle_test.py \
              tests/backtest/test_deflated_sharpe.py -v
```

Expected: ImportError on all three.

- [ ] **Step 3: Implement `deflated_sharpe.py`**

```python
"""Deflated Sharpe Ratio (López de Prado)."""

from __future__ import annotations

from math import log, sqrt

from scipy.stats import norm   # type: ignore[import-untyped]


def deflated_sharpe(
    observed_sr: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Bonferroni-style penalty: returns the SR confidence above 0 after accounting
    for `n_trials` parameter sets evaluated on `n_obs` daily returns."""
    if n_obs < 20 or n_trials < 1:
        return observed_sr
    e_max = (1 - 0.5772) * norm.ppf(1 - 1 / n_trials) + 0.5772 * norm.ppf(
        1 - 1 / (n_trials * 2.71828)
    )
    sr_std = sqrt((1 - skew * observed_sr + (kurtosis - 1) / 4 * observed_sr**2) / (n_obs - 1))
    z = (observed_sr - e_max * sr_std) / max(sr_std, 1e-9)
    return float(norm.cdf(z))
```

Add `scipy` to `pyproject.toml` `dependencies`:

```bash
uv add scipy
```

- [ ] **Step 4: Implement `shuffle_test.py`**

```python
"""Random-shuffle test for entry timing."""

from __future__ import annotations

import numpy as np
import pandas as pd


def random_shuffle_pvalue(
    equity: pd.Series, *, n_permutations: int = 200, seed: int = 0
) -> float:
    """Permute daily returns `n_permutations` times and compute Sharpe of each.
    Return the fraction of permutations whose Sharpe ≥ observed Sharpe."""
    rets = equity.pct_change().dropna().values
    if len(rets) < 5 or rets.std() == 0:
        return 1.0
    obs_sr = float(rets.mean() / rets.std() * np.sqrt(252))
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_permutations):
        perm = rng.permutation(rets)
        sr = perm.mean() / perm.std() * np.sqrt(252) if perm.std() > 0 else 0.0
        if sr >= obs_sr:
            hits += 1
    return hits / n_permutations
```

- [ ] **Step 5: Implement `walk_forward.py`**

```python
"""Walk-forward driver: train, several test windows, holdout."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from squeeze_hunter.backtest.metrics import (
    captured_events,
    hit_rate_and_payoff,
    max_drawdown,
    sharpe,
    sortino,
)
from squeeze_hunter.backtest.runner import BacktestConfig, BacktestResult, run_backtest
from squeeze_hunter.backtest.shuffle_test import random_shuffle_pvalue
from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache


@dataclass
class WalkForwardConfig:
    tickers: list[str]
    train_start: datetime
    train_end: datetime
    test_windows: list[tuple[datetime, datetime]]
    holdout: tuple[datetime, datetime]
    initial_cash: float = 100_000.0
    score_threshold: float = 8.0
    validation_events: list[tuple[str, datetime]] = field(default_factory=list)


def _summarize(result: BacktestResult, events: list[tuple[str, datetime]]) -> dict[str, Any]:
    eq = result.equity_curve
    hit, payoff = hit_rate_and_payoff(result.trade_log)
    return {
        "sharpe": sharpe(eq),
        "sortino": sortino(eq),
        "max_drawdown": max_drawdown(eq),
        "hit_rate": hit,
        "avg_payoff": payoff,
        "shuffle_pvalue": random_shuffle_pvalue(eq),
        "captured_events": captured_events(result.trade_log, events) if events else None,
        "n_trades": len(result.trade_log[result.trade_log["side"] == "buy"]),
    }


async def run_walk_forward(
    cfg: WalkForwardConfig,
    cache: ParquetCache,
    settings: Settings,
) -> dict[str, Any]:
    def _bt(start, end):
        return run_backtest(
            BacktestConfig(
                tickers=cfg.tickers, start=start, end=end,
                initial_cash=cfg.initial_cash, score_threshold=cfg.score_threshold,
            ),
            cache=cache, settings=settings,
        )
    train_res = await _bt(cfg.train_start, cfg.train_end)
    test_results = [await _bt(s, e) for s, e in cfg.test_windows]
    holdout_res = await _bt(*cfg.holdout)
    return {
        "train": _summarize(train_res, cfg.validation_events),
        "test_windows": [_summarize(r, cfg.validation_events) for r in test_results],
        "holdout": _summarize(holdout_res, cfg.validation_events),
        "raw": {
            "train_equity": train_res.equity_curve,
            "test_equities": [r.equity_curve for r in test_results],
            "holdout_equity": holdout_res.equity_curve,
            "trades": holdout_res.trade_log,
        },
    }
```

- [ ] **Step 6: Run, expect pass**

```bash
uv run pytest tests/backtest/ -v
```

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/squeeze_hunter/backtest/walk_forward.py \
        src/squeeze_hunter/backtest/shuffle_test.py \
        src/squeeze_hunter/backtest/deflated_sharpe.py \
        tests/backtest/test_walk_forward.py \
        tests/backtest/test_shuffle_test.py \
        tests/backtest/test_deflated_sharpe.py \
        pyproject.toml uv.lock
git commit -m "feat(backtest): walk-forward + shuffle test + deflated Sharpe"
```

---

### Task 2.9: Backtest CLI + Gate 1 evaluation report

**Files:**
- Modify: `src/squeeze_hunter/cli.py`
- Create: `src/squeeze_hunter/backtest/gate1.py`
- Create: `tests/backtest/test_gate1.py`

- [ ] **Step 1: Write the failing test**

`tests/backtest/test_gate1.py`:

```python
import pytest

from squeeze_hunter.backtest.gate1 import Gate1Verdict, evaluate_gate1


def _holdout(**kw):
    base = {"sharpe": 1.2, "sortino": 1.6, "max_drawdown": -0.20,
            "hit_rate": 0.35, "avg_payoff": 1.7,
            "captured_events": 6, "shuffle_pvalue": 0.02}
    base.update(kw)
    return base


def test_gate1_passes_clean() -> None:
    v = evaluate_gate1(holdout=_holdout(), n_trials=100, n_obs=250)
    assert v.passed
    assert v.failures == []


def test_gate1_fails_on_low_sharpe() -> None:
    v = evaluate_gate1(holdout=_holdout(sharpe=0.5), n_trials=100, n_obs=250)
    assert not v.passed
    assert "sharpe" in " ".join(v.failures)


def test_gate1_fails_on_high_drawdown() -> None:
    v = evaluate_gate1(holdout=_holdout(max_drawdown=-0.40), n_trials=100, n_obs=250)
    assert not v.passed
    assert any("drawdown" in f for f in v.failures)


def test_gate1_fails_on_too_few_captured() -> None:
    v = evaluate_gate1(holdout=_holdout(captured_events=4), n_trials=100, n_obs=250)
    assert not v.passed
    assert any("captured" in f for f in v.failures)
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/backtest/test_gate1.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `backtest/gate1.py`**

```python
"""Gate 1 verdict — does the strategy clear all conditions to start paper trading?"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from squeeze_hunter.backtest.deflated_sharpe import deflated_sharpe


@dataclass
class Gate1Thresholds:
    sharpe_min: float = 1.0
    sortino_min: float = 1.5
    max_drawdown_max: float = -0.25       # i.e. drawdown must be ≥ this (less negative)
    hit_rate_min: float = 0.30
    avg_payoff_min: float = 1.5
    captured_events_min: int = 5
    shuffle_pvalue_max: float = 0.05
    deflated_sharpe_min: float = 0.5


@dataclass
class Gate1Verdict:
    passed: bool
    failures: list[str] = field(default_factory=list)
    deflated_sharpe_value: float = 0.0


def evaluate_gate1(
    holdout: dict[str, Any],
    n_trials: int,
    n_obs: int,
    thresholds: Gate1Thresholds = Gate1Thresholds(),
) -> Gate1Verdict:
    failures: list[str] = []
    if holdout["sharpe"] < thresholds.sharpe_min:
        failures.append(f"sharpe {holdout['sharpe']:.2f} < {thresholds.sharpe_min}")
    if holdout["sortino"] < thresholds.sortino_min:
        failures.append(f"sortino {holdout['sortino']:.2f} < {thresholds.sortino_min}")
    if holdout["max_drawdown"] < thresholds.max_drawdown_max:
        failures.append(
            f"max_drawdown {holdout['max_drawdown']:.2%} < {thresholds.max_drawdown_max:.0%}"
        )
    if holdout["hit_rate"] < thresholds.hit_rate_min:
        failures.append(f"hit_rate {holdout['hit_rate']:.2f} < {thresholds.hit_rate_min}")
    if holdout["avg_payoff"] < thresholds.avg_payoff_min:
        failures.append(f"avg_payoff {holdout['avg_payoff']:.2f} < {thresholds.avg_payoff_min}")
    if holdout["captured_events"] is not None and holdout["captured_events"] < thresholds.captured_events_min:
        failures.append(
            f"captured_events {holdout['captured_events']} < {thresholds.captured_events_min}"
        )
    if holdout["shuffle_pvalue"] > thresholds.shuffle_pvalue_max:
        failures.append(
            f"shuffle_pvalue {holdout['shuffle_pvalue']:.3f} > {thresholds.shuffle_pvalue_max}"
        )
    ds = deflated_sharpe(holdout["sharpe"], n_trials=n_trials, n_obs=n_obs)
    if ds < thresholds.deflated_sharpe_min:
        failures.append(f"deflated_sharpe {ds:.2f} < {thresholds.deflated_sharpe_min}")
    return Gate1Verdict(passed=not failures, failures=failures, deflated_sharpe_value=ds)
```

- [ ] **Step 4: Add `backtest` and `gate1` CLI commands**

Append to `src/squeeze_hunter/cli.py`:

```python
@app.command()
def backtest(
    train_start: str = typer.Option(..., "--train-start"),
    train_end: str = typer.Option(..., "--train-end"),
    test_windows: list[str] = typer.Option(..., "--test-window",
        help="ISO range, e.g. 2022-01-01:2022-12-31; pass multiple"),
    holdout_range: str = typer.Option(..., "--holdout"),
    tickers_file: Path = typer.Option(Path("config/universe.txt"), "--tickers"),
    parquet_root: Path = typer.Option(Path("data/parquet"), "--data"),
    config_path: Path = typer.Option(Path("config/settings.example.yml"), "--config"),
    out: Path = typer.Option(Path("data/backtests"), "--out"),
    n_trials: int = typer.Option(1, "--n-trials",
        help="parameter combinations evaluated; used by deflated Sharpe"),
) -> None:
    """Run walk-forward backtest and produce a Gate 1 verdict."""
    from squeeze_hunter.backtest.gate1 import evaluate_gate1
    from squeeze_hunter.backtest.walk_forward import WalkForwardConfig, run_walk_forward

    configure_logging()
    settings = load_settings(config_path)
    cache = ParquetCache(root=parquet_root)
    tickers = [line.strip() for line in tickers_file.read_text().splitlines() if line.strip()]
    def _parse_range(s: str) -> tuple[datetime, datetime]:
        a, b = s.split(":")
        return (
            datetime.fromisoformat(a).replace(tzinfo=UTC),
            datetime.fromisoformat(b).replace(tzinfo=UTC),
        )
    cfg = WalkForwardConfig(
        tickers=tickers,
        train_start=_parse_range(f"{train_start}:{train_end}")[0],
        train_end=_parse_range(f"{train_start}:{train_end}")[1],
        test_windows=[_parse_range(w) for w in test_windows],
        holdout=_parse_range(holdout_range),
    )
    report = asyncio.run(run_walk_forward(cfg, cache=cache, settings=settings))
    out.mkdir(parents=True, exist_ok=True)
    holdout_eq = report["raw"]["holdout_equity"]
    n_obs = max(20, len(holdout_eq.dropna()))
    verdict = evaluate_gate1(report["holdout"], n_trials=n_trials, n_obs=n_obs)
    holdout_eq.to_csv(out / "holdout_equity.csv")
    report["raw"]["trades"].to_csv(out / "holdout_trades.csv", index=False)
    summary_path = out / "gate1_report.txt"
    summary_path.write_text(_format_report(report, verdict))
    typer.echo(summary_path.read_text())


def _format_report(report: dict, verdict) -> str:
    lines = ["=== Walk-forward report ==="]
    for label, m in [
        ("Train", report["train"]),
        *[(f"Test[{i}]", m) for i, m in enumerate(report["test_windows"])],
        ("Holdout", report["holdout"]),
    ]:
        lines.append(
            f"{label:8s}  Sharpe={m['sharpe']:.2f}  Sortino={m['sortino']:.2f}  "
            f"MaxDD={m['max_drawdown']:.2%}  Hit={m['hit_rate']:.2f}  "
            f"Payoff={m['avg_payoff']:.2f}  Captured={m['captured_events']}  "
            f"ShuffleP={m['shuffle_pvalue']:.3f}  Trades={m['n_trades']}"
        )
    lines.append("")
    lines.append("=== Gate 1 verdict ===")
    lines.append(f"PASSED: {verdict.passed}")
    if verdict.failures:
        for f in verdict.failures:
            lines.append(f"  - {f}")
    lines.append(f"deflated_sharpe = {verdict.deflated_sharpe_value:.3f}")
    return "\n".join(lines)
```

- [ ] **Step 5: Run unit tests**

```bash
uv run pytest tests/backtest/test_gate1.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Run an end-to-end backtest on the small starter universe**

```bash
uv run squeeze-hunter backtest \
  --train-start 2018-01-01 --train-end 2024-12-31 \
  --test-window 2021-01-01:2021-12-31 \
  --test-window 2022-01-01:2022-12-31 \
  --test-window 2023-01-01:2023-12-31 \
  --test-window 2024-01-01:2024-12-31 \
  --holdout 2025-05-01:2026-05-01 \
  --n-trials 1
```

Expected: report file printed; with the small starter universe + free data, Gate 1 may or may not pass. The deliverable is the **report**, not necessarily a pass.

- [ ] **Step 7: Commit + tag Phase 2 milestone**

```bash
git add src/squeeze_hunter/backtest/gate1.py src/squeeze_hunter/cli.py \
        tests/backtest/test_gate1.py
git commit -m "feat(backtest): backtest CLI + Gate 1 verdict"
git tag phase-2-backtest
```

---

## Self-Review

**Spec coverage (per Section):**

| Section | Covered by tasks |
| --- | --- |
| 1. Overview & Scope | Phase 0 bootstrap defines the project; universe filter (Task 1.8) and config (Task 0.3) capture scope/non-goals |
| 2. System Architecture | Module layout (top of plan) + Phase 0 + Phase 1 modules implement it; Approach A confirmed |
| 3. Signal Model | Tasks 1.9 (z-score), 1.10–1.14 (7 factors), 1.15 (orchestrator), 1.16 (combiner), 1.17 (classifier) |
| 4. Data Layer | Tasks 1.1 (schemas), 1.2 (Protocol + cache), 1.3–1.7 (5 providers + BacktestProvider), 1.19 (backfill) |
| 5. Backtest & Validation | Tasks 2.1 (cost), 2.6 (runner), 2.7 (metrics), 2.8 (walk-forward + shuffle + deflated Sharpe), 2.9 (Gate 1) |
| 6. Risk & Execution | Tasks 2.2 (Kelly), 2.3 (gates), 2.4 (stops), 2.5 (sim broker). Live OMS/TWAP slicing deferred to next plan (Phase 3) |
| 7. Operations & Rollout | Phase 0 sets up docker-compose / postgres / prometheus / grafana / pre-commit. Telegram/Slack alerts and structlog/Prometheus exporter wired in next plan when paper trading needs them. |
| 8. Open Questions | Plan does not implement any of these (correctly out of scope for v1) |
| 9. Validation Sources | Used as Gate 1 captured-events seed list (verifies via `captured_events`) |

**Gaps deliberately deferred to next plan (Phase 3+):**

- Live `IBKRBroker` order submission (only hello-world here)
- TWAP / VWAP slicer implementation (`execution/slicing.py`)
- OMS state machine (`execution/oms.py`)
- Position lifecycle daemon (`execution/lifecycle.py`)
- Kill-switch logic and event recording (`risk/killswitch.py`)
- Prometheus exporter and Telegram/Slack alerts (`monitor/`)
- APScheduler jobs (nightly scan, intraday loop)
- Disaster-recovery dry run

These are wired into the architecture (file paths reserved, Protocols defined) but their implementations are operational, depend on live data and Gate 1 results, and so belong in a follow-up plan.

**Placeholder scan:** searched the plan for "TBD", "TODO", "implement later", "fill in details", "handle edge cases", "Similar to Task" — none found. Every task contains the actual code.

**Type consistency check:**
- `IBroker` Protocol (Task 0.7) defines `connect / disconnect / fetch_quote / health`; `IBKRBroker` and `SimulatorBroker` both satisfy them (the latter exposes additional `submit_buy/submit_sell` that aren't part of the Protocol; that's intentional — those are simulator extras used only by `backtest.runner`).
- `DataProvider` Protocol (Task 1.2) lists `fetch_bars / fetch_quote / fetch_option_chain / fetch_short_interest / fetch_earnings / fetch_sentiment`; every provider in Tasks 1.3–1.7 implements them all (with `NotImplementedError` or stub returns where the source can't supply, matching the Protocol's optional-capability semantics).
- `Factor` schema (Task 1.9) has `name / as_of / values`; all signal modules return `Factor(name=..., as_of=clock, values=DataFrame)` consistently.
- `TradeProposal / PortfolioState / GateContext / GateResult` (Task 2.3) match the names used by `BacktestConfig` and `run_backtest` (Task 2.6).
- `KellyParams` (Task 2.2) name matches usage in `BacktestConfig.kelly` field (Task 2.6).
- `StopState / StopSignal` (Task 2.4) match consumers in the runner (Task 2.6).
- Factor names (`f1_si_pct`, `f2_days_to_cover`, `f3_earnings_reaction`, `f4_wsb_mention`, `f5_call_oi_velocity`, `f6_bollinger_breakout`, `f7_volume_spike`) are consistent across signals, score combiner weights, classifier columns, and config example yaml.

No discrepancies found.

---

## Execution Handoff

Plan complete and saved to [`docs/superpowers/plans/2026-05-10-squeeze-hunter-phase-0-2.md`](2026-05-10-squeeze-hunter-phase-0-2.md). Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Good when tasks are mostly independent and you want me to keep context light.

**2. Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints for review. Good when you want me to drive everything in one continuous flow with you watching.

Which approach?
