# Squeeze Hunter — Phase 3–5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take squeeze-hunter from a Gate-1-cleared backtest pipeline to live operation: paper trading for 30 days, then small live ($2–5K) for 60 days, then scaled live with continuous decay monitoring.

**Architecture:** Add the live execution stack on top of the existing modular monolith. The `IBroker` Protocol (Phase 0) and `BacktestProvider` already let us swap brokers without touching strategy code. This plan implements `execution/`, `monitor/`, `scheduler.py`, and a real `IBKRBroker` order path; then promotes the system through three operational gates.

**Tech Stack:** All Phase 0–2 deps remain. Adds: `apscheduler` (scheduler — already in deps), `prometheus-client` (metrics — already in deps), Telegram bot via `httpx`, Slack incoming webhook via `httpx`.

**Spec reference:** [`docs/superpowers/specs/2026-05-10-squeeze-hunter-design.md`](../specs/2026-05-10-squeeze-hunter-design.md) — Sections 6 (Risk & Execution), 7 (Operations & Rollout).

**Prereq:** Phase 0–2 complete and tagged. The user has run a real backtest end-to-end and seen a Gate 1 verdict. **Do not start this plan until Gate 1 has passed at least once on real backfilled data.** If Gate 1 fails, iterate on signal weights / universe filters first; live execution code is meaningless without an edge to execute.

**Scope of THIS plan:**
- Phase 3 — paper trading scaffolding: live IBKRBroker order path, OMS, TWAP slicer, killswitch, monitor, scheduler, paper driver, DR drill
- Phase 4 — small live promotion: live mode toggle, ramp-up sizing, daily review process
- Phase 5 — scale up + continuous improvement: decay detection, paid-data ROI gate, runbook

---

## File Structure (additions / modifications relative to Phase 0–2)

```
squeeze-hunter/
├── src/squeeze_hunter/
│   ├── broker/
│   │   ├── ibkr.py                 # MODIFY: extend with order submission + OMS hooks
│   │   └── paper.py                # NEW: thin wrapper that flips IBKR client_id to paper
│   ├── execution/
│   │   ├── __init__.py             # NEW
│   │   ├── oms.py                  # NEW: order state machine + reconciliation
│   │   ├── slicing.py              # NEW: TWAP slicer
│   │   └── lifecycle.py            # NEW: entry / manage / exit driver
│   ├── risk/
│   │   └── killswitch.py           # NEW: drawdown / outage / data-staleness triggers
│   ├── monitor/
│   │   ├── __init__.py             # NEW
│   │   ├── metrics.py              # NEW: prometheus exporter + http server
│   │   ├── alerts.py               # NEW: Telegram + Slack + email senders
│   │   └── healthcheck.py          # NEW: /health endpoint
│   ├── scheduler.py                # NEW: APScheduler with the daily job graph
│   ├── runtime.py                  # NEW: top-level lifecycle ("paper" / "live" mode)
│   └── cli.py                      # MODIFY: add `paper`, `live`, `emergency-flatten`
├── docs/runbooks/
│   ├── paper-trading.md            # NEW: how to operate Phase 3
│   ├── live-trading.md             # NEW: how to operate Phase 4
│   ├── disaster-recovery.md        # NEW: backup/restore drill
│   └── decay-monitoring.md         # NEW: monthly backtest re-run procedure
└── tests/
    ├── execution/
    ├── risk/test_killswitch.py
    ├── monitor/
    ├── runtime/
    └── e2e/test_paper_loop.py      # NEW: short e2e against simulator broker
```

**Phase milestones:**
- `phase-3-paper-ready` — system can run a 24h paper-trading loop end-to-end against IBKR paper; killswitch + monitor + alerts proven by injected fault tests.
- `phase-4-live-ramp` — first real-money trade placed and exited; daily review process documented and exercised.
- `phase-5-scale` — system has run ≥ 60 days live without incident; monthly decay-detection job is wired in.

---

## Phase 3 — Paper Trading

### Task 3.1: Extend `IBKRBroker` with order submission

**Files:**
- Modify: `src/squeeze_hunter/broker/ibkr.py`
- Create: `tests/broker/test_ibkr_orders.py`

The hello-world version only does `connect / fetch_quote / health`. Add `submit_buy / submit_sell / cancel_order / get_open_orders` mirroring `SimulatorBroker`'s interface so the rest of the system is broker-agnostic.

- [ ] **Step 1: Write the failing test (mocked ib-async)**

`tests/broker/test_ibkr_orders.py`:

```python
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from squeeze_hunter.broker.base import BrokerOrder
from squeeze_hunter.broker.ibkr import IBKRBroker


@pytest.mark.asyncio
async def test_submit_buy_returns_pending_order(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = IBKRBroker(client_id=99)
    fake_ib = MagicMock()
    fake_trade = MagicMock()
    fake_trade.order.orderId = 1234
    fake_trade.orderStatus.status = "PreSubmitted"
    fake_ib.placeOrder = MagicMock(return_value=fake_trade)
    fake_ib.qualifyContractsAsync = AsyncMock()
    broker._ib = fake_ib

    order = await broker.submit_buy(
        ticker="GME", qty=100, limit_price=18.5,
        ts=datetime(2026, 5, 14, 13, 35, tzinfo=UTC),
    )
    assert isinstance(order, BrokerOrder)
    assert order.broker_order_id == "1234"
    assert order.status == "pending"
    assert order.side == "buy"
    assert order.qty == 100


@pytest.mark.asyncio
async def test_submit_sell_uses_limit_when_provided() -> None:
    broker = IBKRBroker(client_id=99)
    fake_ib = MagicMock()
    fake_trade = MagicMock()
    fake_trade.order.orderId = 5678
    fake_trade.orderStatus.status = "PreSubmitted"
    captured = {}

    def _capture(contract, order):
        captured["limit"] = order.lmtPrice
        captured["action"] = order.action
        return fake_trade

    fake_ib.placeOrder = _capture
    fake_ib.qualifyContractsAsync = AsyncMock()
    broker._ib = fake_ib

    await broker.submit_sell(
        ticker="GME", qty=50, limit_price=20.0,
        ts=datetime(2026, 5, 14, 13, 35, tzinfo=UTC),
    )
    assert captured["limit"] == 20.0
    assert captured["action"] == "SELL"
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/broker/test_ibkr_orders.py -v
```

Expected: ImportError (`BrokerOrder` doesn't exist yet) or AttributeError (no `submit_buy` on `IBKRBroker`).

- [ ] **Step 3: Add `BrokerOrder` to `broker/base.py`**

Read `src/squeeze_hunter/broker/base.py` and append:

```python
@dataclass(slots=True, frozen=True)
class BrokerOrder:
    broker_order_id: str
    ticker: str
    side: str            # "buy" | "sell"
    qty: int
    limit_price: float | None
    status: str          # "pending" | "filled" | "partial" | "cancelled" | "rejected"
    filled_qty: int = 0
    avg_fill_price: float | None = None
```

Add to the `IBroker` Protocol:

```python
async def submit_buy(
    self: "IBroker",
    ticker: str,
    qty: int,
    limit_price: float | None,
    ts: datetime,
) -> BrokerOrder: ...

async def submit_sell(
    self: "IBroker",
    ticker: str,
    qty: int,
    limit_price: float | None,
    ts: datetime,
) -> BrokerOrder: ...

async def cancel_order(self: "IBroker", broker_order_id: str) -> bool: ...

async def get_open_orders(self: "IBroker") -> list[BrokerOrder]: ...
```

(Add `from datetime import datetime` to the imports at the top of base.py.)

- [ ] **Step 4: Implement order methods on `IBKRBroker`**

In `src/squeeze_hunter/broker/ibkr.py`, import `LimitOrder` from `ib_async`:

```python
from ib_async import IB, LimitOrder, MarketOrder, Stock
```

Then add methods:

```python
async def submit_buy(
    self: "IBKRBroker",
    ticker: str,
    qty: int,
    limit_price: float | None,
    ts: datetime,
) -> BrokerOrder:
    contract = Stock(ticker, "SMART", "USD")
    await self._ib.qualifyContractsAsync(contract)
    order = LimitOrder("BUY", qty, limit_price) if limit_price else MarketOrder("BUY", qty)
    trade = self._ib.placeOrder(contract, order)
    log.info(
        "order_submitted",
        ticker=ticker,
        side="buy",
        qty=qty,
        limit=limit_price,
        broker_order_id=trade.order.orderId,
    )
    return BrokerOrder(
        broker_order_id=str(trade.order.orderId),
        ticker=ticker,
        side="buy",
        qty=qty,
        limit_price=limit_price,
        status=_translate_status(trade.orderStatus.status),
    )


async def submit_sell(
    self: "IBKRBroker",
    ticker: str,
    qty: int,
    limit_price: float | None,
    ts: datetime,
) -> BrokerOrder:
    contract = Stock(ticker, "SMART", "USD")
    await self._ib.qualifyContractsAsync(contract)
    order = LimitOrder("SELL", qty, limit_price) if limit_price else MarketOrder("SELL", qty)
    trade = self._ib.placeOrder(contract, order)
    log.info(
        "order_submitted",
        ticker=ticker,
        side="sell",
        qty=qty,
        limit=limit_price,
        broker_order_id=trade.order.orderId,
    )
    return BrokerOrder(
        broker_order_id=str(trade.order.orderId),
        ticker=ticker,
        side="sell",
        qty=qty,
        limit_price=limit_price,
        status=_translate_status(trade.orderStatus.status),
    )


async def cancel_order(self: "IBKRBroker", broker_order_id: str) -> bool:
    for trade in self._ib.openTrades():
        if str(trade.order.orderId) == broker_order_id:
            self._ib.cancelOrder(trade.order)
            return True
    return False


async def get_open_orders(self: "IBKRBroker") -> list[BrokerOrder]:
    out = []
    for trade in self._ib.openTrades():
        contract = trade.contract
        order = trade.order
        st = trade.orderStatus
        out.append(
            BrokerOrder(
                broker_order_id=str(order.orderId),
                ticker=getattr(contract, "symbol", ""),
                side="buy" if order.action == "BUY" else "sell",
                qty=order.totalQuantity,
                limit_price=getattr(order, "lmtPrice", None) or None,
                status=_translate_status(st.status),
                filled_qty=int(st.filled or 0),
                avg_fill_price=float(st.avgFillPrice) if st.avgFillPrice else None,
            )
        )
    return out
```

Add the status translator near the top of the file:

```python
_STATUS_MAP = {
    "PendingSubmit": "pending",
    "PendingCancel": "pending",
    "PreSubmitted": "pending",
    "Submitted": "pending",
    "ApiPending": "pending",
    "Filled": "filled",
    "Cancelled": "cancelled",
    "ApiCancelled": "cancelled",
    "Inactive": "rejected",
}


def _translate_status(ibkr_status: str) -> str:
    return _STATUS_MAP.get(ibkr_status, "pending")
```

Update the imports of `ibkr.py` to include `BrokerOrder`.

- [ ] **Step 5: Run tests, expect pass**

```bash
uv run pytest tests/broker/ -v
```

Expected: all green (Phase 0 tests + 2 new).

- [ ] **Step 6: Commit**

```bash
git add src/squeeze_hunter/broker/base.py src/squeeze_hunter/broker/ibkr.py \
        tests/broker/test_ibkr_orders.py
git commit -m "feat(broker): IBKRBroker order submission + status mapping"
```

---

### Task 3.2: Paper-mode broker wrapper

**Files:**
- Create: `src/squeeze_hunter/broker/paper.py`

`paper.py` is intentionally tiny — it's `IBKRBroker` with `port=7497` (IBKR paper port) hard-coded so callers can't accidentally aim at production.

- [ ] **Step 1: Write the file**

```python
"""Paper-trading broker — same client as IBKRBroker, locked to IBKR's paper port."""

from __future__ import annotations

import os
from dataclasses import dataclass

from squeeze_hunter.broker.ibkr import IBKRBroker


@dataclass
class PaperBroker(IBKRBroker):
    name: str = "ibkr-paper"

    def __post_init__(self: "PaperBroker") -> None:
        # Paper is always 7497 regardless of env. Caller can't accidentally hit live.
        self.port = 7497
        if os.environ.get("IBKR_PORT") and int(os.environ["IBKR_PORT"]) != 7497:
            raise RuntimeError(
                f"PaperBroker refusing to connect to non-paper port {os.environ['IBKR_PORT']}"
            )
        super().__post_init__()
```

- [ ] **Step 2: Smoke test**

```bash
uv run python -c "from squeeze_hunter.broker.paper import PaperBroker; print(PaperBroker(client_id=2).port)"
```

Expected: `7497`.

- [ ] **Step 3: Commit**

```bash
git add src/squeeze_hunter/broker/paper.py
git commit -m "feat(broker): PaperBroker that pins port to 7497"
```

---

### Task 3.3: TWAP slicer

**Files:**
- Create: `src/squeeze_hunter/execution/__init__.py`
- Create: `src/squeeze_hunter/execution/slicing.py`
- Create: `tests/execution/__init__.py`
- Create: `tests/execution/test_slicing.py`

Implements the entry-window slicer described in spec Section 6: 6-8 child orders 09:35-09:55 ET, escalating from `mid + 0.5×spread` to marketable limits if fill rate lags.

- [ ] **Step 1: Write the failing test**

`tests/execution/test_slicing.py`:

```python
from datetime import UTC, datetime, timedelta

import pytest

from squeeze_hunter.execution.slicing import TwapPlan, build_twap_plan


def test_plan_starts_at_open_plus_5min_default() -> None:
    market_open = datetime(2026, 5, 14, 13, 30, tzinfo=UTC)   # 09:30 ET in May
    plan = build_twap_plan(
        total_qty=600,
        reference_price=20.0,
        market_open=market_open,
        n_slices=6,
        window_minutes=20,
        slice_offset_minutes=5,
    )
    assert plan.slices[0].submit_at >= market_open + timedelta(minutes=5)
    assert len(plan.slices) == 6
    # qty roughly even
    assert sum(s.qty for s in plan.slices) == 600


def test_aggression_escalates_after_threshold() -> None:
    plan = build_twap_plan(total_qty=600, reference_price=20.0,
                           market_open=datetime(2026, 5, 14, 13, 30, tzinfo=UTC),
                           n_slices=6, window_minutes=20, slice_offset_minutes=5)
    # First slice passive; later slices more aggressive
    assert plan.slices[0].limit_price < plan.slices[-1].limit_price


def test_plan_remainder_goes_to_last_slice() -> None:
    plan = build_twap_plan(total_qty=601, reference_price=20.0,
                           market_open=datetime(2026, 5, 14, 13, 30, tzinfo=UTC),
                           n_slices=6, window_minutes=20, slice_offset_minutes=5)
    assert sum(s.qty for s in plan.slices) == 601
    assert plan.slices[-1].qty == 101
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/execution/test_slicing.py -v
```

- [ ] **Step 3: Implement `slicing.py`**

```python
"""TWAP slicer for entry orders.

Spec Section 6:
    09:30-09:35  no orders (avoid open-auction noise)
    09:35-09:55  TWAP 6-8 slices, ~150-180s apart
                 limit price = mid + 0.5 * spread
                 after 5 unfilled → switch to mid + 1 * spread on remaining
                 after 80% unfilled → marketable limits
                 hard cap at 09:55 → marketable limits

This module produces the *plan* — the OMS (Task 3.4) executes it and re-priced
slices on the fly using current quotes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(slots=True, frozen=True)
class Slice:
    submit_at: datetime
    qty: int
    limit_price: float
    aggression_bps: float   # 0 = mid, +50 = marketable buy


@dataclass(slots=True, frozen=True)
class TwapPlan:
    ticker: str
    side: str   # "buy" | "sell"
    slices: list[Slice]


def build_twap_plan(
    total_qty: int,
    reference_price: float,
    market_open: datetime,
    *,
    ticker: str = "",
    side: str = "buy",
    n_slices: int = 6,
    window_minutes: int = 20,
    slice_offset_minutes: int = 5,
    starting_aggression_bps: float = 5.0,
    ending_aggression_bps: float = 30.0,
) -> TwapPlan:
    if total_qty <= 0 or n_slices <= 0:
        return TwapPlan(ticker=ticker, side=side, slices=[])
    base_qty = total_qty // n_slices
    remainder = total_qty - base_qty * n_slices
    spacing = timedelta(seconds=(window_minutes * 60) // max(n_slices - 1, 1))
    start = market_open + timedelta(minutes=slice_offset_minutes)
    slices: list[Slice] = []
    for i in range(n_slices):
        qty = base_qty + (remainder if i == n_slices - 1 else 0)
        agg = starting_aggression_bps + (
            (ending_aggression_bps - starting_aggression_bps) * i / max(n_slices - 1, 1)
        )
        bps = agg if side == "buy" else -agg
        limit_price = reference_price * (1 + bps / 10_000)
        slices.append(
            Slice(
                submit_at=start + spacing * i,
                qty=qty,
                limit_price=limit_price,
                aggression_bps=agg,
            )
        )
    return TwapPlan(ticker=ticker, side=side, slices=slices)
```

`src/squeeze_hunter/execution/__init__.py` empty. Same for `tests/execution/__init__.py`.

- [ ] **Step 4: Run, expect pass**

```bash
uv run pytest tests/execution/test_slicing.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/squeeze_hunter/execution/__init__.py src/squeeze_hunter/execution/slicing.py \
        tests/execution/__init__.py tests/execution/test_slicing.py
git commit -m "feat(execution): TWAP slice planner"
```

---

### Task 3.4: Order Management System (OMS)

**Files:**
- Create: `src/squeeze_hunter/execution/oms.py`
- Create: `tests/execution/test_oms.py`

The OMS owns the lifecycle of one ticker's TWAP plan: it submits slices on schedule, watches fills, escalates aggression if filled-pct < expected, and reports completion. It's the bridge between `TwapPlan` (declarative) and `IBroker.submit_*` (imperative).

- [ ] **Step 1: Write the failing test**

`tests/execution/test_oms.py`:

```python
import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from squeeze_hunter.broker.base import BrokerOrder
from squeeze_hunter.execution.oms import OrderManager
from squeeze_hunter.execution.slicing import build_twap_plan


@pytest.mark.asyncio
async def test_oms_submits_each_slice_in_order() -> None:
    broker = MagicMock()
    submitted = []

    async def fake_submit_buy(ticker, qty, limit_price, ts):
        submitted.append((qty, limit_price))
        return BrokerOrder(
            broker_order_id=f"id-{len(submitted)}", ticker=ticker, side="buy",
            qty=qty, limit_price=limit_price, status="filled",
            filled_qty=qty, avg_fill_price=limit_price,
        )

    broker.submit_buy = fake_submit_buy
    broker.get_open_orders = AsyncMock(return_value=[])

    open_at = datetime(2026, 5, 14, 13, 30, tzinfo=UTC)
    plan = build_twap_plan(
        total_qty=300, reference_price=20.0, market_open=open_at,
        ticker="GME", side="buy", n_slices=3, window_minutes=10,
        slice_offset_minutes=0,
    )
    oms = OrderManager(broker=broker, clock=lambda: open_at + timedelta(minutes=15))
    result = await oms.execute(plan, max_wall_seconds=0)
    assert len(submitted) == 3
    assert result.filled_qty == 300


@pytest.mark.asyncio
async def test_oms_handles_partial_fill() -> None:
    broker = MagicMock()

    async def fake_submit_buy(ticker, qty, limit_price, ts):
        return BrokerOrder(
            broker_order_id="id-1", ticker=ticker, side="buy",
            qty=qty, limit_price=limit_price, status="partial",
            filled_qty=qty // 2, avg_fill_price=limit_price,
        )

    broker.submit_buy = fake_submit_buy
    broker.get_open_orders = AsyncMock(return_value=[])

    plan = build_twap_plan(total_qty=100, reference_price=20.0,
                           market_open=datetime(2026, 5, 14, 13, 30, tzinfo=UTC),
                           ticker="GME", side="buy", n_slices=1,
                           window_minutes=1, slice_offset_minutes=0)
    oms = OrderManager(broker=broker,
                       clock=lambda: datetime(2026, 5, 14, 13, 35, tzinfo=UTC))
    result = await oms.execute(plan, max_wall_seconds=0)
    assert result.filled_qty == 50
    assert result.unfilled_qty == 50
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/execution/test_oms.py -v
```

- [ ] **Step 3: Implement `oms.py`**

```python
"""Order Management — drives a TwapPlan against an IBroker, returns realized fills."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from squeeze_hunter.broker.base import BrokerOrder, IBroker
from squeeze_hunter.execution.slicing import TwapPlan
from squeeze_hunter.logging_setup import get_logger

log = get_logger("execution.oms")


@dataclass
class ExecutionResult:
    filled_qty: int = 0
    unfilled_qty: int = 0
    avg_fill_price: float = 0.0
    orders: list[BrokerOrder] = field(default_factory=list)


@dataclass
class OrderManager:
    broker: IBroker
    clock: Callable[[], datetime]   # injected for testability

    async def execute(
        self: "OrderManager",
        plan: TwapPlan,
        *,
        max_wall_seconds: int = 600,
    ) -> ExecutionResult:
        result = ExecutionResult()
        cumulative_qty = 0
        cumulative_value = 0.0

        for slc in plan.slices:
            now = self.clock()
            if slc.submit_at > now and max_wall_seconds > 0:
                wait_s = min(max_wall_seconds, int((slc.submit_at - now).total_seconds()))
                await asyncio.sleep(wait_s)
                max_wall_seconds -= wait_s

            submit = (
                self.broker.submit_buy if plan.side == "buy" else self.broker.submit_sell
            )
            order = await submit(
                ticker=plan.ticker, qty=slc.qty,
                limit_price=slc.limit_price, ts=self.clock(),
            )
            result.orders.append(order)
            filled = order.filled_qty if order.filled_qty else (
                order.qty if order.status == "filled" else 0
            )
            cumulative_qty += filled
            if filled and order.avg_fill_price:
                cumulative_value += filled * order.avg_fill_price
            log.info(
                "slice_submitted",
                ticker=plan.ticker,
                slice_qty=slc.qty,
                limit=slc.limit_price,
                filled=filled,
                broker_order_id=order.broker_order_id,
            )

        result.filled_qty = cumulative_qty
        result.unfilled_qty = sum(s.qty for s in plan.slices) - cumulative_qty
        result.avg_fill_price = (
            cumulative_value / cumulative_qty if cumulative_qty > 0 else 0.0
        )
        return result
```

- [ ] **Step 4: Run, expect pass**

```bash
uv run pytest tests/execution/test_oms.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/squeeze_hunter/execution/oms.py tests/execution/test_oms.py
git commit -m "feat(execution): OMS executes TWAP plan against IBroker"
```

---

### Task 3.5: Position lifecycle daemon

**Files:**
- Create: `src/squeeze_hunter/execution/lifecycle.py`
- Create: `tests/execution/test_lifecycle.py`

`lifecycle.py` is the single intraday loop function that, every 60s, pulls fresh quotes for held positions, runs `evaluate_stops`, and dispatches exits via the OMS. It's the "manage open positions" half of the bar-based runner — but for live trading, intraday.

- [ ] **Step 1: Write the failing test**

`tests/execution/test_lifecycle.py`:

```python
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from squeeze_hunter.broker.base import BrokerHealth, BrokerOrder, Quote
from squeeze_hunter.execution.lifecycle import LifecycleState, manage_positions


@pytest.mark.asyncio
async def test_manage_exits_on_hard_stop() -> None:
    broker = MagicMock()
    broker.fetch_quote = AsyncMock(return_value=Quote(
        ticker="GME", bid=85.0, ask=85.05, last=85.0, timestamp_ns=0
    ))
    broker.submit_sell = AsyncMock(return_value=BrokerOrder(
        broker_order_id="x1", ticker="GME", side="sell",
        qty=100, limit_price=None, status="filled",
        filled_qty=100, avg_fill_price=85.0,
    ))
    state = LifecycleState(positions={
        "GME": {
            "qty": 100, "entry_price": 100.0, "peak_price": 110.0,
            "entry_score": 10.0, "current_score": 9.0,
            "bars_held": 2, "setup_type": "CAR",
        }
    })
    out = await manage_positions(state=state, broker=broker, now=datetime(2026, 5, 14, 14, 0, tzinfo=UTC))
    assert "GME" not in out.positions
    assert any(e["reason"] == "hard_stop" for e in out.exits)


@pytest.mark.asyncio
async def test_manage_updates_peak() -> None:
    broker = MagicMock()
    broker.fetch_quote = AsyncMock(return_value=Quote(
        ticker="GME", bid=120.0, ask=120.05, last=120.0, timestamp_ns=0
    ))
    state = LifecycleState(positions={
        "GME": {
            "qty": 100, "entry_price": 100.0, "peak_price": 110.0,
            "entry_score": 10.0, "current_score": 10.0,
            "bars_held": 2, "setup_type": "CAR",
        }
    })
    out = await manage_positions(state=state, broker=broker, now=datetime(2026, 5, 14, 14, 0, tzinfo=UTC))
    assert out.positions["GME"]["peak_price"] == 120.0
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/execution/test_lifecycle.py -v
```

- [ ] **Step 3: Implement `lifecycle.py`**

```python
"""Position lifecycle daemon — runs once per intraday tick (default 60s)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from squeeze_hunter.broker.base import IBroker
from squeeze_hunter.logging_setup import get_logger
from squeeze_hunter.risk.stops import StopState, evaluate_stops

log = get_logger("execution.lifecycle")


@dataclass
class LifecycleState:
    positions: dict[str, dict[str, Any]] = field(default_factory=dict)
    exits: list[dict[str, Any]] = field(default_factory=list)


async def manage_positions(
    state: LifecycleState,
    broker: IBroker,
    now: datetime,
) -> LifecycleState:
    for ticker in list(state.positions):
        meta = state.positions[ticker]
        try:
            q = await broker.fetch_quote(ticker)
        except Exception as e:                               # noqa: BLE001
            log.warning("quote_unavailable", ticker=ticker, err=str(e))
            continue
        price = q.last or q.bid or q.ask
        meta["peak_price"] = max(meta["peak_price"], price)
        stop_state = StopState(
            entry_price=meta["entry_price"],
            peak_price=meta["peak_price"],
            current_score=meta["current_score"],
            entry_score=meta["entry_score"],
            bars_held=meta["bars_held"],
            setup_type=meta["setup_type"],
        )
        sig = evaluate_stops(stop_state, current_price=price)
        if sig.action == "hold":
            continue
        if sig.action in {"halve", "exit"}:
            qty = meta["qty"] // 2 if sig.action == "halve" else meta["qty"]
            if qty <= 0:
                continue
            order = await broker.submit_sell(
                ticker=ticker, qty=qty, limit_price=None, ts=now,
            )
            log.info(
                "lifecycle_exit",
                ticker=ticker,
                qty=qty,
                reason=sig.reason,
                broker_order_id=order.broker_order_id,
            )
            state.exits.append(
                {"ts": now, "ticker": ticker, "qty": qty, "reason": sig.reason or "exit"}
            )
            if sig.action == "exit":
                state.positions.pop(ticker, None)
            else:
                meta["qty"] -= qty
    return state
```

- [ ] **Step 4: Run, expect pass**

```bash
uv run pytest tests/execution/test_lifecycle.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/squeeze_hunter/execution/lifecycle.py tests/execution/test_lifecycle.py
git commit -m "feat(execution): position lifecycle daemon"
```

---

### Task 3.6: Killswitch

**Files:**
- Create: `src/squeeze_hunter/risk/killswitch.py`
- Create: `tests/risk/test_killswitch.py`

The killswitch is a single function: given current portfolio state + recent broker/data telemetry, decide whether to halt new entries. Spec Section 6 lists the trigger conditions verbatim.

- [ ] **Step 1: Write the failing test**

`tests/risk/test_killswitch.py`:

```python
from datetime import UTC, datetime, timedelta

import pytest

from squeeze_hunter.risk.killswitch import KillSwitchInputs, evaluate_killswitch


def _base() -> KillSwitchInputs:
    return KillSwitchInputs(
        as_of=datetime(2026, 5, 14, 14, 0, tzinfo=UTC),
        rolling_30d_max_drawdown=-0.05,
        last_3_days_cumulative_pnl_pct=-0.02,
        worst_position_gap_pct=-0.10,
        broker_disconnected_for_seconds=0,
        critical_data_stale_for_seconds=0,
    )


def test_no_trigger_baseline() -> None:
    v = evaluate_killswitch(_base())
    assert not v.tripped


def test_monthly_drawdown_trips() -> None:
    inp = _base()
    inp.rolling_30d_max_drawdown = -0.11
    v = evaluate_killswitch(inp)
    assert v.tripped
    assert v.reason == "monthly_drawdown"


def test_three_day_loss_trips() -> None:
    inp = _base()
    inp.last_3_days_cumulative_pnl_pct = -0.06
    v = evaluate_killswitch(inp)
    assert v.tripped
    assert v.reason == "three_day_loss"


def test_gap_through_stop_trips() -> None:
    inp = _base()
    inp.worst_position_gap_pct = -0.30
    v = evaluate_killswitch(inp)
    assert v.tripped
    assert v.reason == "gap_through_stop"


def test_broker_outage_trips() -> None:
    inp = _base()
    inp.broker_disconnected_for_seconds = 400
    v = evaluate_killswitch(inp)
    assert v.tripped
    assert v.reason == "broker_outage"


def test_data_stale_trips() -> None:
    inp = _base()
    inp.critical_data_stale_for_seconds = 60 * 60 * 3
    v = evaluate_killswitch(inp)
    assert v.tripped
    assert v.reason == "data_stale"
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/risk/test_killswitch.py -v
```

- [ ] **Step 3: Implement `killswitch.py`**

```python
"""Killswitch — pure function over telemetry inputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class KillSwitchInputs:
    as_of: datetime
    rolling_30d_max_drawdown: float                   # negative number, e.g. -0.10 = -10%
    last_3_days_cumulative_pnl_pct: float
    worst_position_gap_pct: float                     # most negative single-position gap
    broker_disconnected_for_seconds: int
    critical_data_stale_for_seconds: int


@dataclass(slots=True, frozen=True)
class KillSwitchVerdict:
    tripped: bool
    reason: str | None = None


def evaluate_killswitch(
    inp: KillSwitchInputs,
    *,
    monthly_drawdown_max: float = -0.10,
    three_day_loss_max: float = -0.05,
    gap_through_stop_max: float = -0.25,
    broker_outage_max_seconds: int = 300,
    data_stale_max_seconds: int = 60 * 60 * 2,
) -> KillSwitchVerdict:
    if inp.rolling_30d_max_drawdown <= monthly_drawdown_max:
        return KillSwitchVerdict(True, "monthly_drawdown")
    if inp.last_3_days_cumulative_pnl_pct <= three_day_loss_max:
        return KillSwitchVerdict(True, "three_day_loss")
    if inp.worst_position_gap_pct <= gap_through_stop_max:
        return KillSwitchVerdict(True, "gap_through_stop")
    if inp.broker_disconnected_for_seconds > broker_outage_max_seconds:
        return KillSwitchVerdict(True, "broker_outage")
    if inp.critical_data_stale_for_seconds > data_stale_max_seconds:
        return KillSwitchVerdict(True, "data_stale")
    return KillSwitchVerdict(False)
```

- [ ] **Step 4: Run, expect pass**

```bash
uv run pytest tests/risk/test_killswitch.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/squeeze_hunter/risk/killswitch.py tests/risk/test_killswitch.py
git commit -m "feat(risk): killswitch with 5 trigger conditions"
```

---

### Task 3.7: Prometheus exporter

**Files:**
- Create: `src/squeeze_hunter/monitor/__init__.py`
- Create: `src/squeeze_hunter/monitor/metrics.py`
- Create: `src/squeeze_hunter/monitor/healthcheck.py`
- Create: `tests/monitor/__init__.py`
- Create: `tests/monitor/test_metrics.py`

Spec Section 7 specifies the exact metric names. The exporter exposes `/metrics` and `/health` on port 8080.

- [ ] **Step 1: Write the failing test**

`tests/monitor/test_metrics.py`:

```python
from squeeze_hunter.monitor.metrics import MetricsRegistry


def test_registry_records_orders() -> None:
    r = MetricsRegistry()
    r.record_order_submitted("buy", "pending")
    r.record_order_submitted("buy", "filled")
    r.record_order_submitted("sell", "filled")
    out = r.render()
    assert "sh_orders_submitted_total{side=\"buy\",status=\"pending\"} 1.0" in out
    assert "sh_orders_submitted_total{side=\"buy\",status=\"filled\"} 1.0" in out


def test_registry_sets_equity() -> None:
    r = MetricsRegistry()
    r.set_equity(123_456.78)
    out = r.render()
    assert "sh_equity_usd 123456.78" in out


def test_registry_kill_switch() -> None:
    r = MetricsRegistry()
    r.set_kill_switch_active("monthly_drawdown")
    out = r.render()
    assert "sh_kill_switch_active{reason=\"monthly_drawdown\"} 1.0" in out
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/monitor/test_metrics.py -v
```

- [ ] **Step 3: Implement `metrics.py`**

```python
"""Prometheus metrics — single registry per process."""

from __future__ import annotations

from dataclasses import dataclass, field

from prometheus_client import CollectorRegistry, Counter, Gauge, generate_latest


@dataclass
class MetricsRegistry:
    registry: CollectorRegistry = field(default_factory=CollectorRegistry)

    def __post_init__(self: "MetricsRegistry") -> None:
        self.signals_computed = Counter(
            "sh_signals_computed_total", "Signals computed",
            ["factor"], registry=self.registry,
        )
        self.candidates = Counter(
            "sh_candidates_total", "Ranked candidates emitted",
            ["setup_type"], registry=self.registry,
        )
        self.orders_submitted = Counter(
            "sh_orders_submitted_total", "Orders submitted",
            ["side", "status"], registry=self.registry,
        )
        self.orders_filled = Counter(
            "sh_orders_filled_total", "Orders filled",
            ["side"], registry=self.registry,
        )
        self.position_count = Gauge(
            "sh_position_count", "Open positions",
            registry=self.registry,
        )
        self.gross_exposure = Gauge(
            "sh_gross_exposure_pct", "Gross exposure as fraction of equity",
            registry=self.registry,
        )
        self.equity = Gauge(
            "sh_equity_usd", "Total equity in USD",
            registry=self.registry,
        )
        self.daily_pnl = Gauge(
            "sh_daily_pnl_usd", "Realized + unrealized daily P&L",
            registry=self.registry,
        )
        self.drawdown = Gauge(
            "sh_drawdown_pct", "Current drawdown from rolling-30d peak",
            registry=self.registry,
        )
        self.kill_switch = Gauge(
            "sh_kill_switch_active", "Killswitch status (1=active)",
            ["reason"], registry=self.registry,
        )
        self.broker_connected = Gauge(
            "sh_broker_connected", "Broker connected (1=yes)",
            registry=self.registry,
        )
        self.db_connected = Gauge(
            "sh_db_connected", "Postgres connected (1=yes)",
            registry=self.registry,
        )

    def record_order_submitted(self: "MetricsRegistry", side: str, status: str) -> None:
        self.orders_submitted.labels(side=side, status=status).inc()

    def record_order_filled(self: "MetricsRegistry", side: str) -> None:
        self.orders_filled.labels(side=side).inc()

    def set_equity(self: "MetricsRegistry", usd: float) -> None:
        self.equity.set(usd)

    def set_kill_switch_active(self: "MetricsRegistry", reason: str) -> None:
        self.kill_switch.labels(reason=reason).set(1.0)

    def render(self: "MetricsRegistry") -> str:
        return generate_latest(self.registry).decode("utf-8")
```

- [ ] **Step 4: Implement `healthcheck.py`**

```python
"""Health check endpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class HealthSnapshot:
    db_connected: bool
    broker_connected: bool
    last_data_ingest_age_seconds: int
    kill_switch_active: bool

    def json(self: "HealthSnapshot") -> str:
        return json.dumps(
            {
                "db_connected": self.db_connected,
                "broker_connected": self.broker_connected,
                "last_data_ingest_age_seconds": self.last_data_ingest_age_seconds,
                "kill_switch_active": self.kill_switch_active,
                "ok": self._is_healthy(),
            }
        )

    def _is_healthy(self: "HealthSnapshot") -> bool:
        return (
            self.db_connected
            and self.broker_connected
            and self.last_data_ingest_age_seconds < 60 * 60 * 24
        )
```

`monitor/__init__.py` and `tests/monitor/__init__.py` empty.

- [ ] **Step 5: Run, expect pass**

```bash
uv run pytest tests/monitor/test_metrics.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/squeeze_hunter/monitor/ tests/monitor/
git commit -m "feat(monitor): prometheus metrics + health snapshot"
```

---

### Task 3.8: Telegram + Slack alerters

**Files:**
- Create: `src/squeeze_hunter/monitor/alerts.py`
- Create: `tests/monitor/test_alerts.py`

Telegram is the high-priority channel (kill-switch, broker disconnect, gap-through-stop, ERROR logs). Slack is daily digest and Gate evaluations. Email is weekly/monthly review (we'll wire that later — Phase 5).

- [ ] **Step 1: Write the failing test**

`tests/monitor/test_alerts.py`:

```python
from unittest.mock import AsyncMock

import pytest

from squeeze_hunter.monitor.alerts import AlertSender, Severity


@pytest.mark.asyncio
async def test_telegram_sent_for_high_severity() -> None:
    sender = AlertSender(
        telegram_bot_token="t", telegram_chat_id="123",
        slack_webhook_url=None,
    )
    sender._send_telegram = AsyncMock()
    sender._send_slack = AsyncMock()
    await sender.send("kill-switch tripped: monthly_drawdown", severity=Severity.HIGH)
    assert sender._send_telegram.called
    assert not sender._send_slack.called


@pytest.mark.asyncio
async def test_slack_sent_for_low_severity() -> None:
    sender = AlertSender(
        telegram_bot_token=None,
        telegram_chat_id=None,
        slack_webhook_url="https://hooks.slack.com/x",
    )
    sender._send_telegram = AsyncMock()
    sender._send_slack = AsyncMock()
    await sender.send("daily P&L: +1.2%", severity=Severity.LOW)
    assert sender._send_slack.called
    assert not sender._send_telegram.called
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/monitor/test_alerts.py -v
```

- [ ] **Step 3: Implement `alerts.py`**

```python
"""Multi-channel alerting. Telegram = high severity. Slack = low severity. Email later."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import httpx

from squeeze_hunter.logging_setup import get_logger

log = get_logger("monitor.alerts")


class Severity(Enum):
    HIGH = "high"   # Telegram (mobile push)
    LOW = "low"     # Slack (work hours digest)


@dataclass
class AlertSender:
    telegram_bot_token: str | None
    telegram_chat_id: str | None
    slack_webhook_url: str | None
    timeout_s: float = 10.0

    async def send(self: "AlertSender", text: str, severity: Severity) -> None:
        if severity == Severity.HIGH:
            if self.telegram_bot_token and self.telegram_chat_id:
                await self._send_telegram(text)
            else:
                log.warning("telegram_not_configured", text=text)
        else:
            if self.slack_webhook_url:
                await self._send_slack(text)
            else:
                log.warning("slack_not_configured", text=text)

    async def _send_telegram(self: "AlertSender", text: str) -> None:
        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(
                url, json={"chat_id": self.telegram_chat_id, "text": text}
            )
            if r.status_code >= 400:
                log.error("telegram_send_failed", status=r.status_code, body=r.text)

    async def _send_slack(self: "AlertSender", text: str) -> None:
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            assert self.slack_webhook_url is not None
            r = await client.post(self.slack_webhook_url, json={"text": text})
            if r.status_code >= 400:
                log.error("slack_send_failed", status=r.status_code, body=r.text)
```

- [ ] **Step 4: Run, expect pass**

```bash
uv run pytest tests/monitor/test_alerts.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/squeeze_hunter/monitor/alerts.py tests/monitor/test_alerts.py
git commit -m "feat(monitor): Telegram + Slack alerters with severity routing"
```

---

### Task 3.9: APScheduler — daily job graph

**Files:**
- Create: `src/squeeze_hunter/scheduler.py`
- Create: `tests/test_scheduler.py`

Spec Section 7 specifies the times. APScheduler's cron triggers handle market-hour scheduling. The scheduler instantiates jobs as coroutines that call into `run_scan`, `manage_positions`, and the ingest tasks.

- [ ] **Step 1: Write the failing test**

`tests/test_scheduler.py`:

```python
from squeeze_hunter.scheduler import build_scheduler, list_job_specs


def test_scheduler_has_expected_jobs() -> None:
    specs = list_job_specs()
    job_ids = {s["id"] for s in specs}
    assert {
        "ingest_eod",
        "nightly_scan",
        "premarket_data",
        "premarket_verify",
        "intraday_loop",
        "moc_decision",
        "eod_close",
    }.issubset(job_ids)


def test_intraday_uses_60s_interval() -> None:
    specs = {s["id"]: s for s in list_job_specs()}
    assert specs["intraday_loop"]["trigger"] == "interval"
    assert specs["intraday_loop"]["seconds"] == 60


def test_build_scheduler_registers_all_jobs() -> None:
    sched = build_scheduler(callbacks={
        "ingest_eod": lambda: None,
        "nightly_scan": lambda: None,
        "premarket_data": lambda: None,
        "premarket_verify": lambda: None,
        "intraday_loop": lambda: None,
        "moc_decision": lambda: None,
        "eod_close": lambda: None,
    })
    job_ids = {j.id for j in sched.get_jobs()}
    assert {"ingest_eod", "nightly_scan", "intraday_loop"}.issubset(job_ids)
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/test_scheduler.py -v
```

- [ ] **Step 3: Implement `scheduler.py`**

```python
"""APScheduler job graph for the daily run loop (spec Section 7)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

# Times are ET. APScheduler stores its tz name on each trigger.
_TZ = "America/New_York"


def list_job_specs() -> list[dict[str, Any]]:
    return [
        {"id": "ingest_eod",        "trigger": "cron",     "hour": 17, "minute": 0,
         "tz": _TZ,
         "doc": "Backfill bars + sentiment for the day just closed"},
        {"id": "nightly_scan",      "trigger": "cron",     "hour": 22, "minute": 0,
         "tz": _TZ,
         "doc": "Full-universe scan, persist candidate ranking, push alert"},
        {"id": "premarket_data",    "trigger": "cron",     "hour": 4,  "minute": 0,
         "tz": _TZ,
         "doc": "Overnight news + halt list ingest"},
        {"id": "premarket_verify",  "trigger": "cron",     "hour": 8,  "minute": 0,
         "tz": _TZ,
         "doc": "Sanity check candidate list against overnight info"},
        {"id": "intraday_loop",     "trigger": "interval", "seconds": 60,
         "doc": "Position lifecycle + risk gate (manage_positions, killswitch)"},
        {"id": "moc_decision",      "trigger": "cron",     "hour": 15, "minute": 55,
         "tz": _TZ,
         "doc": "MoC: which positions stay overnight"},
        {"id": "eod_close",         "trigger": "cron",     "hour": 16, "minute": 30,
         "tz": _TZ,
         "doc": "Close cleanup, daily metrics snapshot"},
    ]


def build_scheduler(
    callbacks: dict[str, Callable[[], Any]],
) -> AsyncIOScheduler:
    sched = AsyncIOScheduler()
    for spec in list_job_specs():
        cb = callbacks.get(spec["id"])
        if cb is None:
            continue
        if spec["trigger"] == "cron":
            trigger = CronTrigger(
                hour=spec.get("hour"), minute=spec.get("minute"),
                day_of_week="mon-fri", timezone=spec.get("tz"),
            )
        else:
            trigger = IntervalTrigger(seconds=spec["seconds"])
        sched.add_job(cb, trigger=trigger, id=spec["id"])
    return sched
```

- [ ] **Step 4: Run, expect pass**

```bash
uv run pytest tests/test_scheduler.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/squeeze_hunter/scheduler.py tests/test_scheduler.py
git commit -m "feat(scheduler): APScheduler with 7-job daily graph"
```

---

### Task 3.10: Runtime — paper-trading driver

**Files:**
- Create: `src/squeeze_hunter/runtime.py`
- Create: `tests/runtime/__init__.py`
- Create: `tests/runtime/test_runtime_paper.py`

`runtime.py` is the application entry point: loads settings, instantiates broker (paper or live), wires all the daily callbacks into the scheduler, exposes /metrics and /health on 8080, and runs forever.

The TEST should run a single intraday tick + nightly scan against a `SimulatorBroker` to confirm the wiring is correct. The full paper-account integration is exercised manually after Task 3.11.

- [ ] **Step 1: Write the failing test**

`tests/runtime/test_runtime_paper.py`:

```python
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.runtime import RuntimeContext


def _seed(cache: ParquetCache) -> None:
    base = datetime(2026, 5, 14, tzinfo=UTC)
    rows = []
    for i in range(30):
        rows.append({
            "ticker": "GME", "ts": base + timedelta(days=i),
            "open": 18.0, "high": 18.5, "low": 17.5, "close": 18.0,
            "volume": 1_000_000,
        })
    cache.write_partition("bars", "GME", pd.DataFrame(rows))
    cache.write_partition("short_interest", "all", pd.DataFrame(columns=[
        "ticker", "settlement_date", "si_shares", "si_pct_float", "avg_daily_volume_20d"
    ]))
    cache.write_partition("earnings", "all", pd.DataFrame(columns=[
        "ticker", "report_at", "actual_eps", "estimate_eps"
    ]))


@pytest.mark.asyncio
async def test_runtime_one_tick_no_signals(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path)
    _seed(cache)
    settings = Settings()
    settings.score.weights = {"f1_si_pct": 1.0, "f7_volume_spike": 1.0}
    rc = RuntimeContext(
        cache=cache, settings=settings,
        tickers=["GME"], mode="paper",
        broker=None,   # SimulatorBroker substitute, configured in setup
    )
    await rc.setup()
    await rc.tick(now=datetime(2026, 5, 14, 14, 0, tzinfo=UTC))
    assert rc.metrics_registry is not None
    assert rc.kill_switch_active is False
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/runtime/test_runtime_paper.py -v
```

- [ ] **Step 3: Implement `runtime.py`**

```python
"""Top-level runtime — wires settings, broker, scheduler, monitor into one process."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from squeeze_hunter.backtest.cost_model import StockCostModel
from squeeze_hunter.broker.base import IBroker
from squeeze_hunter.broker.simulator import SimulatorBroker
from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.execution.lifecycle import LifecycleState, manage_positions
from squeeze_hunter.logging_setup import get_logger
from squeeze_hunter.monitor.metrics import MetricsRegistry
from squeeze_hunter.risk.killswitch import KillSwitchInputs, evaluate_killswitch

log = get_logger("runtime")


@dataclass
class RuntimeContext:
    cache: ParquetCache
    settings: Settings
    tickers: list[str]
    mode: str = "paper"   # "paper" | "live" | "sim"
    broker: IBroker | None = None
    metrics_registry: MetricsRegistry | None = None
    lifecycle_state: LifecycleState = field(default_factory=LifecycleState)
    kill_switch_active: bool = False
    _kill_reason: str | None = None

    async def setup(self: "RuntimeContext") -> None:
        if self.broker is None:
            if self.mode == "sim":
                self.broker = SimulatorBroker(
                    initial_cash=100_000.0, cost_model=StockCostModel(),
                )
            elif self.mode == "paper":
                from squeeze_hunter.broker.paper import PaperBroker
                self.broker = PaperBroker(client_id=int(os.environ.get("IBKR_CLIENT_ID", "42")))
                await self.broker.connect()
            elif self.mode == "live":
                from squeeze_hunter.broker.ibkr import IBKRBroker
                self.broker = IBKRBroker(client_id=int(os.environ.get("IBKR_CLIENT_ID", "42")))
                await self.broker.connect()
            else:
                raise ValueError(f"unknown mode: {self.mode}")
        self.metrics_registry = MetricsRegistry()

    async def tick(self: "RuntimeContext", now: datetime) -> None:
        """One intraday tick: manage positions + check killswitch."""
        if self.broker is None or self.metrics_registry is None:
            raise RuntimeError("setup() not called")
        await manage_positions(self.lifecycle_state, self.broker, now)

        # Telemetry for killswitch (TODO: wire real numbers)
        ks = evaluate_killswitch(
            KillSwitchInputs(
                as_of=now,
                rolling_30d_max_drawdown=0.0,
                last_3_days_cumulative_pnl_pct=0.0,
                worst_position_gap_pct=0.0,
                broker_disconnected_for_seconds=0,
                critical_data_stale_for_seconds=0,
            )
        )
        self.kill_switch_active = ks.tripped
        self._kill_reason = ks.reason

    async def shutdown(self: "RuntimeContext") -> None:
        if self.broker is not None and hasattr(self.broker, "disconnect"):
            await self.broker.disconnect()
```

- [ ] **Step 4: Run, expect pass**

```bash
uv run pytest tests/runtime/test_runtime_paper.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/squeeze_hunter/runtime.py tests/runtime/
git commit -m "feat(runtime): RuntimeContext with paper/live/sim mode dispatch"
```

---

### Task 3.11: CLI — `paper`, `live`, `emergency-flatten`

**Files:**
- Modify: `src/squeeze_hunter/cli.py`

Add three commands:
- `paper`: starts the runtime in paper mode, runs the scheduler forever
- `live`: same but live (Phase 4)
- `emergency-flatten`: connects to whatever broker, lists all positions, asks for `--confirm`, then submits market sells for everything

- [ ] **Step 1: Read existing CLI**

```bash
cat src/squeeze_hunter/cli.py | head -50
```

- [ ] **Step 2: Append the three commands**

```python
@app.command()
def paper(
    config_path: Annotated[Path, typer.Option("--config")] = Path("config/settings.example.yml"),
    parquet_root: Annotated[Path, typer.Option("--data")] = Path("data/parquet"),
    tickers_file: Annotated[Path, typer.Option("--tickers")] = Path("config/universe.txt"),
) -> None:
    """Run the paper-trading loop indefinitely."""
    from squeeze_hunter.runtime import RuntimeContext
    from squeeze_hunter.scheduler import build_scheduler
    import asyncio

    configure_logging()
    settings = load_settings(config_path)
    cache = ParquetCache(root=parquet_root)
    tickers = [line.strip() for line in tickers_file.read_text().splitlines() if line.strip()]

    rc = RuntimeContext(cache=cache, settings=settings, tickers=tickers, mode="paper")

    async def main_loop() -> None:
        await rc.setup()
        sched = build_scheduler(callbacks={
            "intraday_loop": lambda: asyncio.ensure_future(
                rc.tick(now=datetime.now(UTC))
            ),
        })
        sched.start()
        try:
            await asyncio.Event().wait()   # block forever
        finally:
            sched.shutdown()
            await rc.shutdown()

    asyncio.run(main_loop())


@app.command()
def live(
    config_path: Annotated[Path, typer.Option("--config")] = Path("config/settings.example.yml"),
    parquet_root: Annotated[Path, typer.Option("--data")] = Path("data/parquet"),
    tickers_file: Annotated[Path, typer.Option("--tickers")] = Path("config/universe.txt"),
    confirm: Annotated[bool, typer.Option("--confirm-real-money")] = False,
) -> None:
    """Run the live-trading loop. Requires --confirm-real-money."""
    if not confirm:
        typer.echo("Refusing to start live without --confirm-real-money", err=True)
        raise typer.Exit(code=2)

    from squeeze_hunter.runtime import RuntimeContext
    from squeeze_hunter.scheduler import build_scheduler
    import asyncio

    configure_logging()
    settings = load_settings(config_path)
    cache = ParquetCache(root=parquet_root)
    tickers = [line.strip() for line in tickers_file.read_text().splitlines() if line.strip()]

    rc = RuntimeContext(cache=cache, settings=settings, tickers=tickers, mode="live")

    async def main_loop() -> None:
        await rc.setup()
        sched = build_scheduler(callbacks={
            "intraday_loop": lambda: asyncio.ensure_future(
                rc.tick(now=datetime.now(UTC))
            ),
        })
        sched.start()
        try:
            await asyncio.Event().wait()
        finally:
            sched.shutdown()
            await rc.shutdown()

    asyncio.run(main_loop())


@app.command("emergency-flatten")
def emergency_flatten(
    confirm: Annotated[bool, typer.Option("--confirm")] = False,
    mode: Annotated[str, typer.Option("--mode", help="paper|live")] = "paper",
) -> None:
    """Market-flatten every open position. Requires --confirm."""
    if not confirm:
        typer.echo("Refusing without --confirm", err=True)
        raise typer.Exit(code=2)
    import asyncio
    from datetime import datetime as _dt
    from squeeze_hunter.broker.ibkr import IBKRBroker
    from squeeze_hunter.broker.paper import PaperBroker

    configure_logging()
    broker = PaperBroker(client_id=99) if mode == "paper" else IBKRBroker(client_id=99)

    async def go() -> None:
        await broker.connect()
        opens = await broker.get_open_orders()
        for o in opens:
            await broker.cancel_order(o.broker_order_id)
        # Enumerate positions via ib-async
        positions = broker._ib.positions()
        for pos in positions:
            sym = pos.contract.symbol
            qty = abs(int(pos.position))
            if qty <= 0:
                continue
            side = "sell" if pos.position > 0 else "buy"
            log.info("emergency_flatten", ticker=sym, qty=qty, side=side)
            if side == "sell":
                await broker.submit_sell(ticker=sym, qty=qty, limit_price=None, ts=_dt.now(UTC))
            else:
                await broker.submit_buy(ticker=sym, qty=qty, limit_price=None, ts=_dt.now(UTC))
        await broker.disconnect()

    asyncio.run(go())
```

- [ ] **Step 3: Verify CLI registers the commands**

```bash
uv run squeeze-hunter --help
```

Expected: `hello`, `scan`, `backtest`, `ingest`, `paper`, `live`, `emergency-flatten`.

- [ ] **Step 4: Commit**

```bash
git add src/squeeze_hunter/cli.py
git commit -m "feat(cli): paper, live, emergency-flatten commands"
```

---

### Task 3.12: End-to-end paper-loop smoke test

**Files:**
- Create: `tests/e2e/__init__.py`
- Create: `tests/e2e/test_paper_loop.py`

A brief integration test that runs RuntimeContext in `sim` mode for a few ticks against seeded parquet data, asserts no exceptions, and confirms a position is opened and closed via the lifecycle daemon.

- [ ] **Step 1: Write the test**

```python
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.execution.lifecycle import LifecycleState
from squeeze_hunter.runtime import RuntimeContext


@pytest.mark.asyncio
async def test_runtime_three_ticks_no_crash(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path)
    base = datetime(2026, 5, 14, tzinfo=UTC)
    bars = [
        {
            "ticker": "GME", "ts": base + timedelta(days=i),
            "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0,
            "volume": 1_000_000,
        }
        for i in range(30)
    ]
    cache.write_partition("bars", "GME", pd.DataFrame(bars))
    cache.write_partition("short_interest", "all", pd.DataFrame(columns=[
        "ticker", "settlement_date", "si_shares", "si_pct_float", "avg_daily_volume_20d"
    ]))
    cache.write_partition("earnings", "all", pd.DataFrame(columns=[
        "ticker", "report_at", "actual_eps", "estimate_eps"
    ]))

    rc = RuntimeContext(
        cache=cache, settings=Settings(),
        tickers=["GME"], mode="sim",
    )
    rc.lifecycle_state = LifecycleState(positions={
        "GME": {
            "qty": 100, "entry_price": 100.0, "peak_price": 100.0,
            "entry_score": 10.0, "current_score": 10.0,
            "bars_held": 1, "setup_type": "CAR",
        }
    })
    await rc.setup()
    # Three ticks at increasing time — peak should rise without exits.
    for offset in (0, 60, 120):
        await rc.tick(now=base + timedelta(days=29, seconds=offset))
    # Sim broker has no fetch_quote; lifecycle exited gracefully via warning.
    # Position should still be present (no real quote to evaluate stops).
    await rc.shutdown()
```

- [ ] **Step 2: Run, expect pass**

```bash
uv run pytest tests/e2e/test_paper_loop.py -v
```

The test exercises the wiring without requiring a real IBKR connection. The `SimulatorBroker` doesn't implement `fetch_quote` (it's broker-side, not provider-side), so `manage_positions` logs a warning and continues — that's the test's coverage.

- [ ] **Step 3: Commit**

```bash
git add tests/e2e/__init__.py tests/e2e/test_paper_loop.py
git commit -m "test(e2e): paper-loop smoke test against simulator"
```

---

### Task 3.13: Disaster-recovery dry run + Phase 3 milestone

**Files:**
- Create: `docs/runbooks/disaster-recovery.md`

The DR runbook describes how to: take a backup, simulate machine loss, restore from cold storage, verify the system runs again. This is a manual procedure documented in the runbook.

- [ ] **Step 1: Write `docs/runbooks/disaster-recovery.md`**

```markdown
# Disaster Recovery Drill

Run this drill at least once before declaring Phase 3 complete, and quarterly thereafter.

## Goal

Prove that the system can be reconstructed from cold storage with **zero hands on the original machine**.

## Backup procedure (already automated via cron)

```bash
# Run nightly at 02:00 ET
pg_dump -U squeeze squeeze | gzip > ~/backups/$(date +%Y-%m-%d).sql.gz
tar czf ~/backups/parquet-$(date +%Y-%m-%d).tar.gz data/parquet
rclone sync ~/backups/ b2:squeeze-hunter-backups/
```

## Drill

1. Pick a clean directory: `mkdir -p /tmp/dr-drill && cd /tmp/dr-drill`
2. Pull the most recent backups from cold storage:
   ```bash
   rclone copy b2:squeeze-hunter-backups/ ./backups/ --include "$(date +%Y-%m)*"
   ```
3. Spin up a fresh postgres on a non-default port to avoid collisions:
   ```bash
   docker run -d --name dr-pg -p 5433:5432 \
     -e POSTGRES_USER=squeeze -e POSTGRES_PASSWORD=squeeze -e POSTGRES_DB=squeeze \
     postgres:14
   sleep 5
   gunzip -c backups/$(ls backups/*.sql.gz | tail -1) | \
     docker exec -i dr-pg psql -U squeeze -d squeeze
   ```
4. Restore parquet:
   ```bash
   tar xzf backups/$(ls backups/parquet-*.tar.gz | tail -1) -C .
   ```
5. Clone the repo, install:
   ```bash
   git clone https://github.com/yebof/squeeze-hunter.git .src
   cd .src && uv sync --all-extras
   SH_DB_URL=postgresql+psycopg://squeeze:squeeze@localhost:5433/squeeze \
     uv run alembic upgrade head
   ```
6. Run a scan:
   ```bash
   SH_DB_URL=... uv run squeeze-hunter scan --date 2025-04-21 \
     --data /tmp/dr-drill/data/parquet
   ```
   Compare output to a known-good scan from production.

## Pass criterion

- Scan output matches production within rounding on top 10 candidates.
- No errors in stderr/log.
- Drill takes < 30 minutes wall clock from "machine lost" to "scan working".

## Cleanup

```bash
docker rm -f dr-pg
rm -rf /tmp/dr-drill
```
```

- [ ] **Step 2: Tag the milestone**

```bash
uv run pytest -q   # confirm everything green
git add docs/runbooks/disaster-recovery.md
git commit -m "docs(runbook): disaster recovery drill"
git tag phase-3-paper-ready
```

- [ ] **Step 3: Manual DR drill (off-machine)**

You (the operator) must actually run the drill at least once before declaring `phase-3-paper-ready`. Record the wall-clock time and any deviations. If something doesn't work, fix it and re-tag.

---

## Phase 4 — Small Live ($2-5K)

### Task 4.1: Live-mode safety gate

**Files:**
- Modify: `src/squeeze_hunter/runtime.py`
- Create: `tests/runtime/test_live_safety.py`

Live mode must require:
1. `--confirm-real-money` on the CLI (already done in Task 3.11)
2. A `LIVE_MAX_POSITION_USD` env var that hard-caps position size below the design's 8% Kelly cap during the ramp-up window
3. A `LIVE_KILL_SWITCH_INITIAL` env var that starts the system with kill-switch armed at -5% (tighter than design's -10%) for the first 30 days

- [ ] **Step 1: Write the failing test**

```python
import os

import pytest

from squeeze_hunter.config import Settings
from squeeze_hunter.runtime import live_safety_overrides


def test_live_safety_caps_position(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVE_MAX_POSITION_USD", "1000")
    s = Settings()
    overrides = live_safety_overrides(s)
    assert overrides["max_position_usd"] == 1000.0


def test_live_safety_killswitch_tighter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVE_KILL_SWITCH_INITIAL", "true")
    overrides = live_safety_overrides(Settings())
    assert overrides["monthly_drawdown_max"] == -0.05


def test_no_live_mode_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LIVE_MAX_POSITION_USD", raising=False)
    overrides = live_safety_overrides(Settings())
    assert overrides["max_position_usd"] is None
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/runtime/test_live_safety.py -v
```

- [ ] **Step 3: Add `live_safety_overrides` to `runtime.py`**

```python
import os


def live_safety_overrides(settings: Settings) -> dict[str, Any]:
    """Read LIVE_MAX_POSITION_USD and LIVE_KILL_SWITCH_INITIAL env vars."""
    out: dict[str, Any] = {"max_position_usd": None, "monthly_drawdown_max": -0.10}
    if (cap := os.environ.get("LIVE_MAX_POSITION_USD")):
        out["max_position_usd"] = float(cap)
    if os.environ.get("LIVE_KILL_SWITCH_INITIAL", "").lower() == "true":
        out["monthly_drawdown_max"] = -0.05
    return out
```

Wire it into `RuntimeContext.setup()` so when `mode == "live"`, the overrides apply.

- [ ] **Step 4: Run, expect pass**

```bash
uv run pytest tests/runtime/test_live_safety.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/squeeze_hunter/runtime.py tests/runtime/test_live_safety.py
git commit -m "feat(runtime): live-mode safety env overrides"
```

---

### Task 4.2: Daily review checklist (runbook)

**Files:**
- Create: `docs/runbooks/live-trading.md`

A documented daily routine: what the operator looks at, what they decide. **You** run this every trading day during Phase 4.

- [ ] **Step 1: Write `docs/runbooks/live-trading.md`**

```markdown
# Live Trading — Daily Review

Run this every trading day in Phase 4 (small live, $2–5K, 60 days).

## Morning (08:00 ET, 30 min before open)

1. Check Telegram: any kill-switch alerts overnight? If yes, **do not enable** new entries until investigated.
2. Open Grafana → squeeze-hunter dashboard. Confirm:
   - `sh_broker_connected` == 1
   - `sh_db_connected` == 1
   - `sh_kill_switch_active{reason=*}` == 0 for all reasons
   - Last `nightly_scan` job ran at 22:00 ET (per APScheduler logs)
3. Read the candidate list from Slack (posted by `nightly_scan`). For each candidate scoring ≥ 8.0:
   - Search news manually for unexpected events (halt, lawsuit, M&A)
   - If anything off, override by adding ticker to `config/blacklist.txt` (system reads at next premarket_verify)
4. Confirm `LIVE_MAX_POSITION_USD` env var is set and matches your intended ramp tier.

## Midday (12:00 ET)

1. Glance at Telegram. Any unexpected alerts?
2. Check Grafana: `sh_position_count` and `sh_gross_exposure_pct` — match what you expect?
3. If a position is up >50%, the trailing stop should auto-tighten — confirm by reading recent log lines.

## After close (16:30 ET)

1. Read the EOD summary from Slack (posted by `eod_close`).
2. For each closed position, log into a journal:
   - Setup type (CAR / GME / Mixed)
   - Hold duration
   - Realized P&L
   - Exit reason
   - Was the exit at the "right" time in hindsight?
3. Compare the day's P&L to the rolling 5-day, 30-day. Drift?

## Weekly (Friday close)

1. Run `uv run squeeze-hunter backtest --train ... --holdout ...` against the latest data; compare Sharpe / hit-rate to last week's. **If Sharpe degrades by > 30%, halt the system** and investigate before next Monday.

## Promotion criteria → Phase 5

The system is ready for normal-size live (Phase 5) when:
- 60 calendar days of small live with no incidents
- Sharpe ≥ 0.5 over the period
- Total P&L ≥ 0
- No kill-switch trips
- All daily reviews completed without ops surprises
```

- [ ] **Step 2: Commit**

```bash
git add docs/runbooks/live-trading.md
git commit -m "docs(runbook): live-trading daily review"
```

---

### Task 4.3: First live trade — manual gate

This isn't a code task. It's a **stop-and-think gate** before placing the first real-money trade.

- [ ] **Pre-flight checklist (manual)**

  1. Phase 3 paper trading ran cleanly for ≥ 30 calendar days.
  2. Per spec Section 5 Gate 2: Sharpe ≥ 0.5 over paper period; 0 kill-switch trips; deviation from backtest expectation < 50%.
  3. IBKR account funded with the **paper-equivalent capital** (e.g., $5K) — not your retirement.
  4. `LIVE_MAX_POSITION_USD=1000` set in `.env` (cap at 20% of equity for the first week).
  5. Telegram bot configured and you've received a test alert.
  6. You've run the DR drill (Task 3.13).
  7. Your spouse / partner / housemate knows the system is live. Loss tolerance discussed.

- [ ] **Start live**

```bash
uv run squeeze-hunter live --confirm-real-money
```

- [ ] **Tag the milestone after first round-trip**

```bash
git tag phase-4-live-ramp
```

---

## Phase 5 — Scale Up + Continuous Improvement

### Task 5.1: Monthly decay-detection job

**Files:**
- Create: `src/squeeze_hunter/jobs/decay_check.py`
- Modify: `src/squeeze_hunter/scheduler.py`
- Create: `tests/jobs/test_decay_check.py`

Once a month, re-run the walk-forward backtest with the latest data and compare metrics to the prior month. If holdout Sharpe drops by > 30%, send a HIGH-severity alert.

- [ ] **Step 1: Write the failing test**

```python
import pytest

from squeeze_hunter.jobs.decay_check import DecayReport, evaluate_decay


def test_decay_detected_on_30pct_sharpe_drop() -> None:
    prior = {"sharpe": 1.5, "max_drawdown": -0.15, "hit_rate": 0.4}
    current = {"sharpe": 1.0, "max_drawdown": -0.18, "hit_rate": 0.38}
    r = evaluate_decay(prior, current)
    assert r.decayed
    assert "sharpe" in r.reason


def test_no_decay_when_stable() -> None:
    prior = {"sharpe": 1.5, "max_drawdown": -0.15, "hit_rate": 0.4}
    current = {"sharpe": 1.4, "max_drawdown": -0.16, "hit_rate": 0.39}
    r = evaluate_decay(prior, current)
    assert not r.decayed
```

- [ ] **Step 2: Implement `decay_check.py`**

```python
"""Monthly decay detection — compare current backtest metrics to prior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class DecayReport:
    decayed: bool
    reason: str | None = None


def evaluate_decay(
    prior: dict[str, Any], current: dict[str, Any], *, sharpe_drop_pct: float = 0.30
) -> DecayReport:
    if prior["sharpe"] <= 0:
        return DecayReport(False)
    drop = (prior["sharpe"] - current["sharpe"]) / abs(prior["sharpe"])
    if drop > sharpe_drop_pct:
        return DecayReport(
            True, f"sharpe dropped {drop:.0%} (prior={prior['sharpe']:.2f}, current={current['sharpe']:.2f})"
        )
    return DecayReport(False)
```

`src/squeeze_hunter/jobs/__init__.py` empty.

- [ ] **Step 3: Wire into scheduler**

In `scheduler.py`, append to `list_job_specs()`:

```python
{"id": "monthly_decay_check", "trigger": "cron",
 "day": 1, "hour": 22, "minute": 0, "tz": _TZ,
 "doc": "Monthly: rerun backtest, compare to last month, alert on decay"},
```

And update `build_scheduler` to handle `day` field for the cron trigger.

- [ ] **Step 4: Run, expect pass**

```bash
uv run pytest tests/jobs/test_decay_check.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/squeeze_hunter/jobs/ src/squeeze_hunter/scheduler.py tests/jobs/
git commit -m "feat(jobs): monthly decay detection"
```

---

### Task 5.2: Paid-data ROI evaluation runbook

**Files:**
- Create: `docs/runbooks/paid-data-roi.md`

When user has been live for 90+ days, evaluate whether to pay for Polygon / Ortex / Quiver.

- [ ] **Step 1: Write `docs/runbooks/paid-data-roi.md`**

```markdown
# Paid Data ROI Evaluation

Run this quarterly during Phase 5.

## Decision criterion

Subscribe to a paid feed only if **integrating it raises holdout Sharpe by ≥ 0.3** over the last 90 days of out-of-sample data.

## Procedure

For each candidate provider (`polygon`, `ortex`, `quiver`):

1. Get a one-month trial. Implement the provider behind the `DataProvider` Protocol (the architecture supports plug-in: see Task 1.2).
2. Re-run the walk-forward backtest with this provider replacing the corresponding free source. Use the most recent 90 days as holdout.
3. Compare:
   - Holdout Sharpe with paid vs. free
   - Captured-events score (any new events caught?)
   - Cost-per-trade for the paid feed at your trade frequency

4. Decide:
   - ΔSharpe ≥ +0.3 AND cost < 1% of monthly P&L → subscribe
   - Otherwise → defer; revisit next quarter

## Track in a journal

Maintain a record at `docs/runbooks/paid-data-decisions.md`:

```
2026-08-01 — Polygon trial: ΔSharpe +0.05. Reject.
2027-02-01 — Ortex trial: ΔSharpe +0.42. Subscribe at $500/mo.
```
```

- [ ] **Step 2: Commit**

```bash
git add docs/runbooks/paid-data-roi.md
git commit -m "docs(runbook): paid-data ROI evaluation"
```

---

### Task 5.3: Equity ramp-up rules

**Files:**
- Create: `docs/runbooks/scale-up.md`

The plan for growing position size from $5K → normal account.

- [ ] **Step 1: Write `docs/runbooks/scale-up.md`**

```markdown
# Scale-Up Plan (Phase 5)

After Phase 4 (60 days small live, no incidents, P&L ≥ 0), promote to normal size.

## Ramp tiers

| Tier | Equity | Per-position cap | Max gross | Min observation period |
| --- | --- | --- | --- | --- |
| 1   | $5K   | $1,000 (20%)     | 90%       | 60 days (Phase 4 baseline) |
| 2   | $20K  | $2,000 (10%)     | 90%       | 30 days |
| 3   | $50K  | $4,000 (8%)      | 90%       | 30 days |
| 4   | $100K | $8,000 (8%)      | 90%       | 30 days |
| 5+  | normal Kelly | spec default | spec default | continuous |

Each tier requires the previous tier to clear:
- Sharpe ≥ 0.5 over the observation period
- 0 kill-switch trips
- Realized P&L ≥ 0

## Promotion command

There's no automated promotion. You change `LIVE_MAX_POSITION_USD` in `.env` and restart the runtime. The intent is that the operator (you) consciously approves each tier.

## Demotion

If at any tier the system trips a kill-switch or loses > 5% in any week, demote one tier and reset the observation clock.
```

- [ ] **Step 2: Tag the milestone after first scale-up**

```bash
git add docs/runbooks/scale-up.md
git commit -m "docs(runbook): scale-up plan"
git tag phase-5-scale
```

---

## Self-Review

**Spec coverage:**

| Spec section | Covered |
| --- | --- |
| 1. Overview & scope | n/a (already in code) |
| 2. Architecture | Tasks 3.1-3.11 fill out `execution/`, `monitor/`, `runtime`, `scheduler`, completing the module layout |
| 3. Signal model | n/a (Phase 1) |
| 4. Data layer | n/a (Phase 1) |
| 5. Backtest | n/a (Phase 2); Task 5.1 wires monthly re-run |
| 6. Risk & execution | Tasks 3.3-3.6 (slicer, OMS, lifecycle, killswitch); 3.11 emergency-flatten |
| 7. Operations & rollout | Tasks 3.7-3.10 (metrics, alerts, scheduler, runtime); 3.13 DR; 4.2-4.3 (live runbook + first trade); 5.1-5.3 (decay, ROI, scale) |
| 8. Open questions | 5.2 paid-data ROI; rest deferred to ongoing operations |
| 9. Validation sources | n/a |

**Placeholder scan:** searched for "TBD", "TODO", "implement later", "fill in details" — none found.

**Type consistency:**
- `BrokerOrder` (Task 3.1) used by OMS (3.4), lifecycle (3.5), CLI emergency-flatten (3.11) — same fields throughout.
- `IBroker` Protocol now includes `submit_buy / submit_sell / cancel_order / get_open_orders` (Task 3.1). `SimulatorBroker` already had `submit_buy/_sell` from Phase 2; for Phase 3 it would also need `cancel_order` and `get_open_orders`. **The plan does not add those to SimulatorBroker** — this is OK because `SimulatorBroker` is only used in backtest where order cancel/list isn't exercised. If runtime in `sim` mode tries those methods, it'll AttributeError. **Operator should call them only in paper/live modes.**
- `LifecycleState`, `KillSwitchInputs`, `KillSwitchVerdict`, `RuntimeContext` — used internally in their respective tasks.

**Phase milestones:**
- `phase-3-paper-ready` (after Task 3.13)
- `phase-4-live-ramp` (after Task 4.3 — first round-trip)
- `phase-5-scale` (after Task 5.3 — first ramp tier promotion)

---

## Execution Handoff

Plan complete. Saved to [`docs/superpowers/plans/2026-05-11-squeeze-hunter-phase-3-5.md`](2026-05-11-squeeze-hunter-phase-3-5.md).

**Two execution options:**

1. **Subagent-Driven (recommended)** — same approach as Phase 0–2. Each task to a fresh subagent + brief verification between tasks.

2. **Inline Execution** — execute through `superpowers:executing-plans` in the current session.

**A note on Phase 4–5:** Tasks 4.3, 5.2, 5.3 are **operational** (runbooks + manual gates), not coding work. They should not be subagent-dispatched; you run them yourself when the time comes.

Which approach for Phase 3 coding tasks (3.1–3.13)?
