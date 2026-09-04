"""Round-12: the backtest feeds REAL liquidity and price inputs to the gates.

`runner.py` used to pass ADV20 = 1e9 for every ticker and never applied the
YAML universe price floor, so the liquidity and universe gates could not fire
in a backtest and Gate 1 never validated them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from squeeze_hunter.data.cache import ParquetCache
from tests.backtest.test_runner_session_alignment import _bar, _run, _sessions, _write


def _buys(log):
    return log if log.empty else log[log["side"] == "buy"]


@pytest.mark.asyncio
async def test_runner_rejects_entries_below_the_universe_price_floor(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path)
    d = _sessions("2024-06-03", 4)
    _write(cache, [_bar(t, 3.0, 3.1, 2.9, 3.0) for t in d])  # below min_price 5.0
    log = await _run(cache, "2024-06-03", "2024-06-06")
    assert _buys(log).empty, _buys(log).to_dict("records")


@pytest.mark.asyncio
async def test_runner_rejects_entries_without_liquidity(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path)
    d = _sessions("2024-06-03", 4)
    # $100 x 10 shares/day = $1,000 ADV; any sensible position needs 100x its size.
    _write(cache, [_bar(t, 100, 101, 99, 100, v=10) for t in d])
    log = await _run(cache, "2024-06-03", "2024-06-06")
    assert _buys(log).empty, _buys(log).to_dict("records")
