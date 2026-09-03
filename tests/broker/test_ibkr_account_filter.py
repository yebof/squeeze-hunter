"""Round-12: get_position_qty must only count the configured account."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from squeeze_hunter.broker.ibkr import IBKRBroker


def _fake_ib(rows: list[tuple[str, int]]) -> MagicMock:
    fake = MagicMock()

    async def req_positions() -> None:
        return None

    def positions() -> list[MagicMock]:
        out = []
        for account, qty in rows:
            pos = MagicMock()
            pos.contract.symbol = "GME"
            pos.account = account
            pos.position = qty
            out.append(pos)
        return out

    fake.reqPositionsAsync = req_positions
    fake.positions = positions
    return fake


@pytest.mark.asyncio
async def test_get_position_qty_filters_by_configured_account() -> None:
    broker = IBKRBroker(client_id=99, account="DU111")
    broker._ib = _fake_ib([("DU111", 100), ("U222", 50)])
    assert await broker.get_position_qty("GME") == 100


@pytest.mark.asyncio
async def test_get_position_qty_sums_all_accounts_when_none_configured() -> None:
    broker = IBKRBroker(client_id=99, account="")
    broker._ib = _fake_ib([("DU111", 100), ("U222", 50)])
    assert await broker.get_position_qty("GME") == 150
