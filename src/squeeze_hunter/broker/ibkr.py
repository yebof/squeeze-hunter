"""IBKRBroker — ib-async wrapper. Phase 0 only implements connect / quote / health."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field

from ib_async import IB, Stock

from squeeze_hunter.broker.base import BrokerHealth, Quote
from squeeze_hunter.logging_setup import get_logger

log = get_logger("broker.ibkr")

_IBKR_HOST = os.environ.get("IBKR_HOST", "127.0.0.1")
_IBKR_PORT = int(os.environ.get("IBKR_PORT", "7497"))
_IBKR_CLIENT_ID = int(os.environ.get("IBKR_CLIENT_ID", "42"))
_IBKR_ACCOUNT = os.environ.get("IBKR_ACCOUNT", "")


@dataclass
class IBKRBroker:
    name: str = "ibkr"
    host: str = field(default_factory=lambda: _IBKR_HOST)
    port: int = field(default_factory=lambda: _IBKR_PORT)
    client_id: int = field(default_factory=lambda: _IBKR_CLIENT_ID)
    account: str = field(default_factory=lambda: _IBKR_ACCOUNT)

    def __post_init__(self: IBKRBroker) -> None:
        self._ib = IB()

    async def connect(self: IBKRBroker) -> None:
        log.info("connecting", host=self.host, port=self.port)
        await self._ib.connectAsync(self.host, self.port, clientId=self.client_id)
        log.info("connected", server_version=self._ib.client.serverVersion())

    async def disconnect(self: IBKRBroker) -> None:
        self._ib.disconnect()

    async def fetch_quote(self: IBKRBroker, ticker: str) -> Quote:
        contract = Stock(ticker, "SMART", "USD")
        await self._ib.qualifyContractsAsync(contract)
        ticker_data = self._ib.reqMktData(contract, "", snapshot=True, regulatorySnapshot=False)
        # Wait for snapshot — ib-async populates fields as updates arrive
        for _ in range(40):
            await asyncio.sleep(0.25)
            if ticker_data.last is not None or ticker_data.bid is not None:
                break
        return Quote(
            ticker=ticker,
            bid=float(ticker_data.bid or 0.0),
            ask=float(ticker_data.ask or 0.0),
            last=float(ticker_data.last or ticker_data.close or 0.0),
            timestamp_ns=time.time_ns(),
        )

    async def health(self: IBKRBroker) -> BrokerHealth:
        return BrokerHealth(
            connected=self._ib.isConnected(),
            last_ping_ms=0,
            account=self.account,
        )
