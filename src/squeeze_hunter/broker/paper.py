"""Paper-trading broker — same client as IBKRBroker, locked to IBKR's paper port."""

from __future__ import annotations

import os
from dataclasses import dataclass

from squeeze_hunter.broker.ibkr import IBKRBroker


@dataclass
class PaperBroker(IBKRBroker):
    name: str = "ibkr-paper"

    def __post_init__(self: PaperBroker) -> None:
        # Paper is always 7497 regardless of env. Caller can't accidentally hit live.
        self.port = 7497
        if os.environ.get("IBKR_PORT") and int(os.environ["IBKR_PORT"]) != 7497:
            raise RuntimeError(
                f"PaperBroker refusing to connect to non-paper port {os.environ['IBKR_PORT']}"
            )
        super().__post_init__()
