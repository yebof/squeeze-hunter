"""Paper-trading broker — same client as IBKRBroker, locked to IBKR's paper port."""

from __future__ import annotations

import os
from dataclasses import dataclass

from squeeze_hunter.broker.ibkr import IBKRBroker
from squeeze_hunter.logging_setup import get_logger

log = get_logger("broker.paper")


@dataclass
class PaperBroker(IBKRBroker):
    name: str = "ibkr-paper"

    def __post_init__(self: PaperBroker) -> None:
        # R6.M3: check the env FIRST, then set port. IBKRBroker's
        # default_factory has already evaluated IBKR_PORT at dataclass init
        # time (could be 8888); the env-mismatch check raises before we
        # overwrite, which gives a clearer error path.
        # R5.C4: guard the env-var int() with try/except so a malformed
        # IBKR_PORT (e.g., "abc") doesn't raise cryptic ValueError.
        raw = os.environ.get("IBKR_PORT")
        if raw:
            try:
                env_port = int(raw)
            except ValueError:
                log.warning("ibkr_port_invalid_paper_using_7497", value=raw)
                env_port = 7497
            if env_port != 7497:
                raise RuntimeError(f"PaperBroker refusing to connect to non-paper port {env_port}")
        # Paper is always 7497 regardless of env. Caller can't accidentally hit live.
        self.port = 7497
        super().__post_init__()
