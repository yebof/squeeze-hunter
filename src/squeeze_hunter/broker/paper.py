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
        # Paper is always 7497 regardless of env. Caller can't accidentally hit live.
        self.port = 7497
        # R5.C4: guard the env-var int() with try/except so a malformed
        # IBKR_PORT (e.g., "abc") raises a clear error rather than a cryptic
        # ValueError. Mirrors the IBKRBroker module-load guard.
        raw = os.environ.get("IBKR_PORT")
        if raw:
            try:
                env_port = int(raw)
            except ValueError:
                log.warning("ibkr_port_invalid_paper_using_7497", value=raw)
                env_port = 7497
            if env_port != 7497:
                raise RuntimeError(f"PaperBroker refusing to connect to non-paper port {env_port}")
        super().__post_init__()
