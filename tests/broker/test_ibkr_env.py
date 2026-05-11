"""A3: IBKR_PORT / IBKR_CLIENT_ID env-var parsing must not crash import."""

from __future__ import annotations

import importlib
import os
import sys


def test_ibkr_module_imports_with_malformed_env() -> None:
    """A3: a bad IBKR_PORT/IBKR_CLIENT_ID env var must not crash the import.
    Previously: int() conversion at module load → ValueError → import failure.
    """
    os.environ["IBKR_PORT"] = "abc"
    try:
        for mod in list(sys.modules):
            if mod.startswith("squeeze_hunter.broker.ibkr"):
                del sys.modules[mod]
        # This must not raise
        importlib.import_module("squeeze_hunter.broker.ibkr")
    finally:
        del os.environ["IBKR_PORT"]
        for mod in list(sys.modules):
            if mod.startswith("squeeze_hunter.broker.ibkr"):
                del sys.modules[mod]


def test_paper_broker_handles_malformed_ibkr_port() -> None:
    """R5.C4 regression: PaperBroker.__post_init__ used a bare int() on
    IBKR_PORT env. With IBKR_PORT=abc the constructor raised ValueError
    instead of the intended 'refusing non-paper port' RuntimeError (or
    just safely defaulting to 7497). Symmetric to the IBKRBroker fix.
    """
    os.environ["IBKR_PORT"] = "abc"
    try:
        # Clear cached module so the dataclass picks up env afresh
        for mod in list(sys.modules):
            if mod.startswith("squeeze_hunter.broker.paper") or mod.startswith(
                "squeeze_hunter.broker.ibkr"
            ):
                del sys.modules[mod]
        from squeeze_hunter.broker.paper import PaperBroker

        # Should not raise — malformed env logs a warning and falls back to 7497
        b = PaperBroker(client_id=99)
        assert b.port == 7497
    finally:
        del os.environ["IBKR_PORT"]
        for mod in list(sys.modules):
            if mod.startswith("squeeze_hunter.broker"):
                del sys.modules[mod]
