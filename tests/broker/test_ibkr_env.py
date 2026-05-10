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
