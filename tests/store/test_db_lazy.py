"""A5: store.db must not create an engine at import time."""

from __future__ import annotations

import importlib
import sys


def test_store_db_does_not_create_engine_at_import() -> None:
    """A5: importing store.db must not connect to or create the engine."""
    # Remove cached module so the test exercises fresh import
    for mod in list(sys.modules):
        if mod.startswith("squeeze_hunter.store.db"):
            del sys.modules[mod]
    db = importlib.import_module("squeeze_hunter.store.db")
    # No call to get_engine yet → no instance
    assert db._engine_instance is None
    # Calling get_engine creates it lazily
    eng = db.get_engine()
    assert eng is not None
    assert db._engine_instance is eng
    # Idempotent
    assert db.get_engine() is eng
