"""Database session factory."""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

_engine_instance: Engine | None = None
# R3.6: serialize first-time engine creation across threads. Without the lock,
# two concurrent get_engine() calls could each see None and both call
# create_engine, leaking a connection pool. Harmless in the current asyncio
# single-thread setup but a time bomb if threads are added later.
_engine_lock = threading.Lock()


def get_engine() -> Engine:
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            # Re-check inside the lock — another thread may have created it
            # while we were waiting (double-checked locking).
            if _engine_instance is None:
                _engine_instance = create_engine(
                    os.environ.get(
                        "SH_DB_URL",
                        "postgresql+psycopg://squeeze:squeeze@localhost:5432/squeeze",
                    ),
                    future=True,
                    pool_pre_ping=True,
                )
    return _engine_instance


def _make_session() -> Session:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)()


@contextmanager
def session_scope() -> Iterator[Session]:
    s = _make_session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
