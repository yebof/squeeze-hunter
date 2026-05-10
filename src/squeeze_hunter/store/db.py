"""Database session factory."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

_engine_instance: Engine | None = None


def get_engine() -> Engine:
    global _engine_instance
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
