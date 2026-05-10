"""Database session factory."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


def _engine() -> Engine:
    url = os.environ.get(
        "SH_DB_URL",
        "postgresql+psycopg://squeeze:squeeze@localhost:5432/squeeze",
    )
    return create_engine(url, future=True, pool_pre_ping=True)


SessionLocal = sessionmaker(bind=_engine(), expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
