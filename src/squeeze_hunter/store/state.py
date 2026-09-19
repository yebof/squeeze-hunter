"""Runtime state snapshots (P2 of the architecture-hardening plan).

The book, pending orders, killswitch lockout and telemetry used to live only
in memory: a restart orphaned real exposure at the broker and erased the
7-day cooldown. `JsonStateStore` writes one JSON document atomically (temp
file + rename) after every job; Postgres can replace it behind the same
protocol later.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from squeeze_hunter.logging_setup import get_logger

log = get_logger("store.state")

SNAPSHOT_VERSION = 1


class StateStore(Protocol):
    def save(self, snapshot: dict[str, Any]) -> None: ...
    def load(self) -> dict[str, Any] | None: ...


class JsonStateStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def save(self, snapshot: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(snapshot)
        payload.setdefault("version", SNAPSHOT_VERSION)
        payload["saved_at"] = datetime.now(UTC).isoformat()
        # Atomic: write a sibling temp file, fsync, then rename over the target
        # so a crash mid-write can never leave a truncated snapshot.
        fd, tmp = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=1, sort_keys=True, default=str)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    def load(self) -> dict[str, Any] | None:
        if not self.path.is_file():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            # Keep the bad file for forensics; start clean rather than crash
            # the runtime — startup reconciliation rebuilds the book.
            aside = self.path.with_name(
                f"{self.path.name}.corrupt-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
            )
            with contextlib.suppress(OSError):
                os.replace(self.path, aside)
            log.error(
                "state_snapshot_unreadable", path=str(self.path), moved_to=str(aside), err=str(e)
            )
            return None
        if not isinstance(data, dict):
            log.error("state_snapshot_not_an_object", path=str(self.path))
            return None
        return data
