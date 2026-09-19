"""Per-dataset freshness stamps (P6), stored next to the parquet cache.

`as_of` is the newest data point the dataset holds (last bar timestamp,
latest settlement date, last report date); `recorded_at` is when the ingest
ran. The premarket gate compares `as_of` with its budget in
`settings.data`; the EOD job uses `recorded_at` to decide whether a slow
dataset (FINRA, earnings) is due for a refresh.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from squeeze_hunter.logging_setup import get_logger

log = get_logger("ingest.freshness")

FRESHNESS_FILE = "_freshness.json"


def _path(root: Path) -> Path:
    return Path(root) / FRESHNESS_FILE


def read_freshness(root: Path) -> dict[str, dict[str, Any]]:
    path = _path(root)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        log.warning("freshness_unreadable", path=str(path), err=str(e))
        return {}
    return data if isinstance(data, dict) else {}


def record_freshness(
    root: Path,
    dataset: str,
    *,
    as_of: datetime,
    rows: int,
    now: datetime | None = None,
    note: str | None = None,
) -> None:
    stamps = read_freshness(root)
    stamps[dataset] = {
        "as_of": as_of.astimezone(UTC).isoformat(),
        "recorded_at": (now or datetime.now(UTC)).astimezone(UTC).isoformat(),
        "rows": int(rows),
        "note": note,
    }
    path = _path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(stamps, fh, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _parse(ts: Any) -> datetime | None:
    if not isinstance(ts, str):
        return None
    try:
        parsed = datetime.fromisoformat(ts)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def dataset_age_days(root: Path, dataset: str, now: datetime) -> float | None:
    """Days since the dataset's newest data point; None if never recorded."""
    as_of = _parse(read_freshness(root).get(dataset, {}).get("as_of"))
    if as_of is None:
        return None
    return (now.astimezone(UTC) - as_of).total_seconds() / 86_400


def recorded_age_days(root: Path, dataset: str, now: datetime) -> float | None:
    """Days since the dataset was last ingested; None if never."""
    recorded = _parse(read_freshness(root).get(dataset, {}).get("recorded_at"))
    if recorded is None:
        return None
    return (now.astimezone(UTC) - recorded).total_seconds() / 86_400
