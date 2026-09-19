"""P6 — per-dataset freshness stamps next to the parquet cache."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from squeeze_hunter.ingest.freshness import dataset_age_days, read_freshness, record_freshness


def test_record_and_age(tmp_path: Path) -> None:
    now = datetime(2026, 6, 10, 22, 0, tzinfo=UTC)
    record_freshness(tmp_path, "bars", as_of=now - timedelta(days=2), rows=18, now=now)
    record_freshness(tmp_path, "short_interest", as_of=now - timedelta(days=30), rows=400, now=now)
    stamps = read_freshness(tmp_path)
    assert stamps["bars"]["rows"] == 18
    assert dataset_age_days(tmp_path, "bars", now) == pytest.approx(2.0)
    assert dataset_age_days(tmp_path, "short_interest", now) == pytest.approx(30.0)
    assert dataset_age_days(tmp_path, "earnings", now) is None


def test_missing_or_corrupt_file_reads_as_empty(tmp_path: Path) -> None:
    assert read_freshness(tmp_path) == {}
    (tmp_path / "_freshness.json").write_text("{nope")
    assert read_freshness(tmp_path) == {}
