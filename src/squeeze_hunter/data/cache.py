"""Parquet on-disk cache, partitioned by `domain/partition_key/`."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# CDX-P2-6: domain + partition_key are concatenated into a filesystem path and
# both can carry operator-edited / external strings (universe tickers flow
# straight into partition keys). Restrict to a safe charset and explicitly
# reject the parent-dir token so a key like "../../etc/foo" can't read or
# write parquet outside the cache root. The allowed set covers everything the
# codebase actually uses: tickers (incl. dotted BRK.B), ISO dates
# (2024-05-13), and `{ticker}__{date}` composite keys.
_SAFE_PARTITION_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate_path_component(kind: str, value: str) -> None:
    if not value or not _SAFE_PARTITION_RE.match(value) or ".." in value:
        raise ValueError(
            f"unsafe {kind} {value!r}: must match [A-Za-z0-9._-]+ and contain no '..' "
            "(path-traversal guard)"
        )


@dataclass
class ParquetCache:
    root: Path
    dedup_keys: list[str] = field(default_factory=list)

    def _path(self: ParquetCache, domain: str, partition_key: str) -> Path:
        _validate_path_component("domain", domain)
        _validate_path_component("partition_key", partition_key)
        return self.root / domain / f"{partition_key}.parquet"

    def write_partition(
        self: ParquetCache,
        domain: str,
        partition_key: str,
        df: pd.DataFrame,
        *,
        dedup_keys: list[str] | None = None,
    ) -> None:
        path = self._path(domain, partition_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = dedup_keys if dedup_keys is not None else self.dedup_keys
        if keys:
            df = df.drop_duplicates(keys, keep="last")
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)

    def read_partition(self: ParquetCache, domain: str, partition_key: str) -> pd.DataFrame:
        path = self._path(domain, partition_key)
        if not path.exists():
            return pd.DataFrame()
        return pq.read_table(path).to_pandas()

    def append_partition(
        self: ParquetCache,
        domain: str,
        partition_key: str,
        df: pd.DataFrame,
        *,
        dedup_keys: list[str] | None = None,
    ) -> None:
        existing = self.read_partition(domain, partition_key)
        merged = pd.concat([existing, df], ignore_index=True) if not existing.empty else df
        keys = dedup_keys if dedup_keys is not None else self.dedup_keys
        if keys:
            merged = merged.drop_duplicates(keys, keep="last")
        self.write_partition(domain, partition_key, merged, dedup_keys=None)
