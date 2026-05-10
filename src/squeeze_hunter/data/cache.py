"""Parquet on-disk cache, partitioned by `domain/partition_key/`."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class ParquetCache:
    root: Path
    dedup_keys: list[str] = field(default_factory=list)

    def _path(self: ParquetCache, domain: str, partition_key: str) -> Path:
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
