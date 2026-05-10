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
        self: ParquetCache, domain: str, partition_key: str, df: pd.DataFrame
    ) -> None:
        path = self._path(domain, partition_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.dedup_keys:
            df = df.drop_duplicates(self.dedup_keys, keep="last")
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)

    def read_partition(self: ParquetCache, domain: str, partition_key: str) -> pd.DataFrame:
        path = self._path(domain, partition_key)
        if not path.exists():
            return pd.DataFrame()
        return pq.read_table(path).to_pandas()

    def append_partition(
        self: ParquetCache, domain: str, partition_key: str, df: pd.DataFrame
    ) -> None:
        existing = self.read_partition(domain, partition_key)
        merged = pd.concat([existing, df], ignore_index=True) if not existing.empty else df
        if self.dedup_keys:
            merged = merged.drop_duplicates(self.dedup_keys, keep="last")
        self.write_partition(domain, partition_key, merged)
