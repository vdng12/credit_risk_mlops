"""Local parquet write-through archive.

Every entity lands here regardless of lane - it is the immutable local
archive (batch consumers read history from it, object-store consumers read
S3). Layout: ``<output_dir>/<run_id>/<entity>_<yyyy-mm>.parquet``.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


class LocalParquetSink:
    """Zero-config local sink: write-through parquet per (entity, month)."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        df: pl.DataFrame,
        entity_name: str,
        snapshot_date: str | None = None,
        run_id: str = "default",
    ) -> str:
        date_part = f"_{snapshot_date}" if snapshot_date else ""
        path = self.output_dir / run_id / f"{entity_name}{date_part}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
        return str(path)
