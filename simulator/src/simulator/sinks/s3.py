"""S3 / SeaweedFS sink for cold-facts (immutable parquet snapshots)."""

from __future__ import annotations

import polars as pl


class S3Sink:
    """Writes parquet to an S3-compatible object store (SeaweedFS)."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        region: str,
    ) -> None:
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket = bucket
        self.region = region

    def write(
        self,
        df: pl.DataFrame,
        entity_name: str,
        snapshot_date: str | None = None,
        run_id: str = "default",
    ) -> str:
        date_part = f"_{snapshot_date}" if snapshot_date else ""
        key = f"{run_id}/{entity_name}{date_part}.parquet"
        s3_path = f"s3://{self.bucket}/{key}"
        storage_options = {
            "aws_endpoint_url": self.endpoint,
            "aws_access_key_id": self.access_key,
            "aws_secret_access_key": self.secret_key,
            "aws_region": self.region,
        }
        df.write_parquet(s3_path, storage_options=storage_options)
        return s3_path
