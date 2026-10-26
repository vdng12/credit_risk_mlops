"""Output sinks: file, S3/SeaweedFS, Postgres, Redpanda."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import polars as pl


class SinkManager:
    """Write DataFrames to configured destinations."""

    def __init__(self, config) -> None:
        self.cfg = config
        Path(self.cfg.sinks.file_output_dir).mkdir(parents=True, exist_ok=True)

    def to_parquet(
        self,
        df: pl.DataFrame,
        entity_name: str,
        snapshot_date: Optional[str] = None,
        run_id: str = "backfill",
    ) -> str:
        date_part = f"_{snapshot_date}" if snapshot_date else ""
        fname = f"{entity_name}{date_part}.parquet"
        path = os.path.join(self.cfg.sinks.file_output_dir, run_id, fname)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
        return path

    def to_s3(
        self,
        df: pl.DataFrame,
        entity_name: str,
        snapshot_date: Optional[str] = None,
        run_id: str = "backfill",
    ) -> str:
        date_part = f"_{snapshot_date}" if snapshot_date else ""
        key = f"{run_id}/{entity_name}{date_part}.parquet"
        s3_path = f"s3://{self.cfg.sinks.s3_bucket}/{key}"
        storage_options = {
            "aws_endpoint_url": self.cfg.sinks.s3_endpoint,
            "aws_access_key_id": self.cfg.sinks.s3_access_key,
            "aws_secret_access_key": self.cfg.sinks.s3_secret_key,
            "aws_region": self.cfg.sinks.s3_region,
        }
        df.write_parquet(s3_path, storage_options=storage_options)
        return s3_path

    def to_postgres(
        self,
        df: pl.DataFrame,
        table_name: str,
        if_exists: str = "append",
    ) -> None:
        import sqlalchemy as sa
        engine = sa.create_engine(self.cfg.sinks.postgres_dsn)
        pdf = df.to_pandas()
        pdf.to_sql(table_name, engine, if_exists=if_exists, index=False)
        engine.dispose()

    def to_kafka(
        self,
        df: pl.DataFrame,
        topic: str,
        key_column: Optional[str] = None,
    ) -> None:
        try:
            from kafka import KafkaProducer
            import json
        except ImportError:
            raise ImportError("kafka-python not installed")
        producer = KafkaProducer(
            bootstrap_servers=self.cfg.sinks.kafka_brokers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
        )
        for row in df.iter_rows(named=True):
            key = row.get(key_column) if key_column else None
            producer.send(topic, value=row, key=key)
        producer.flush()
        producer.close()