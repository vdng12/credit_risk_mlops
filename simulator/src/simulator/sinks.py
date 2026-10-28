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

    @property
    def s3_enabled(self) -> bool:
        return self.cfg.sinks.enable_s3

    @property
    def postgres_enabled(self) -> bool:
        return self.cfg.sinks.enable_postgres

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

        # `postgresql://` defaults to the psycopg (v3) dialect in modern
        # SQLAlchemy; pin the installed driver (psycopg2-binary) when no
        # explicit `+driver` is present in the DSN.
        url = sa.engine.make_url(self.cfg.sinks.postgres_dsn)
        if url.drivername == "postgresql":
            url = url.set(drivername="postgresql+psycopg2")
        engine = sa.create_engine(url)
        pdf = df.to_pandas()
        with engine.connect() as conn:
            # Tolerant append: project onto the target table's columns (the
            # OLTP schema is the contract; e.g. loan_applications carries only
            # pre-decision fields, decision outcome lives in decisions).
            inspector = sa.inspect(conn)
            if table_name in inspector.get_table_names():
                existing = {c["name"] for c in inspector.get_columns(table_name)}
                pdf = pdf[[c for c in pdf.columns if c in existing]]
        pdf.to_sql(table_name, engine, if_exists=if_exists, index=False)
        engine.dispose()

