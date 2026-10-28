"""Concrete sinks (parquet / S3 / postgres) + the legacy SinkManager facade.

Sinks are intentionally dumb: they write one frame to one destination. The
*routing* (which entity goes where, i.e. the lanes) lives in
``simulator.paths``; ``SinkManager`` keeps the old flat surface so existing
callers (``MonthBatch.to_sinks``) work unchanged.
"""

from __future__ import annotations

from simulator.sinks.parquet import LocalParquetSink
from simulator.sinks.postgres import PostgresSink
from simulator.sinks.s3 import S3Sink


class SinkManager:
    """Build the concrete sinks from SimConfig (legacy facade surface)."""

    def __init__(self, config) -> None:
        self.cfg = config
        s = config.sinks
        self.parquet = LocalParquetSink(s.file_output_dir)
        self.s3 = S3Sink(
            endpoint=s.s3_endpoint,
            access_key=s.s3_access_key,
            secret_key=s.s3_secret_key,
            bucket=s.s3_bucket,
            region=s.s3_region,
        )
        self.postgres = PostgresSink(s.postgres_dsn)
        # Lane writers (local import: simulator.paths imports sinks).
        from simulator.paths import OfflinePath, OnlinePath
        self.offline = OfflinePath(self.parquet, self.s3, s.enable_s3)
        self.online = OnlinePath(self.postgres, s.enable_postgres)

    @property
    def s3_enabled(self) -> bool:
        return self.cfg.sinks.enable_s3

    @property
    def postgres_enabled(self) -> bool:
        return self.cfg.sinks.enable_postgres

    def to_parquet(self, df, entity_name, snapshot_date=None, run_id="backfill"):
        return self.parquet.write(df, entity_name, snapshot_date, run_id)

    def to_s3(self, df, entity_name, snapshot_date=None, run_id="backfill"):
        return self.s3.write(df, entity_name, snapshot_date, run_id)

    def to_postgres(self, df, table_name, if_exists="append"):
        return self.postgres.write(df, table_name, if_exists=if_exists)


__all__ = ["LocalParquetSink", "PostgresSink", "S3Sink", "SinkManager"]
