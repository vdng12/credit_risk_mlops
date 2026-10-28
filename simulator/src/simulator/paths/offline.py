"""Offline (batch) lane writes.

Lane = consumption, not cadence: batch jobs (Spark / DP pipelines) read
these artifacts. Everything is archived to local parquet (write-through)
and the immutable-facts subset goes to S3/SeaweedFS snapshots.

    parquet  -> every entity, every month (both lanes' entities)
    S3       -> clients, application_decisions, pos_balances,
                card_balances, bureau   (the cold-facts set)
"""

from __future__ import annotations

import polars as pl

from simulator.sinks.parquet import LocalParquetSink
from simulator.sinks.s3 import S3Sink

# Object-store cold facts: every BUSINESS ARTIFACT, never the pure
# real-time event streams. Rule: what is a stored product (decision /
# payment files, dims, snapshots, gold) vs a transient event row?
#
#   S3            = {clients, application_master, application_decisions,
#                    payments, pos_balances, card_balances, bureau}
#   pure-stream   = {loan_applications, card_authorizations} (Kafka + the
#                    parquet write-through archive only)
#
# Dual-lane: payments/decision outcomes are nightly FILES on S3 AND stream
# through CDC (the bank keeps the file and streams the events); clients is
# a monthly snapshot on S3 plus a dim stream.
_S3_COLD_FACTS = frozenset({
    "clients", "application_master", "application_decisions",
    "payments", "pos_balances", "card_balances", "bureau",
})


class OfflinePath:
    """Batch lane writer: write-through archive + S3 cold facts."""

    def __init__(self, parquet: LocalParquetSink, s3: S3Sink,
                 s3_enabled: bool) -> None:
        self.parquet = parquet
        self.s3 = s3
        self.s3_enabled = s3_enabled

    def write(self, entities: dict[str, pl.DataFrame | None],
              month: str, run_id: str) -> list[str]:
        paths: list[str] = []
        for name, df in entities.items():
            if df is None or df.height == 0:
                continue
            paths.append(self.parquet.write(df, name, month, run_id))
            if name in _S3_COLD_FACTS and self.s3_enabled:
                paths.append(self.s3.write(df, name, month, run_id))
        return paths
