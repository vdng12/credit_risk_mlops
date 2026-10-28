"""Online (stream) lane writes.

Lane = consumption: stream processors (Flink) consume these via the
Debezium CDC stream. The generator does NOT write Kafka/Redpanda directly;
it appends the OLTP tables on postgres-source and CDC re-publishes them.

    postgres -> clients, loan_applications, card_authorizations,
                application_decisions, payments   (CDC set)
                + application_master              (gold, no CDC)

Cadence is orthogonal to the lane: payments/decisions land as nightly
files and are then streamed; clients is a monthly master that also streams.
"""

from __future__ import annotations

import polars as pl

from simulator.sinks.postgres import PostgresSink

# Tables the Debezium connector re-publishes to `lending.public.*` topics,
# plus the gold layer written to the same database (no CDC).
_STREAM_OLTP = frozenset({
    "clients", "loan_applications", "card_authorizations",
    "application_decisions", "payments", "application_master",
})


class OnlinePath:
    """Stream lane writer: OLTP postgres appends (the CDC source)."""

    def __init__(self, postgres: PostgresSink, enabled: bool) -> None:
        self.postgres = postgres
        self.enabled = enabled

    def write(self, entities: dict[str, pl.DataFrame | None],
              month: str, run_id: str) -> list[str]:
        paths: list[str] = []
        if not self.enabled:
            return paths
        for name, df in entities.items():
            if name not in _STREAM_OLTP or df is None or df.height == 0:
                continue
            self.postgres.write(df, table_name=name)
            paths.append(f"postgres:{name}")
        return paths
