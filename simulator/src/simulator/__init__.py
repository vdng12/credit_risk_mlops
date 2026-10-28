"""
t-Copula Data Simulator for Default Risk MLOps.

Stateful iterator-style simulator producing synthetic lending data across
8 entity types, all released COLD-first: local parquet write-through, S3/
SeaweedFS for object-store entities, Postgres (lending_db) for the OLTP +
gold entities. Kafka/Redpanda is NOT written directly - the Debezium CDC
connector on postgres-source re-streams the OLTP tables.

Usage:
    from simulator import Simulator, SimConfig

    cfg = SimConfig(n_customers=50_000, start_date=date(2020, 1, 1))
    sim = Simulator(cfg)

    sim.backfill(until="2023-12-31")

    # Routing per design (AGENTS.md): file write-through + S3 object-store
    # entities + Postgres OLTP/gold entities (CDC re-streams them to Kafka).
    for month_batch in sim.play_forward(months=24):
        month_batch.to_sinks(sim.sink, run_id="incremental")
"""

from simulator.config import SimConfig, CopulaConfig, EntityConfig, TimeConfig
from simulator.core import Simulator

__all__ = [
    "SimConfig",
    "CopulaConfig",
    "EntityConfig",
    "TimeConfig",
    "Simulator",
]