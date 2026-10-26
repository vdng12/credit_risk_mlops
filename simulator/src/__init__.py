"""
t-Copula Data Simulator for Default Risk MLOps.

Stateful iterator-style simulator producing synthetic lending data across
8 entity types through two paths: online event streams (Redpanda/Kafka) and
offline batch data (Postgres + SeaweedFS/S3).

Usage:
    from simulator import Simulator, SimConfig

    cfg = SimConfig(n_customers=50_000, start_date=date(2020, 1, 1))
    sim = Simulator(cfg)

    sim.backfill(until="2023-12-31")

    for month_batch in sim.play_forward(months=24):
        month_batch.to_postgres(engine)
        month_batch.to_seaweedfs(s3_client, bucket="risk-data")
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