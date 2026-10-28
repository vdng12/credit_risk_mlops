"""Phase 3: lane path routing (offline batch vs online stream)."""

from __future__ import annotations

from datetime import date

import polars as pl

from simulator import SimConfig, Simulator
from simulator.paths.offline import _S3_COLD_FACTS, OfflinePath
from simulator.paths.online import _STREAM_OLTP, OnlinePath
from simulator.sinks.parquet import LocalParquetSink
from simulator.sinks.s3 import S3Sink


def test_lane_routing_sets_are_explicit():
    # offline: every business ARTIFACT on object storage (object-store rule)
    assert _S3_COLD_FACTS == {
        "clients", "application_master", "application_decisions",
        "payments", "pos_balances", "card_balances", "bureau",
    }
    # pure-stream entities never go to S3
    assert "loan_applications" not in _S3_COLD_FACTS
    assert "card_authorizations" not in _S3_COLD_FACTS
    # online: OLTP tables on postgres-source (the Debezium CDC source) + gold
    assert _STREAM_OLTP == {
        "clients", "loan_applications", "card_authorizations",
        "application_decisions", "payments", "application_master",
    }
    # every CDC-published table is part of the stream lane
    cdc = {"loan_applications", "card_authorizations", "payments",
           "application_decisions", "clients"}
    assert cdc <= _STREAM_OLTP


class _RecordingS3:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def write(self, df, entity_name, snapshot_date=None, run_id="default"):
        self.calls.append(entity_name)
        return f"s3://bucket/{entity_name}_{snapshot_date}.parquet"


def test_offline_path_archives_everything_s3_only_cold_facts(tmp_path):
    parquet = LocalParquetSink(tmp_path)
    s3 = _RecordingS3()
    offline = OfflinePath(parquet, s3, s3_enabled=True)

    entities = {
        "clients": pl.DataFrame({"client_id": [1]}),           # artifact
        "bureau": pl.DataFrame({"bureau_id": ["b1"]}),         # artifact
        "application_master": pl.DataFrame({"application_id": ["a1"]}),
        "payments": pl.DataFrame({"payment_id": ["p1"]}),       # artifact
        "loan_applications": None,                             # empty -> skipped
    }
    paths = offline.write(entities, "2020-01", "run1")
    # every non-empty entity archived locally; the None is skipped
    local = [p for p in paths if not p.startswith("s3://")]
    assert {p.split("/")[-1] for p in local} == {
        "clients_2020-01.parquet", "bureau_2020-01.parquet",
        "application_master_2020-01.parquet", "payments_2020-01.parquet"}
    # S3 got exactly the artifact subset: clients + bureau + master + payments
    assert sorted(s3.calls) == ["application_master", "bureau",
                                "clients", "payments"]
    assert len(paths) == 8  # 4 parquet + 4 S3


class _RecordingPostgres:
    def __init__(self) -> None:
        self.tables: list[str] = []

    def write(self, df, table_name, if_exists="append"):
        self.tables.append(table_name)


def test_online_path_only_touches_oltp_entities():
    pg = _RecordingPostgres()
    online = OnlinePath(pg, enabled=True)
    entities = {
        "clients": pl.DataFrame({"client_id": [1]}),
        "loan_applications": pl.DataFrame({"application_id": ["a1"]}),
        "card_authorizations": pl.DataFrame({"auth_id": ["x1"]}),
        "application_decisions": pl.DataFrame({"application_id": ["a1"]}),
        "payments": pl.DataFrame({"payment_id": ["p1"]}),
        "application_master": pl.DataFrame({"application_id": ["a1"]}),
        "pos_balances": pl.DataFrame({"pos_balance_id": ["pb1"]}),
        "card_balances": pl.DataFrame({"card_balance_id": ["cb1"]}),
        "bureau": pl.DataFrame({"bureau_id": ["b1"]}),
    }
    paths = online.write(entities, "2020-01", "run1")
    assert set(pg.tables) == {
        "clients", "loan_applications", "card_authorizations",
        "application_decisions", "payments", "application_master"}
    assert len(paths) == 6


def test_online_path_off_when_disabled():
    pg = _RecordingPostgres()
    online = OnlinePath(pg, enabled=False)
    online.write({"clients": pl.DataFrame({"client_id": [1]})}, "m", "r")
    assert pg.tables == []


def test_month_batch_to_sinks_routes_through_both_lanes(tmp_path):
    cfg = SimConfig(n_customers=50, start_date=date(2020, 1, 1), seed=7)
    cfg.sinks.enable_s3 = False
    cfg.sinks.enable_postgres = False
    cfg.sinks.file_output_dir = str(tmp_path)
    sim = Simulator(cfg)
    batch = sim.backfill("2020-01-31")[0]

    paths = batch.to_sinks(sim.sink, run_id="lane_test")
    files = sorted(p for p in paths if p.endswith(".parquet"))
    assert files, "parquet write-through must cover the month"
    # offline lane wrote snapshots for the cold-facts entities
    assert any("clients_2020-01.parquet" in p for p in files)
    assert any("bureau_2020-01.parquet" in p for p in files)
    # postgres stream lane is disabled -> no postgres: paths
    assert not any(p.startswith("postgres:") for p in paths)
    # every entity is archived (write-through covers both lanes)
    import os
    # strip only the trailing "_<yyyy-mm>" date part
    archived = {f.rsplit("_", 1)[0] for f in os.listdir(tmp_path / "lane_test")}
    assert {"clients", "loan_applications", "application_decisions",
            "application_master", "card_authorizations", "pos_balances",
            "card_balances", "bureau", "payments"} <= archived