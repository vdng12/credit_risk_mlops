"""Phase 4: offline problems - schema evolution (bureau), payload hash, skew."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from simulator import SimConfig, Simulator
from simulator.evolution import SchemaEvolutionProblem
from simulator.payload_hash import frame_payload_hashes, row_payload_hash

_BUREAU = pl.DataFrame({
    "bureau_id": ["BU-1-0001"],
    "client_id": ["C00000001"],
    "snapshot_date": [date(2020, 1, 31)],
    "credit_duration_days": [30],
    "days_credit": [-30],
    "amt_credit_sum": [1000.0],
    "amt_credit_sum_overdue": [200.0],
})


# --------------------------------------------------------------------------
# Schema evolution (bureau only)
# --------------------------------------------------------------------------

def test_evolution_version_mapping():
    ev = SchemaEvolutionProblem(
        change_date=date(2020, 2, 1),
        breaking_change_date=date(2020, 3, 1))
    assert ev.version(date(2020, 1, 15)) == 1
    assert ev.version(date(2020, 2, 15)) == 2
    assert ev.version(date(2020, 3, 15)) == 3


def test_evolution_v1_omits_new_field_v2_adds():
    ev = SchemaEvolutionProblem(change_date=date(2020, 2, 1))
    out1, v1 = ev.apply(_BUREAU, date(2020, 1, 15))
    assert v1 == 1
    assert "debt_ratio" not in out1.columns   # schema differs between files
    assert out1["schema_version"][0] == 1

    out2, v2 = ev.apply(_BUREAU, date(2020, 2, 15))
    assert v2 == 2
    assert "debt_ratio" in out2.columns
    assert out2["debt_ratio"][0] == pytest.approx(0.2)  # overdue / sum


def test_breaking_change_retires_credit_duration_days():
    ev = SchemaEvolutionProblem(
        change_date=date(2020, 2, 1),
        breaking_change_date=date(2020, 3, 1))
    out3, v3 = ev.apply(_BUREAU, date(2020, 3, 15))
    assert v3 == 3
    assert "credit_duration_days" not in out3.columns
    assert "credit_duration_months" in out3.columns
    assert out3["credit_duration_months"][0] == pytest.approx(1.0)


def test_evolution_disabled_is_noop():
    ev = SchemaEvolutionProblem(change_date=None)
    assert ev.version(date(2020, 5, 15)) == 1


def test_simulator_evolves_bureau_by_month(tmp_path):
    cfg = SimConfig(n_customers=150, start_date=date(2020, 1, 1), seed=7)
    cfg.problems.schema_evolution_enabled = True
    cfg.problems.schema_evolution_change_date = date(2020, 2, 1)
    cfg.problems.schema_evolution_breaking_change_date = date(2020, 3, 1)
    cfg.sinks.enable_s3 = cfg.sinks.enable_postgres = False
    sim = Simulator(cfg)
    batches = sim.backfill("2020-03-31")
    assert len(batches) == 3

    jan, feb, mar = batches
    assert jan.bureau["schema_version"][0] == 1
    assert "debt_ratio" not in jan.bureau.columns

    assert feb.bureau["schema_version"][0] == 2
    assert "debt_ratio" in feb.bureau.columns

    assert mar.bureau["schema_version"][0] == 3
    assert "credit_duration_days" not in mar.bureau.columns
    assert "credit_duration_months" in mar.bureau.columns
    # the non-bureau entities are never versioned
    assert "schema_version" not in jan.payments.columns


# --------------------------------------------------------------------------
# Payload hash (content identity)
# --------------------------------------------------------------------------

def test_payload_hash_deterministic_and_sensitive():
    record = {"application_id": "APP-1", "amount": 100.0, "schema_version": 1}
    assert row_payload_hash(record) == row_payload_hash(dict(record))
    assert row_payload_hash(record) != row_payload_hash(
        {**record, "schema_version": 2})
    assert row_payload_hash(record) != row_payload_hash(
        {**record, "amount": 101.0})


def test_frame_payload_hashes_identity_for_duplicates():
    df = pl.DataFrame({
        "application_id": ["APP-1", "APP-1", "APP-2"],
        "amount": [100.0, 100.0, 200.0],
    })
    hashes = frame_payload_hashes(df).to_list()
    assert hashes[0] == hashes[1]   # exact duplicate -> identical hash
    assert hashes[0] != hashes[2]


# --------------------------------------------------------------------------
# Generation-time skew (region concentration)
# --------------------------------------------------------------------------

def _region_ratios(ratio: float) -> pl.Series:
    cfg = SimConfig(n_customers=600, start_date=date(2020, 1, 1), seed=7)
    cfg.problems.region_skew_top_ratio = ratio
    cfg.sinks.enable_s3 = cfg.sinks.enable_postgres = False
    sim = Simulator(cfg)
    batch = sim.backfill("2020-01-31")[0]
    return batch.clients["region_rating"]


def _top_share(s: pl.Series) -> float:
    # null-aware share of region=5 (one missing-injection null may exist)
    return int((s == 5).sum()) / s.len()


def test_region_skew_knob():
    no_skew = _region_ratios(0.0)
    half = _region_ratios(0.5)
    all_top = _region_ratios(1.0)

    # ratio=1.0 forces the whole portfolio onto the top region (modulo the
    # problem injector's missing-value null)
    assert (all_top.fill_null(5) == 5).all()
    # skew monotonically pushes the top-region share up
    assert _top_share(half) > _top_share(no_skew)
    # data still deterministic for a fixed knob
    assert _region_ratios(0.5).equals(_region_ratios(0.5))