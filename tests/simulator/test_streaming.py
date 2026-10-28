"""Phase 5: streaming problems + day-granular emitter."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import polars as pl

from simulator import SimConfig, Simulator
from simulator.clock import SimClock
from simulator.streaming.emitter import days_in_month, slice_day
from simulator.streaming.problems import (
    BurstTrafficProblem,
    DuplicateReplayProblem,
    LateArrivalProblem,
    StreamProblemPipeline,
)


# --------------------------------------------------------------------------
# Burst
# --------------------------------------------------------------------------

def test_burst_traffic_fires_every_n_ticks():
    burst = BurstTrafficProblem(every_n_ticks=3, multiplier=4)
    assert [burst.events_for_tick(t, 1) for t in (1, 2, 3, 4, 5, 6)] == [
        1, 1, 4, 1, 1, 4]


def test_burst_disabled_when_zero():
    burst = BurstTrafficProblem(every_n_ticks=0, multiplier=5)
    assert burst.events_for_tick(7, 100) == 100


# --------------------------------------------------------------------------
# Late arrival
# --------------------------------------------------------------------------

def test_late_arrival_shifts_subset_into_the_past():
    rng = np.random.default_rng(4)
    ts = [datetime(2020, 1, 5, 10, 0) + timedelta(minutes=i) for i in range(10)]
    df = pl.DataFrame({"auth_id": range(10), "event_timestamp": ts})

    # rate=0 -> untouched
    out, n, d = LateArrivalProblem(rng, 0.0, 5, 45).apply(df, "event_timestamp", now=None)
    assert n == 0 and out.equals(df)

    # rate=1, fixed delay -> every row 10 minutes in the past
    late = LateArrivalProblem(rng, 1.0, 10, 10)
    out, n, d = late.apply(df, "event_timestamp", now=None)
    assert n == 10 and d == 600
    shifted = out["event_timestamp"].to_list()
    assert all(out["event_timestamp"][i] == df["event_timestamp"][i]
               - timedelta(minutes=10) for i in range(10))
    assert shifted[0] == ts[0] - timedelta(minutes=10)


# --------------------------------------------------------------------------
# Duplicate replay
# --------------------------------------------------------------------------

def test_duplicate_replay_redelivers_a_past_bundle():
    rng = np.random.default_rng(4)
    replay = DuplicateReplayProblem(rng, rate=1.0, history_size=5)
    day1 = {"card_authorizations": pl.DataFrame({
        "auth_id": ["a1"], "event_timestamp": [datetime(2020, 1, 1, 9, 0)],
        "ingestion_ts": [datetime(2020, 1, 1, 9, 0)]})}
    replay.remember(day1)

    now = datetime(2020, 1, 2, 9, 0)
    bundle = replay.replay(now)
    assert bundle is not None
    frame = bundle["card_authorizations"]
    # event time stays at the original; ingestion re-stamped to now
    assert frame["event_timestamp"][0] == datetime(2020, 1, 1, 9, 0)
    assert frame["ingestion_ts"][0] == now
    # history bounded
    for i in range(10):
        replay.remember({"x": pl.DataFrame({"i": [i]})})
    assert len(replay._history) == 5


def test_duplicate_replay_rate_zero_never_redelivers():
    replay = DuplicateReplayProblem(np.random.default_rng(4), rate=0.0, history_size=5)
    replay.remember({"card_authorizations": pl.DataFrame({"auth_id": ["a"]})})
    assert replay.replay(datetime(2020, 1, 2)) is None


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Live stage: hot + nightly + monthly dispatch (replaces the old emitter)
# --------------------------------------------------------------------------

def _stream_cfg(seed=7, n=150, **problem_overrides):
    cfg = SimConfig(n_customers=n, start_date=date(2020, 1, 1), seed=seed)
    cfg.sinks.enable_s3 = cfg.sinks.enable_postgres = False
    cfg.streaming.enabled = True
    # legacy batch date-shift off (moves event timestamps FORWARD, breaking
    # within-month day slicing; retired by the post-Phase-5 refactor)
    cfg.problems.late_arrival_prob = 0.0
    for key, value in problem_overrides.items():
        setattr(cfg.streaming.problems.late_arrival, key, value)
    return cfg


def _month_batch(cfg):
    sim = Simulator(cfg)
    return sim.backfill("2020-01-31")[0], sim


class _RecLane:
    def __init__(self, label: str) -> None:
        self.label = label
        self.calls: list[dict] = []

    def write(self, entities, snapshot, run_id):
        self.calls.append({
            "at": snapshot,
            "entities": {k: v.height for k, v in entities.items()
                         if v is not None}})
        return [f"{self.label}:{snapshot}"]


class _RecSink:
    def __init__(self) -> None:
        self.online = _RecLane("online")
        self.offline = _RecLane("offline")

    @property
    def s3_enabled(self) -> bool:
        return False

    @property
    def postgres_enabled(self) -> bool:
        return True


def _stage(cfg):
    from simulator.livestage import LiveStage
    from simulator.streaming.problems import StreamProblemPipeline
    import numpy as np
    sink = _RecSink()
    pipeline = StreamProblemPipeline.from_config(
        np.random.default_rng(cfg.seed + 3), cfg.streaming)
    stage = LiveStage(sink, pipeline,
                      bureau_lag_days=cfg.time.bureau_drop_lag_days)
    return stage, sink


def _emit(cfg):
    batch, _ = _month_batch(cfg)
    stage, sink = _stage(cfg)
    stats = stage.emit(batch, run_id="t")
    return stats, sink, batch


def test_livestage_hot_covers_every_day_once():
    cfg = _stream_cfg(seed=7, n=150)
    stats, sink, batch = _emit(cfg)
    hot = [s for s in stats if s["kind"] in ("new", "replay")]
    assert len(hot) == days_in_month(batch.month) == 31
    assert all(s["kind"] == "new" for s in hot)
    apps = batch.applications.height if batch.applications is not None else 0
    assert sum(s["rows"] for s in hot) == batch.card_auths.height + apps


def test_livestage_nightly_files_carry_day_slices():
    cfg = _stream_cfg(seed=7, n=150)
    stats, sink, batch = _emit(cfg)
    nights = [s for s in stats if s["kind"] == "nightly"]
    assert nights, "at least one nightly file with content"
    # every payment row lands in exactly one nightly file (day-sliced)
    payments_in_nights = sum(
        call["entities"].get("payments", 0) for call in sink.offline.calls)
    assert payments_in_nights == batch.payments.height
    # every decision row too
    decisions_in_nights = sum(
        call["entities"].get("application_decisions", 0)
        for call in sink.offline.calls)
    assert decisions_in_nights == batch.application_decisions.height


def test_livestage_monthly_close_written_once_with_all_snapshots():
    cfg = _stream_cfg(seed=7, n=150)
    stats, sink, batch = _emit(cfg)
    closes = [s for s in stats if s["kind"] == "monthly"]
    assert len(closes) == 1
    offline_names = {
        k for call in sink.offline.calls for k in call["entities"]}
    assert {"clients", "pos_balances", "card_balances", "bureau",
            "application_master"} <= offline_names
    # gold master reaches the online lane too
    online_names = {
        k for call in sink.online.calls for k in call["entities"]}
    assert "application_master" in online_names


def test_livestage_is_deterministic():
    recs = []
    for _ in range(2):
        cfg = _stream_cfg(seed=9, n=120)
        stats, _, _ = _emit(cfg)
        recs.append([(s["day"], s["kind"], s["rows"]) for s in stats])
    assert recs[0] == recs[1]


def test_livestage_late_arrival_all_rows_shifted():
    cfg = _stream_cfg(seed=7, n=150)
    cfg.streaming.problems.late_arrival.rate = 1.0
    cfg.streaming.problems.late_arrival.delay_minutes_min = 5
    cfg.streaming.problems.late_arrival.delay_minutes_max = 45
    stats, _, _ = _emit(cfg)
    assert all(s["late"] >= s["rows"] - 1 for s in stats
               if s["kind"] == "new" and s["rows"] > 0)


def test_livestage_burst_repeats_the_day_bundle():
    cfg = _stream_cfg(seed=7, n=150)
    cfg.streaming.problems.burst.every_n_ticks = 1
    cfg.streaming.problems.burst.multiplier = 2
    stats, _, batch = _emit(cfg)
    hot = [s for s in stats if s["kind"] in ("new", "replay")]
    assert len(hot) == 2 * days_in_month(batch.month)


def test_livestage_replay_redelivers_on_rate_one():
    cfg = _stream_cfg(seed=7, n=150)
    cfg.streaming.problems.duplicate_replay.rate = 1.0
    stats, _, _ = _emit(cfg)
    hot = [s for s in stats if s["kind"] in ("new", "replay")]
    assert hot[0]["kind"] == "new"
    assert all(s["kind"] == "replay" for s in hot[1:])
