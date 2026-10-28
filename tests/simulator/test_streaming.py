"""Phase 5: streaming problems + day-granular emitter."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import polars as pl

from simulator import SimConfig, Simulator
from simulator.clock import SimClock
from simulator.streaming.emitter import days_in_month, emit_month_stream, slice_day
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
# Emitter: month re-paced day by day
# --------------------------------------------------------------------------

def _stream_cfg(seed=7, n=150, **problem_overrides):
    cfg = SimConfig(n_customers=n, start_date=date(2020, 1, 1), seed=seed)
    cfg.sinks.enable_s3 = cfg.sinks.enable_postgres = False
    cfg.streaming.enabled = True
    # Legacy generic layer off for the emitter tests: its batch date-shift
    # moves event timestamps FORWARD into the next month, which breaks the
    # emitter's within-month day-slicing invariant. The real streaming
    # late-arrival (backward shift) is what belongs on this lane; the legacy
    # layer is retired by the post-Phase-5 Problem-interface refactor.
    cfg.problems.late_arrival_prob = 0.0
    for key, value in problem_overrides.items():
        setattr(cfg.streaming.problems.late_arrival, key, value)
    return cfg


def _month_batch(cfg):
    sim = Simulator(cfg)
    return sim.backfill("2020-01-31")[0], sim


def _recording_writer():
    calls = []

    def writer(bundle, day, kind):
        calls.append((day, kind, bundle))
        return {}
    return calls, writer


def test_emitter_delivers_every_day_once_no_problems():
    cfg = _stream_cfg(seed=7, n=150)
    batch, sim = _month_batch(cfg)
    pipeline = StreamProblemPipeline.from_config(
        np.random.default_rng(cfg.seed + 3), cfg.streaming)
    clock = SimClock(start_date=batch.month, bureau_lag_days=21)

    calls, writer = _recording_writer()
    stats = emit_month_stream(batch, clock, pipeline, writer)

    assert len(stats) == days_in_month(batch.month) == 31
    assert all(s["kind"] == "new" for s in stats)
    # every hot row of the month was delivered exactly once, day-sliced
    apps = batch.applications.height if batch.applications is not None else 0
    assert sum(s["rows"] for s in stats) == batch.card_auths.height + apps
    # day 15 slice equals the clean frame's day-15 rows
    day15 = [b for (d, _, b) in calls if d.day == 15][0]
    expected = slice_day(batch.card_auths, "event_timestamp", date(2020, 1, 15))
    assert day15["card_authorizations"].equals(expected)


def test_emitter_is_deterministic():
    records = []
    for _ in range(2):
        cfg = _stream_cfg(seed=9, n=120)
        batch, sim = _month_batch(cfg)
        pipeline = StreamProblemPipeline.from_config(
            np.random.default_rng(cfg.seed + 3), cfg.streaming)
        clock = SimClock(start_date=batch.month)
        calls, writer = _recording_writer()
        stats = emit_month_stream(batch, clock, pipeline, writer)
        records.append([(s["day"], s["rows"], s["late"]) for s in stats])
    assert records[0] == records[1]


def test_emitter_late_arrival_all_rows_shifted():
    cfg = _stream_cfg(seed=7, n=150)
    cfg.streaming.problems.late_arrival.rate = 1.0
    cfg.streaming.problems.late_arrival.delay_minutes_min = 5
    cfg.streaming.problems.late_arrival.delay_minutes_max = 45
    batch, sim = _month_batch(cfg)
    pipeline = StreamProblemPipeline.from_config(
        np.random.default_rng(cfg.seed + 3), cfg.streaming)
    calls, writer = _recording_writer()
    stats = emit_month_stream(batch, SimClock(start_date=batch.month),
                              pipeline, writer)
    # every fresh delivery had its auths late-stamped (rate=1)
    assert all(s["late"] >= s["rows"] - 1 for s in stats if s["rows"] > 0)
    # delivered event timestamps must be strictly before the clean slices
    delivered = [b for (d, _, b) in calls if d.day == 5][0]
    clean = slice_day(batch.card_auths, "event_timestamp", date(2020, 1, 5))
    assert all(
        delivered["card_authorizations"]["event_timestamp"][i]
        < clean["event_timestamp"][i]
        for i in range(clean.height))


def test_emitter_burst_repeats_bundle():
    cfg = _stream_cfg(seed=7, n=150)
    cfg.streaming.problems.burst.every_n_ticks = 1
    cfg.streaming.problems.burst.multiplier = 2
    batch, sim = _month_batch(cfg)
    pipeline = StreamProblemPipeline.from_config(
        np.random.default_rng(cfg.seed + 3), cfg.streaming)
    calls, writer = _recording_writer()
    stats = emit_month_stream(batch, SimClock(start_date=batch.month),
                              pipeline, writer)
    assert len(stats) == 2 * days_in_month(batch.month)


def test_emitter_replay_redelivers_on_rate_one():
    cfg = _stream_cfg(seed=7, n=150)
    cfg.streaming.problems.duplicate_replay.rate = 1.0
    batch, sim = _month_batch(cfg)
    pipeline = StreamProblemPipeline.from_config(
        np.random.default_rng(cfg.seed + 3), cfg.streaming)
    calls, writer = _recording_writer()
    stats = emit_month_stream(batch, SimClock(start_date=batch.month),
                              pipeline, writer)
    assert stats[0]["kind"] == "new"         # nothing remembered yet
    assert all(s["kind"] == "replay" for s in stats[1:])
