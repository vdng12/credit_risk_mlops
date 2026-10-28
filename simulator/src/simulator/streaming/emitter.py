"""Day-granular delivery of a month's hot entities (SimClock-driven).

The generator produces a whole month's clean hot entities at once; the
emitter RE-PACES delivery day by day through SimClock, applies the
streaming problems per day, and hands each day's bundle to a writer (in
production: the online lane -> postgres -> Debezium CDC).

Determinism: day slices are pure subsets of the (seed, entity, month)
frames (Phase 0), so the monthly batch and the per-day stream carry the
same event set; only the real ingestion stamps differ.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any, Callable

import polars as pl

from simulator.clock import SimClock
from simulator.streaming.problems import StreamProblemPipeline

# hot entity -> the event timestamp column used for day slicing and late shift
_HOT_TS = {
    "card_authorizations": "event_timestamp",
    "loan_applications": "submission_timestamp",
}


def days_in_month(month_start) -> int:
    nxt = month_start.replace(day=28) + timedelta(days=4)
    last = nxt - timedelta(days=nxt.day)
    return last.day


def slice_day(frame: pl.DataFrame | None, ts_col: str, day) -> pl.DataFrame | None:
    if frame is None or frame.height == 0:
        return frame
    return frame.filter(pl.col(ts_col).dt.date() == day)


def _now_for(day) -> datetime:
    return datetime.combine(day, datetime.min.time())


def emit_month_stream(
    batch,
    clock: SimClock,
    pipeline: StreamProblemPipeline,
    writer: Callable[[dict[str, pl.DataFrame], Any, str], dict[str, Any]],
    tick_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    """Deliver ``batch.month``'s hot entities day by day.

    For each simulated day:

    1. slice the month's auths/applications to that day
    2. burst: repeat the day bundle ``events_for_tick(tick, 1)`` times
    3. late arrival: re-stamp a fraction of the day's event timestamps
    4. duplicate replay (rate): deliver a remembered past day bundle
       instead of a fresh one (at-least-once; volume stays put)
    5. hand each delivery to ``writer(bundle, day, kind)``

    Returns per-delivery stats (day, kind, rows, late, max_delay_seconds).
    """
    month = batch.month
    auths, apps = batch.card_auths, batch.applications
    deliveries: list[dict[str, Any]] = []
    tick_index = 0

    for _ in range(days_in_month(month)):
        # deliver the clock's CURRENT day first, advance after - so the
        # month's last day is not skipped by the final tick
        day = clock.current_date
        now = _now_for(day)

        bundle = {
            "card_authorizations": slice_day(auths, "event_timestamp", day),
            "loan_applications": slice_day(apps, "submission_timestamp", day),
        }

        late_total = 0
        max_delay = 0
        for name, frame in bundle.items():
            if frame is None or frame.height == 0:
                continue
            frame, n, d = pipeline.apply_late(frame, _HOT_TS[name], now)
            bundle[name] = frame
            late_total += n
            max_delay = max(max_delay, d)

        replay = pipeline.replay(now)
        if replay is not None:
            kind = "replay"
            delivery = replay
        else:
            kind = "new"
            delivery = bundle
            pipeline.remember(bundle)

        repeat = pipeline.events_for_tick(tick_index, 1)
        for _ in range(repeat):
            rows = sum(
                f.height for f in delivery.values() if f is not None)
            stats = writer(delivery, day, kind)
            deliveries.append({
                "day": day.isoformat(),
                "kind": kind,
                "rows": rows,
                "late": late_total if kind == "new" else 0,
                "max_delay_seconds": max_delay if kind == "new" else 0,
                **stats,
            })
            if tick_seconds:
                time.sleep(tick_seconds)
        tick_index += 1
        clock.tick(days=1)   # move to the next delivery day

    return deliveries
