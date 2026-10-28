"""Live stage: the three-rhythm dispatcher (SimClock-driven).

Re-paces one generated month's delivery day by day and dispatches to the
LANES on the clock's boundary flags. Routing knowledge stays in
``simulator.paths`` - this module only groups entities by cadence:

    every tick   -> HOT bundle   (auths + applications + streaming problems)
                      -> online lane (postgres -> Debezium CDC)
    payment bdry -> NIGHTLY file (payments + application_decisions slices)
                      -> offline lane (parquet + S3) AND online lane (stream)
    month bdry   -> MONTHLY close (clients, balances, bureau, gold master)
                      -> offline lane (parquet + S3), gold -> online lane

Everything is deterministic (pure day slices of the seed-month frames +
seeded streaming problems); only real ingestion stamps differ.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any

import polars as pl

from simulator.clock import SimClock
from simulator.streaming.emitter import days_in_month, now_for, prepare_hot_bundle
from simulator.streaming.problems import StreamProblemPipeline

# nightly entities: the day's payment + decision slices (dual-lane: file
# on the batch lane AND a CDC stream)
_NIGHTLY = ("payments", "application_decisions")
# monthly close: snapshots + gold written at the month boundary
_MONTHLY = ("clients", "pos_balances", "card_balances", "bureau",
            "application_master")


def _frame(batch, name):
    attr = {
        "payments": "payments",
        "application_decisions": "application_decisions",
        "clients": "clients",
        "pos_balances": "pos_balances",
        "card_balances": "card_balances",
        "bureau": "bureau",
        "application_master": "application_master",
    }[name]
    return getattr(batch, attr)


def _day_slice(batch, name, day):
    frame = _frame(batch, name)
    ts_col = {"payments": "payment_date",
              "application_decisions": "decision_timestamp"}.get(name)
    if ts_col is None or frame is None or frame.height == 0:
        return frame
    return frame.filter(pl.col(ts_col).dt.date() == day)


class LiveStage:
    """Day-granular delivery of a month with hot/nightly/monthly dispatch."""

    def __init__(
        self,
        sink,
        pipeline: StreamProblemPipeline,
        bureau_lag_days: int = 21,
    ) -> None:
        self.sink = sink
        self.pipeline = pipeline
        self.bureau_lag_days = int(bureau_lag_days)

    def emit(self, batch, run_id: str = "default",
             tick_seconds: float = 0.0) -> list[dict[str, Any]]:
        month = batch.month
        # start one day BEFORE the month so the first tick lands on the 1st
        clock = SimClock(
            start_date=month - timedelta(days=1),
            bureau_lag_days=self.bureau_lag_days)
        deliveries: list[dict[str, Any]] = []

        for _ in range(days_in_month(month)):
            flags = clock.tick(days=1)
            day = clock.current_date

            # ---- HOT: always ----
            bundle, late, max_delay = prepare_hot_bundle(batch, day, self.pipeline)
            replay = self.pipeline.replay(now_for(day))
            repeat = self.pipeline.events_for_tick(len(deliveries), 1)
            kind = "replay" if replay is not None else "new"
            out = replay if replay is not None else bundle

            for _ in range(repeat):
                paths = self.sink.online.write(out, day.isoformat(), run_id)
                deliveries.append({
                    "day": day.isoformat(), "kind": kind,
                    "rows": sum(f.height for f in out.values() if f is not None),
                    "late": late if kind == "new" else 0,
                    "max_delay_seconds": max_delay if kind == "new" else 0,
                    "paths": paths,
                })
                if tick_seconds:
                    time.sleep(tick_seconds)
            if replay is None:
                self.pipeline.remember(out)

            # ---- NIGHTLY: payment / decision files ----
            if flags["payment"]:
                night = {"payments": _day_slice(batch, "payments", day),
                         "application_decisions": _day_slice(
                             batch, "application_decisions", day)}
                for name, frame in list(night.items()):
                    if frame is None or frame.height == 0:
                        night.pop(name)
                if night:
                    rows = sum(f.height for f in night.values())
                    paths = self.sink.offline.write(
                        night, day.isoformat(), run_id)
                    night_paths = self.sink.online.write(
                        night, day.isoformat(), run_id)
                    deliveries.append({
                        "day": day.isoformat(), "kind": "nightly",
                        "rows": rows, "paths": paths + night_paths,
                    })

            # ---- MONTHLY: close snapshots + gold ----
            if flags["monthly"]:
                close = {name: _frame(batch, name) for name in _MONTHLY}
                close = {k: v for k, v in close.items()
                         if v is not None and v.height > 0}
                rows = sum(f.height for f in close.values())
                paths = self.sink.offline.write(close, month.isoformat(), run_id)
                gold_paths = self.sink.online.write(
                    {"application_master": close.get("application_master")},
                    month.isoformat(), run_id)
                deliveries.append({
                    "day": day.isoformat(), "kind": "monthly", "rows": rows,
                    "paths": paths + gold_paths,
                })

        return deliveries
