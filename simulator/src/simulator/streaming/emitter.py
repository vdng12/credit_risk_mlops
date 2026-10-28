"""Hot-lane day helpers (streaming only).

The SimClock day loop and the lane dispatch live in ``simulator.livestage``;
this module provides what the HOT lane needs: day-slicing of the month's
events and the streaming-problem application on a single day's bundle.
"""

from __future__ import annotations

from datetime import datetime, date, timedelta

import polars as pl

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


def now_for(day) -> datetime:
    return datetime.combine(day, datetime.min.time())


def prepare_hot_bundle(
    batch, day: date, pipeline: StreamProblemPipeline
) -> tuple[dict[str, pl.DataFrame | None], int, int]:
    """The day's hot entities with late arrival applied.

    Returns (bundle, late_count, max_delay_seconds). Duplicate replay and
    burst are handled by the dispatcher (livestage), which decides whether
    a replay replaces this day's slot.
    """
    late_total = 0
    max_delay = 0
    now = now_for(day)
    bundle: dict[str, pl.DataFrame | None] = {}
    for name, ts_col in _HOT_TS.items():
        frame = slice_day(getattr(batch, {
            "card_authorizations": "card_auths",
            "loan_applications": "applications",
        }[name]), ts_col, day)
        if frame is not None and frame.height > 0:
            frame, n, d = pipeline.apply_late(frame, ts_col, now)
            late_total += n
            max_delay = max(max_delay, d)
        bundle[name] = frame
    return bundle, late_total, max_delay
