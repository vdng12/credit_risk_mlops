"""Streaming data problems (online lane) - Phase 5.

Three problems, scope = the stream (CDC) entities only (see the problem ->
entity contract in simulator.problems):

    burst traffic     - periodic Nx volume multiplier on a tick index
    late arrival      - a fraction of rows re-stamped with event
                        timestamps in the past (event happened earlier,
                        ingestion remains the delivery moment)
    duplicate replay  - at-least-once re-delivery: a whole previous day
                        bundle reappears with new ingestion timestamps

All randomness derives from the injected RNG (seeded), so re-running the
same month yields identical corruption; only the real ingestion timestamps
(the DB write moment) are not reproducible.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta
from typing import Any

import polars as pl


class BurstTrafficProblem:
    """Periodic volume spikes; every_n_ticks == 0 disables burst."""

    def __init__(self, every_n_ticks: int, multiplier: int) -> None:
        self.every_n_ticks = int(every_n_ticks)
        self.multiplier = int(multiplier)

    def events_for_tick(self, tick: int, normal_events: int) -> int:
        if self.every_n_ticks > 0 and tick % self.every_n_ticks == 0:
            return int(normal_events) * self.multiplier
        return int(normal_events)


class LateArrivalProblem:
    """Re-stamp a fraction of rows with past event timestamps.

    Returns ``(frame, late_count, max_delay_seconds)`` ground truth.
    """

    def __init__(
        self, rng, rate: float, delay_minutes_min: int, delay_minutes_max: int
    ) -> None:
        self.rng = rng
        self.rate = float(rate)
        self.minutes_min = int(delay_minutes_min)
        self.minutes_max = int(delay_minutes_max)

    def apply(
        self, frame: pl.DataFrame, ts_col: str, now: datetime
    ) -> tuple[pl.DataFrame, int, int]:
        del now  # ingestion time is the caller's concern; only event times shift
        if self.rate <= 0 or frame.height == 0 or ts_col not in frame.columns:
            return frame, 0, 0
        indices = [i for i in range(frame.height) if self.rng.random() < self.rate]
        n = len(indices)
        if n == 0:
            return frame, 0, 0
        delays = self.rng.integers(
            self.minutes_min, self.minutes_max + 1, size=n)
        ts = frame[ts_col].to_list()
        shifted = list(ts)
        for k, i in enumerate(indices):
            shifted[i] = ts[i] - timedelta(minutes=int(delays[k]))
        out = frame.with_columns(pl.Series(ts_col, shifted))
        return out, n, int(delays.max()) * 60


class DuplicateReplayProblem:
    """At-least-once re-delivery of a whole remembered day bundle.

    On ``replay``, with probability ``rate`` a random past bundle is
    deep-copied and its ingestion-side timestamps re-stamped to ``now``
    (event timestamps stay at their original values - that is the
    watermark material downstream).
    """

    def __init__(self, rng, rate: float, history_size: int) -> None:
        self.rng = rng
        self.rate = float(rate)
        self.history_size = int(history_size)
        self._history: list[dict[str, pl.DataFrame]] = []

    def remember(self, bundle: dict[str, pl.DataFrame]) -> None:
        self._history.append({k: v.clone() for k, v in bundle.items()})
        self._history = self._history[-self.history_size:]

    def replay(
        self, now: datetime
    ) -> dict[str, pl.DataFrame] | None:
        if not self._history or self.rng.random() >= self.rate:
            return None
        bundle = copy.deepcopy(self.rng.choice(self._history))
        for name, frame in bundle.items():
            for col in ("ingestion_ts", "created_ts", "updated_ts"):
                if col in frame.columns:
                    frame = frame.with_columns(
                        pl.lit(now, dtype=pl.Datetime).alias(col))
            bundle[name] = frame
        return bundle


class StreamProblemPipeline:
    """Compose the three problems from config; event-day oriented."""

    def __init__(self, rng, burst, replay, late) -> None:
        self.burst = burst
        self.duplicates = replay
        self.late = late

    @classmethod
    def from_config(cls, rng, config) -> "StreamProblemPipeline":
        p = config.problems
        return cls(
            rng,
            BurstTrafficProblem(
                p.burst.every_n_ticks, p.burst.multiplier),
            DuplicateReplayProblem(
                rng, p.duplicate_replay.rate, p.duplicate_replay.history_size),
            LateArrivalProblem(
                rng, p.late_arrival.rate,
                p.late_arrival.delay_minutes_min,
                p.late_arrival.delay_minutes_max),
        )

    def events_for_tick(self, tick: int, normal_events: int) -> int:
        return self.burst.events_for_tick(tick, normal_events)

    def apply_late(
        self, frame: pl.DataFrame, ts_col: str, now: datetime
    ) -> tuple[pl.DataFrame, int, int]:
        return self.late.apply(frame, ts_col, now)

    def remember(self, bundle: dict[str, pl.DataFrame]) -> None:
        self.duplicates.remember(bundle)

    def replay(self, now: datetime) -> dict[str, pl.DataFrame] | None:
        return self.duplicates.replay(now)
