"""Pretend clock for simulated time (revived, 2026-10-29).

Manages a simulated calendar and detects month / bureau / payment
boundaries. Two layers:

1. **Restart-proof date predicates** - pure functions of a date, no state.
   A resumed process can derive every boundary day from the date alone.
        - ``is_month_end(d)``        is d the last day of its month?
        - ``is_bureau_drop_day(d)``  is d the first day the lagged vendor
                                     extract (d - bureau_lag_days) entered a
                                     new month? (fires ~ once a month)
        - nightly payment: any new day (predicate = always for a fresh day)

2. **Stateful pacing** - ``tick()`` advances the date, evaluates the
   predicates against ``_last_*_processed`` markers, reports booleans and
   advances the markers. Intended caller: the online pull-based emitter
   (tick day-by-day; flush hot stream / nightly file / monthly close).

Boundary flags are booleans, not counts: a ``tick(days>1)`` jump that lands
past several boundaries reports them as one event (or misses them). Tick
day-by-day for full boundary fidelity.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Iterator


class SimClock:
    """Pretend clock driving the simulation schedule."""

    def __init__(
        self,
        start_date: date = date(2020, 1, 1),
        bureau_lag_days: int = 21,
    ) -> None:
        self.start_date = start_date
        self.current_date = start_date
        self.bureau_lag_days = int(bureau_lag_days)
        self._last_month_processed: date | None = None
        self._last_bureau_processed: date | None = None
        self._last_payment_processed: date | None = None

    # ------------------------------------------------------------------
    # Restart-proof date predicates (pure, no state)
    # ------------------------------------------------------------------

    @staticmethod
    def is_month_end(d: date) -> bool:
        """True if d is the last day of its month (a monthly close day)."""
        return (d + timedelta(days=1)).month != d.month

    def is_bureau_drop_day(self, d: date) -> bool:
        """True if the lagged vendor extract entered a new month on d.

        The bureau drop for month M is ``as-of`` the 1st and arrives
        ``bureau_lag_days`` later: (d - lag).day == 1 means a fresh extract
        just became available.
        """
        return (d - timedelta(days=self.bureau_lag_days)).day == 1

    @property
    def current_month(self) -> date:
        return self.current_date.replace(day=1)

    # ------------------------------------------------------------------
    # Stateful boundary flags (pacing)
    # ------------------------------------------------------------------

    @property
    def is_month_boundary(self) -> bool:
        """Month-end close, fired once per month (not on the month's first
        days - a fresh start ticks day-by-day and only fires at month end)."""
        if not self.is_month_end(self.current_date):
            return False
        if self._last_month_processed is None:
            return True
        return (self.current_date.replace(day=1)
                != self._last_month_processed.replace(day=1))

    @property
    def is_bureau_boundary(self) -> bool:
        """A new bureau month became available (drop day, unfired yet)."""
        if not self.is_bureau_drop_day(self.current_date):
            return False
        if self._last_bureau_processed is None:
            return True
        lagged = (self.current_date - timedelta(days=self.bureau_lag_days))
        return (lagged.replace(day=1)
                != self._last_bureau_processed.replace(day=1))

    @property
    def is_payment_boundary(self) -> bool:
        """A new day arrived (nightly payment file cadence)."""
        return (self._last_payment_processed is None
                or self.current_date > self._last_payment_processed)

    # ------------------------------------------------------------------
    # Pacing
    # ------------------------------------------------------------------

    def tick(self, days: int = 1) -> dict[str, bool]:
        self.current_date += timedelta(days=days)
        monthly = self.is_month_boundary
        bureau = self.is_bureau_boundary
        payment = self.is_payment_boundary
        if monthly:
            self._last_month_processed = self.current_date.replace(day=1)
        if bureau:
            self._last_bureau_processed = self.current_date
        if payment:
            self._last_payment_processed = self.current_date
        return {"monthly": monthly, "bureau": bureau, "payment": payment}

    def tick_to_month_end(self) -> dict[str, bool]:
        """Tick to the last day of the current month.

        If the current month is already closed (we are on/past its last
        day), close the NEXT month instead - so a caller may invoke it in
        sequence without ever ticking 0 days.
        """
        target = self._last_day_of(self.current_date)
        if target <= self.current_date:
            base = (self.current_date.replace(day=1)
                    + timedelta(days=32)).replace(day=1)
            target = self._last_day_of(base)
        return self.tick(days=(target - self.current_date).days)

    @staticmethod
    def _last_day_of(d: date) -> date:
        """Last day of the month containing d (any month's 28th + 4 days is
        guaranteed to be in the next month)."""
        nxt = d.replace(day=28) + timedelta(days=4)
        return nxt - timedelta(days=nxt.day)

    def month_iterator(self, n_months: int) -> Iterator[date]:
        """Yield the month just completed after each month-end tick.

        Starting on the 1st of month M, this yields M, M+1, ... M+n-1
        (first-days of the completed months).
        """
        for _ in range(n_months):
            self.tick_to_month_end()
            yield self.current_date.replace(day=1)
