"""Pretend clock for the simulator.

Manages simulated time, detects month/day boundaries, and provides
iteration over simulation periods.
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
        self.bureau_lag_days = bureau_lag_days
        self._last_month_processed: date | None = None
        self._last_bureau_processed: date | None = None
        self._last_payment_processed: date | None = None

    @property
    def current_month(self) -> date:
        return self.current_date.replace(day=1)

    @property
    def is_month_boundary(self) -> bool:
        m = self.current_date.replace(day=1)
        return m != self._last_month_processed

    @property
    def is_bureau_boundary(self) -> bool:
        if self._last_bureau_processed is None:
            return True
        bureau_date = self.current_date - timedelta(days=self.bureau_lag_days)
        bureau_month = bureau_date.replace(day=1)
        last_bureau_month = self._last_bureau_processed.replace(day=1)
        return bureau_month != last_bureau_month

    @property
    def is_payment_boundary(self) -> bool:
        if self._last_payment_processed is None:
            return True
        return self.current_date > self._last_payment_processed

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
        next_month = self.current_date.replace(day=28) + timedelta(days=4)
        last_day = next_month - timedelta(days=next_month.day)
        days = (last_day - self.current_date).days + 1
        return self.tick(days=days)

    def month_iterator(self, n_months: int) -> Iterator[date]:
        for _ in range(n_months):
            self.tick_to_month_end()
            yield self.current_date