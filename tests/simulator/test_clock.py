"""SimClock revival: date predicates + boundary pacing semantics."""

from __future__ import annotations

from datetime import date

from simulator.clock import SimClock


def test_pure_date_predicates():
    clock = SimClock(bureau_lag_days=21)
    # month end: Jan 31 yes, Jan 30 no, Feb 28 yes (2020 leap-friendly)
    assert clock.is_month_end(date(2020, 1, 31))
    assert not clock.is_month_end(date(2020, 1, 30))
    assert clock.is_month_end(date(2020, 2, 29))
    # bureau drop day: lagged date (d - 21) is the 1st -> d is the 22nd
    assert clock.is_bureau_drop_day(date(2020, 1, 22))
    assert not clock.is_bureau_drop_day(date(2020, 1, 21))
    assert clock.is_bureau_drop_day(date(2020, 2, 22))


def test_pure_predicates_are_restart_proof():
    # a fresh clock (no state) evaluates the same boundary days as a clock
    # that has already ticked through them
    fresh = SimClock(bureau_lag_days=21)
    used = SimClock(bureau_lag_days=21)
    for _ in range(70):           # tick well past the boundary days
        used.tick()
    for d in (date(2020, 1, 22), date(2020, 2, 22), date(2020, 1, 31)):
        assert fresh.is_bureau_drop_day(d) == used.is_bureau_drop_day(d)
        assert fresh.is_month_end(d) == used.is_month_end(d)


def test_day_ticking_month_boundary_fires_only_at_month_end():
    clock = SimClock(start_date=date(2020, 1, 1), bureau_lag_days=21)
    fired: list[date] = []
    for _ in range(60):            # Jan + Feb, day by day
        flags = clock.tick(days=1)
        if flags["monthly"]:
            fired.append(clock.current_date)
    # fires exactly on the month-end days, never on the 1st/2nd
    assert fired == [date(2020, 1, 31), date(2020, 2, 29)]


def test_day_ticking_bureau_boundary_fires_on_drop_days():
    clock = SimClock(start_date=date(2020, 1, 1), bureau_lag_days=21)
    fired: list[date] = []
    for _ in range(60):
        flags = clock.tick(days=1)
        if flags["bureau"]:
            fired.append(clock.current_date)
    # Jan's extract becomes available on Jan 22, Feb's on Feb 22
    assert fired == [date(2020, 1, 22), date(2020, 2, 22)]


def test_payment_boundary_fires_every_new_day():
    clock = SimClock(start_date=date(2020, 1, 1))
    flags = clock.tick(days=1)
    assert flags["payment"] is True        # first day
    assert flags["monthly"] is False       # not month end
    flags = clock.tick(days=1)
    assert flags["payment"] is True
    # same-day re-check does not re-fire
    clock._last_payment_processed = clock.current_date
    assert clock.is_payment_boundary is False


def test_tick_to_month_end_lands_exactly_on_last_day():
    clock = SimClock(start_date=date(2020, 1, 15))
    flags = clock.tick_to_month_end()
    assert clock.current_date == date(2020, 1, 31)
    assert flags["monthly"] is True
    assert flags["bureau"] is False


def test_month_iterator_yields_completed_months():
    clock = SimClock(start_date=date(2020, 1, 1), bureau_lag_days=21)
    months = list(clock.month_iterator(3))
    assert months == [date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1)]
    assert clock.current_date == date(2020, 3, 31)


def test_large_jump_skips_boundaries_boolean_not_count():
    # documenting the contract: a multi-day jump reports at most one event
    # per flag (or misses boundaries entirely) - tick day-by-day for fidelity
    clock = SimClock(start_date=date(2020, 1, 1), bureau_lag_days=21)
    flags = clock.tick(days=31)            # lands on Feb 1, past Jan 31
    assert clock.current_date == date(2020, 2, 1)
    assert flags["monthly"] is False        # Jan 31 was jumped over
    assert flags["bureau"] is False         # Jan 22 was jumped over