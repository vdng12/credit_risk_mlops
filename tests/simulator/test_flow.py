"""Run-flow month sequencing (play_forward semantics)."""

from __future__ import annotations

from datetime import date

from simulator import SimConfig, Simulator


def _sim(n=100, start=date(2020, 1, 1), seed=7):
    cfg = SimConfig(n_customers=n, start_date=start, seed=seed)
    cfg.sinks.enable_s3 = cfg.sinks.enable_postgres = False
    return Simulator(cfg)


def test_forward_only_includes_the_start_month():
    sim = _sim()
    months = [b.month for b in sim.play_forward(3)]
    assert months == [date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1)]


def test_forward_after_backfill_continues_after_boundary():
    sim = _sim()
    assert [b.month for b in sim.backfill("2020-02-28")] == [
        date(2020, 1, 1), date(2020, 2, 1)]
    months = [b.month for b in sim.play_forward(2)]
    assert months == [date(2020, 3, 1), date(2020, 4, 1)]


def test_backfill_then_forward_no_gap_no_overlap():
    sim = _sim()
    all_months = (
        [b.month for b in sim.backfill("2020-03-31")]
        + [b.month for b in sim.play_forward(2)])
    assert sorted(all_months) == all_months           # strictly increasing
    assert len(set(all_months)) == len(all_months)    # no duplicates
    assert all_months == [
        date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1),
        date(2020, 4, 1), date(2020, 5, 1)]


def test_next_month_after_fresh_start_is_the_start_month():
    sim = _sim()
    assert sim.next_month().month == date(2020, 1, 1)