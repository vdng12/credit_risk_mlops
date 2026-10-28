"""Phase 0 determinism tests: per-(entity, month) indexed RNG streams.

Guards the property that a month's generated content is a pure function
of (seed, entity, month), enabling:

- byte-identical replay of a run (same seed -> same months)
- idempotent months (retry / resume / parallel generation safe)
- pull-based per-day emission without disturbing other streams

The statistical properties are covered by tests/test_statistics.py; these
tests only assert identity, ordering, and sensitivity.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from simulator import SimConfig, Simulator
from simulator.randstreams import month_rng
from simulator.problems import ProblemInjector

_ENTITY_ATTRS = (
    "clients",
    "applications",
    "application_decisions",
    "application_master",
    "card_auths",
    "pos_balances",
    "card_balances",
    "bureau",
    "payments",
)


def _run(
    seed: int, n_customers: int = 300,
    start: date = date(2020, 1, 1), until: date = date(2020, 4, 30),
):
    cfg = SimConfig(n_customers=n_customers, start_date=start, seed=seed)
    cfg.sinks.enable_s3 = False
    cfg.sinks.enable_postgres = False
    return Simulator(cfg).backfill(until)


def _frames_equal(fa: pl.DataFrame | None, fb: pl.DataFrame | None) -> bool:
    if fa is None or fb is None:
        return fa is None and fb is None
    return fa.equals(fb)


def test_same_seed_reproduces_identical_months():
    a = _run(42)
    b = _run(42)
    assert len(a) == len(b)
    for ba, bb in zip(a, b):
        assert ba.month == bb.month
        for name in _ENTITY_ATTRS:
            fa, fb = getattr(ba, name), getattr(bb, name)
            assert _frames_equal(fa, fb), (
                f"month={ba.month} entity={name} differs across "
                "same-seed runs")


def test_different_seed_gives_different_data():
    a = _run(42)
    b = _run(1337)
    # at least one entity differs in every month
    for ba, bb in zip(a, b):
        assert any(
            not _frames_equal(getattr(ba, name), getattr(bb, name))
            for name in _ENTITY_ATTRS
        ), f"month={ba.month}: different seeds produced identical data"


def test_problems_layer_is_per_month_deterministic():
    """Same clean frame + same (entity, month) -> same dirty frame, even on
    fresh injector instances (the corruption stream is derived, not
    sequential across entities/months)."""
    clean = pl.DataFrame({
        "client_id": [1, 2, 3, 4, 5],
        "income_total": [10.0, 20.0, 30.0, 40.0, 50.0],
        "is_cardholder": [True, False, True, False, True],
    })
    cfg = SimConfig(n_customers=5, start_date=date(2020, 1, 1), seed=42)
    a = ProblemInjector(cfg, seed=7)
    b = ProblemInjector(cfg, seed=7)
    m = date(2020, 3, 1)
    assert a.apply(clean, "client", m).equals(b.apply(clean, "client", m))
    # the same month re-derived identically even after another month was
    # injected in between (per-month purity, not a shared tape)
    a.apply(clean, "client", date(2020, 2, 1))
    assert a.apply(clean, "client", m).equals(b.apply(clean, "client", m))


def test_duplicates_only_target_event_stream_entities():
    """Exact duplicates are scoped to the event-stream entities with a
    realistic failure mode (payment file re-delivery, submission retry);
    other entities never grow rows from duplication."""
    cfg = SimConfig(n_customers=10, start_date=date(2020, 1, 1), seed=42)
    cfg.problems.duplicate_prob = 0.0       # legacy generic knob off
    cfg.problems.payment_duplicate_rate = 0.5
    cfg.problems.application_duplicate_rate = 0.5
    inj = ProblemInjector(cfg, seed=7)
    m = date(2020, 1, 1)
    pay = pl.DataFrame({"payment_id": range(10), "amount": [1.0] * 10})
    app = pl.DataFrame({"application_id": range(10), "loan_amount": [100.0] * 10})
    cli = pl.DataFrame({"client_id": range(10), "income_total": [1.0] * 10})
    assert inj.apply(pay, "payment", m).height == 15
    assert inj.apply(app, "application", m).height == 15
    # clients never get duplicates (no rows added), only possible missing
    assert inj.apply(cli, "client", m).height == 10


def test_month_rng_is_stable_and_sensitive():
    m = date(2020, 3, 1)
    # determinism: same inputs -> identical first draw
    g1 = month_rng(42, "applications", m)
    g2 = month_rng(42, "applications", m)
    assert g1.random() == g2.random()
    # sensitivity: entity, month, and root seed all change the stream
    draws = {
        month_rng(42, "applications", m).random(),
        month_rng(42, "card_auths", m).random(),
        month_rng(42, "applications", date(2020, 4, 1)).random(),
        month_rng(7, "applications", m).random(),
    }
    assert len(draws) == 4
    # monotone month progression must not collide either
    d1 = month_rng(42, "ledger", date(2020, 1, 1)).random()
    d2 = month_rng(42, "ledger", date(2020, 2, 1)).random()
    assert d1 != d2
