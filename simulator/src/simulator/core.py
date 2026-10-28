"""Core stateful iterator simulator.

The Simulator maintains a persistent client pool (ids + 7-factor vectors)
and an account ledger (loans, cards, bureau accounts). Each month it:

  1. snapshots clients from the stable factor matrix
  2. generates loan applications that inherit the applicant's factors,
     originating approved loans into the ledger
  3. advances the ledger one month (payments, DPD, bureau STATUS Markov)
  4. emits child snapshots (POS, card balance, bureau, payments) from the
     ledger so FK links and temporal panels hold across months
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Iterator, Optional

import numpy as np
import polars as pl

from simulator.clock import SimClock
from simulator.copula import CopulaEngine
from simulator.entities import (
    generate_card_auths,
    generate_clients,
    generate_loan_applications,
)
from simulator.params import Params
from simulator.problems import ProblemInjector
from simulator.quality import QualityValidator
from simulator.randstreams import month_rng
from simulator.sinks import SinkManager
from simulator.state import AccountLedger, draw_from_model


@dataclass
class MonthBatch:
    """All entities produced during one simulated month.

    Two release paths - both COLD-first; Kafka is CDC-derived downstream:

      OLTP source tables (streamed by Debezium CDC on postgres-source):
          clients               -- monthly customer master
          loan_applications     -- pre-decision submission events
          card_authorizations   -- authorization events (incl. declined)
          application_decisions -- nightly outcome rows (status, decision)
          payments              -- nightly instalment payment file

      Analytical / object store:
          application_master    -- submitted + decision joined (the master
                                  table with cutover semantics; gold layer)
          pos_balances          -- monthly instalment balance
          card_balances         -- monthly card balance
          bureau                -- external bureau monthly drop

      Kafka/Redpanda topics are NOT written by the generator; the Debezium
      connector on postgres-source re-streams the OLTP tables.
    """

    month: date
    clients: Optional[pl.DataFrame] = None
    applications: Optional[pl.DataFrame] = None
    application_decisions: Optional[pl.DataFrame] = None
    application_master: Optional[pl.DataFrame] = None
    card_auths: Optional[pl.DataFrame] = None
    pos_balances: Optional[pl.DataFrame] = None
    card_balances: Optional[pl.DataFrame] = None
    bureau: Optional[pl.DataFrame] = None
    payments: Optional[pl.DataFrame] = None
    issues: list = field(default_factory=list)

    # ------------------------------------------------------------------
    # Hot / cold path split
    # ------------------------------------------------------------------

    def hot_events(self) -> dict[str, Optional[pl.DataFrame]]:
        """Stream-shaped entities: written to OLTP tables, re-streamed by CDC."""
        return {
            "loan_applications": self.applications,     # pre-decision
            "card_authorizations": self.card_auths,
        }

    def cold_batches(self) -> dict[str, Optional[pl.DataFrame]]:
        """Batch entities: snapshot/partition writes -> object store + PG."""
        return {
            "clients": self.clients,
            "application_decisions": self.application_decisions,
            "application_master": self.application_master,
            "pos_balances": self.pos_balances,
            "card_balances": self.card_balances,
            "bureau": self.bureau,
            "payments": self.payments,
        }

    def all_entities(self) -> dict[str, Optional[pl.DataFrame]]:
        """Every entity produced this month (cold + stream-shaped)."""
        return {**self.cold_batches(), **self.hot_events()}

    def to_sinks(
        self,
        sink: SinkManager,
        run_id: str = "incremental",
    ) -> list[str]:
        paths = []
        month_str = self.month.strftime("%Y-%m")

        # design routing (AGENTS.md, Data generator section): every entity lands in
        # the local file sink as write-through archive; object-store entities
        # go to S3; the OLTP + gold entities go to Postgres lending_db (the
        # Debezium CDC connector turns the OLTP tables into the Kafka stream).
        # Sink failures are loud - nothing is silently swallowed.
        s3_entities = {"clients", "application_decisions",
                       "pos_balances", "card_balances", "bureau"}
        pg_entities = {"clients", "loan_applications", "card_authorizations",
                       "application_decisions", "application_master", "payments"}

        for name, df in self.all_entities().items():
            if df is None or df.height == 0:
                continue
            paths.append(sink.to_parquet(df, name, month_str, run_id))
            if name in s3_entities and sink.s3_enabled:
                paths.append(sink.to_s3(df, name, month_str, run_id))
            if name in pg_entities and sink.postgres_enabled:
                sink.to_postgres(df, table_name=name)
                paths.append(f"postgres:{name}")
        return paths


class Simulator:
    """Stateful t-copula simulator with structural FK + temporal dependencies."""

    def __init__(self, config) -> None:
        self.cfg = config
        self.seed = config.seed
        self.engine = CopulaEngine(
            nu=config.copula.nu,
            corr_matrix=np.array(config.copula.corr_matrix),
            seed=config.seed,
        )
        self.clock = SimClock(
            start_date=config.time.start_date,
            bureau_lag_days=config.time.bureau_drop_lag_days,
        )
        self.injector = ProblemInjector(config, seed=config.seed + 1)
        self.validator = QualityValidator(config)
        self.sink = SinkManager(config)
        self._rng = np.random.default_rng(config.seed + 2)

        self.params = Params()
        self.ledger = AccountLedger(
            self.params, self._rng,
            root_seed=self.seed,
            target_default_rate=self.cfg.entities.target_default_rate,
            performance_window=self.cfg.entities.performance_window_months,
            trigger_months=self.cfg.entities.default_trigger_months)
        self._client_static: Optional[pl.DataFrame] = None

        self._client_ids: Optional[list[str]] = None
        self._client_factors: Optional[np.ndarray] = None
        self._is_cardholder: Optional[np.ndarray] = None
        self._is_loanholder: Optional[np.ndarray] = None
        self._ledger_initialized: bool = False
        self._current_month: Optional[date] = None
        self._historical_master: Optional[pl.DataFrame] = None

    # ------------------------------------------------------------------
    # Client pool bootstrap
    # ------------------------------------------------------------------

    def _ensure_client_pool(self) -> None:
        """Sample the stable per-client factor matrix and flags once."""
        if self._client_ids is None:
            n = self.cfg.entities.n_customers
            self._client_ids = [f"C{i:08d}" for i in range(n)]
            self._client_factors = self.engine.sample_latent_factors(n)
            # stable cardholder / loanholder flags from the fixed factors
            self._is_cardholder = self.engine.bernoulli(
                self._client_factors[:, 5],
                self.cfg.entities.card_penetration, factor_weight=0.25)
            self._is_loanholder = self.engine.bernoulli(
                self._client_factors[:, 2],
                self.cfg.entities.loan_penetration, factor_weight=0.30)
            # target_default_rate is defined over *originated* loans.
            # Applicants are drawn uniformly from the pool and approved with
            # p_approved(f), so the borrower population is the pool weighted
            # by approval probability.
            from simulator.state import _binned_rate
            w = _binned_rate(self.params.model("previous_application", "p_approved"),
                             self._client_factors, self.params.factor_idx) \
                if self.params.model("previous_application", "p_approved")["kind"] == "binned" \
                else np.ones(n)
            self.ledger.calibrate_default_level(self._client_factors, weights=w)

    def _originate_initial_accounts(self, clients: pl.DataFrame) -> None:
        """Seed the ledger with pre-existing bureau, cards and loans."""
        F = self._client_factors
        cid = np.array(self._client_ids)
        is_card = clients["is_cardholder"].to_numpy()
        is_loan = clients["is_loanholder"].to_numpy()

        # Everyone has a bureau history
        if self.cfg.entities.generate_bureau:
            self.ledger.originate_bureau({"client_id": cid, "factors": F})

        # Cardholders get cards
        if self.cfg.entities.generate_cards and is_card.any():
            Fc = F[is_card]
            limits = np.exp(draw_from_model(
                self.params.model("credit_card_balance", "log_credit_limit"),
                Fc, self._rng, self.params.factor_idx))
            util = np.clip(draw_from_model(
                self.params.model("credit_card_balance", "utilization"),
                Fc, self._rng, self.params.factor_idx), 0.0, 1.2)
            self.ledger.originate_cards({
                "client_id": cid[is_card],
                "factors": Fc,
                "limit": limits,
                "utilization": util,
            })

        # Loanholders get a pre-existing loan. These are *historical*
        # applications -> a one-time backfill of the application master.
        if is_loan.any():
            Fl = F[is_loan]
            n_hist = int(is_loan.sum())
            principal = np.exp(draw_from_model(
                self.params.model("previous_application", "log_amt_credit"),
                Fl, self._rng, self.params.factor_idx))
            term = np.clip(draw_from_model(
                self.params.model("previous_application", "term_months"),
                Fl, self._rng, self.params.factor_idx), 6, 72).astype(int)
            rate = np.clip(
                5.0 + (8 - Fl[:, 0] * 1.5)
                + self._rng.standard_normal(n_hist) * 1.5, 3.0, 25.0)
            mr = rate / 1200.0
            annuity = principal * mr / (1 - (1 + mr) ** (-term))
            hist_app_ids = np.array([f"APP-HIST-{i:07d}" for i in range(n_hist)])
            self.ledger.originate_loans({
                "application_id": hist_app_ids,
                "client_id": cid[is_loan],
                "factors": Fl,
                "term_months": term,
                "principal": principal,
                "annuity": annuity,
            })
            # pre-advance the initial book through its unobserved history
            # with each loan's own hazard (DPD state consistent with risk)
            hist_loans = self.ledger.loans[-n_hist:]
            self.ledger.burn_in(hist_loans, np.array([
                int(self._rng.integers(1, max(2, l.term_months)))
                for l in hist_loans]))

            # one-time backfill of the application master for the
            # pre-existing (historical) loan book
            self._historical_master = pl.DataFrame({
                "application_id": [l.application_id for l in self.ledger.loans],
                "client_id": [l.client_id for l in self.ledger.loans],
                "submission_timestamp": [
                    datetime.fromordinal(
                        (self.cfg.time.start_date
                         - timedelta(days=30 * l.months_paid)).toordinal()
                        + 1)
                    for l in self.ledger.loans],
                "loan_amount": [float(l.principal) for l in self.ledger.loans],
                "loan_term_months": [l.term_months for l in self.ledger.loans],
                "interest_rate": [None] * len(self.ledger.loans),
                "purpose": ["historical"] * len(self.ledger.loans),
                "annuity": [float(l.annuity) for l in self.ledger.loans],
                "goods_price": [None] * len(self.ledger.loans),
                "contract_type": ["Cash loans"] * len(self.ledger.loans),
                "status": ["approved"] * len(self.ledger.loans),
                "decision_timestamp": [
                    datetime.fromordinal(
                        (self.cfg.time.start_date
                         - timedelta(days=30 * l.months_paid - 5)).toordinal()
                        + 1)
                    for l in self.ledger.loans],
                "decision_by": ["UW-HIST"] * len(self.ledger.loans),
            })

    # ------------------------------------------------------------------
    # Month generation
    # ------------------------------------------------------------------

    def _generate_month(self, month: date) -> MonthBatch:
        self._ensure_client_pool()
        n = len(self._client_ids)
        F = self._client_factors

        # Client attributes are drawn ONCE (the non-factor noise in age,
        # children, employment... used to be re-drawn every month, so a
        # client's age jumped by years between snapshots). Each month only
        # the snapshot date and derived age move.
        if self._client_static is None:
            self._client_static = generate_clients(
                self.engine, n, month,
                client_ids=self._client_ids,
                card_penetration=self.cfg.entities.card_penetration,
                loan_penetration=self.cfg.entities.loan_penetration,
                rng=self._rng,
                factors=F,
                is_cardholder=self._is_cardholder,
                is_loanholder=self._is_loanholder,
            )
        clients = self._client_static.with_columns([
            pl.lit(month).alias("snapshot_date"),
            ((pl.lit(month) - pl.col("birth_date")).dt.total_days() // 365.25)
            .cast(pl.Int32).alias("age_years"),
            pl.lit(datetime(month.year, month.month, month.day)).alias("created_at"),
        ])

        if not self._ledger_initialized:
            self._originate_initial_accounts(clients)
            self._ledger_initialized = True

        # Stream: loan applications (pre-decision view + decision outcome).
        # Per-month stream: the month's applications are a pure function of
        # (seed, "applications", month) and replay byte-identically.
        applications, decisions, approvals = generate_loan_applications(
            self.engine, clients, F, month, self.params,
            monthly_app_rate=self.cfg.entities.monthly_app_rate,
            rng=month_rng(self.seed, "applications", month),
        )
        self.ledger.originate_loans(approvals)
        # Unified master table: submitted + decision joined on application_id
        application_master = applications.join(
            decisions, on="application_id", how="left")
        # Prepend the one-time historical backfill rows (pre-existing loans)
        # only once, then rely on the stream + nightly CDC thereafter.
        if self._historical_master is not None:
            hist = self._historical_master.with_columns([
                pl.col("interest_rate").cast(pl.Float64),
                pl.col("goods_price").cast(pl.Float64),
                pl.col("submission_timestamp").cast(pl.Datetime),
                pl.col("decision_timestamp").cast(pl.Datetime),
            ]).select(application_master.columns)
            application_master = pl.concat([hist, application_master])
            self._historical_master = None  # backfill emitted once

        # Stream: card authorizations
        month_start = datetime(month.year, month.month, 1)
        if month.month == 12:
            next_month = datetime(month.year + 1, 1, 1)
        else:
            next_month = datetime(month.year, month.month + 1, 1)
        if self.cfg.entities.generate_card_auths:
            card_auths = generate_card_auths(
                self.engine, clients, month_start, next_month,
                auths_per_card_per_day=self.cfg.entities.auths_per_card_per_day,
                decline_rate=self.cfg.entities.auth_decline_rate,
                merchant_categories=self.cfg.entities.merchant_categories,
                rng=month_rng(self.seed, "card_auths", month),
                card_ids=[c.card_id for c in self.ledger.cards],
                client_factors=F,
            )
        else:
            card_auths = pl.DataFrame({"auth_id": [], "card_id": [],
                                      "client_id": [], "event_timestamp": [],
                                      "amount": [], "currency": [],
                                      "merchant_category": [], "merchant_name": [],
                                      "approved": [], "decline_reason": []})

        # Advance ledger (returns this month's payment rows), then emit
        # snapshots of the month-end state.
        payment_rows = self.ledger.advance_month(month)
        active_loan_ids = {r["loan_id"] for r in payment_rows}

        pos_balances = self.ledger.emit_pos_balances(month, active_loan_ids)
        card_balances = self.ledger.emit_card_balances(month)
        bureau = self.ledger.emit_bureau(month)
        payments = pl.DataFrame(payment_rows)

        # Inject data problems - clean-then-inject, one entity, one month.
        #
        # The entity generators above produced CLEAN, relationally
        # consistent frames; contamination is a separate stage that never
        # touches generation logic (see problems.py for the problem ->
        # entity scope contract - e.g. schema evolution is bureau-only,
        # exact duplicates are payments/applications only).
        #
        # application_decisions / application_master are deliberately NOT
        # injected: decisions is a derived outcome frame and master is the
        # join of applications + decisions, so the stream entities are the
        # single injection point - corrupting the derived views would
        # double-corrupt or break the join.
        #
        # Each apply() derives its corruption stream from
        # (root seed, entity, month) so the same clean month always yields
        # the same dirty month (Phase 0 indexed determinism).
        clients = self.injector.apply(clients, "client", month)
        applications = self.injector.apply(applications, "application", month)
        card_auths = self.injector.apply(card_auths, "card_auth", month)
        pos_balances = self.injector.apply(pos_balances, "pos_balance", month)
        card_balances = self.injector.apply(card_balances, "card_balance", month)
        bureau = self.injector.apply(bureau, "bureau", month)
        payments = self.injector.apply(payments, "payment", month)

        # Validate (per-entity + referential integrity)
        issues = self.validator.validate_batch(
            {
                "client": clients,
                "application": applications,
                "card_auth": card_auths,
                "pos_balance": pos_balances,
                "card_balance": card_balances,
                "bureau": bureau,
                "payment": payments,
            },
            snapshot_date=month,
        )

        return MonthBatch(
            month=month,
            clients=clients,
            applications=applications,
            application_decisions=decisions,
            application_master=application_master,
            card_auths=card_auths,
            pos_balances=pos_balances,
            card_balances=card_balances,
            bureau=bureau,
            payments=payments,
            issues=issues,
        )

    # ------------------------------------------------------------------
    # Public API: backfill / play forward / next month
    # ------------------------------------------------------------------

    def backfill(
        self,
        until: str | date,
        start: Optional[str | date] = None,
    ) -> list[MonthBatch]:
        if isinstance(until, str):
            until = date.fromisoformat(until)
        start = start or self.cfg.time.start_date
        if isinstance(start, str):
            start = date.fromisoformat(start)

        batches = []
        month = date(start.year, start.month, 1)
        while month <= until:
            batches.append(self._generate_month(month))
            month = self._next_month_date(month)
        self._current_month = until.replace(day=1)
        return batches

    def play_forward(self, months: int) -> Iterator[MonthBatch]:
        start = self._current_month or self.cfg.time.start_date
        for _ in range(months):
            month = self._next_month_date(start)
            batch = self._generate_month(month)
            self._current_month = month
            start = month
            yield batch

    def next_month(self) -> MonthBatch:
        return next(self.play_forward(1))

    @staticmethod
    def _next_month_date(month: date) -> date:
        if month.month == 12:
            return date(month.year + 1, 1, 1)
        return date(month.year, month.month + 1, 1)