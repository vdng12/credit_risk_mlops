"""Data quality validator.

Two layers:

1. Per-entity checks  -- emptiness, null fractions, ID nulls, date sanity
2. Referential integrity -- FK links between child and parent entities
   (payments -> loans, applications/cards/bureau -> clients, etc.)

The FK checks are what verify the structural dependencies the ledger is
meant to uphold.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Optional

import polars as pl

# Entity name -> missing-rate config key, used for null thresholds
_ENTITY_KEYS = {
    "client": "client",
    "application": "application",
    "card_auth": "card_auth",
    "pos_balance": "pos_balance",
    "card_balance": "card_balance",
    "bureau": "bureau",
    "payment": "payment",
}

_ID_COLS = (
    "client_id", "application_id", "auth_id", "card_id", "loan_id",
    "bureau_id", "payment_id", "pos_balance_id", "card_balance_id",
)

# Columns that are null by design for a subset of rows (conditional nulls), so
# high null fractions are expected and not a data-quality issue.
_STRUCTURAL_NULL_COLS = {"decline_reason"}


class QualityValidator:
    def __init__(self, config) -> None:
        self.cfg = config

    # ------------------------------------------------------------------
    # Batch-level entrypoint: per-entity + referential integrity
    # ------------------------------------------------------------------

    def validate_batch(
        self,
        entities: dict[str, pl.DataFrame],
        snapshot_date: date,
    ) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        for name, df in entities.items():
            issues.extend(self.validate(df, name, snapshot_date))
        issues.extend(self.validate_referential_integrity(entities))
        return issues

    # ------------------------------------------------------------------
    # Per-entity checks
    # ------------------------------------------------------------------

    def validate(
        self,
        df: pl.DataFrame,
        entity_type: str,
        snapshot_date: Optional[date] = None,
    ) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        issues.extend(self._check_empty(df, entity_type))
        issues.extend(self._check_null_fraction(df, entity_type))
        issues.extend(self._check_id_nulls(df, entity_type, snapshot_date))
        issues.extend(self._check_date_ranges(df, entity_type, snapshot_date))
        return issues

    def _check_empty(self, df: pl.DataFrame, entity: str) -> list[dict]:
        if df.height == 0:
            return [{"entity": entity, "check": "empty", "severity": "warn",
                     "detail": "DataFrame has 0 rows"}]
        return []

    def _check_null_fraction(self, df: pl.DataFrame, entity: str) -> list[dict]:
        issues = []
        cfg_key = _ENTITY_KEYS.get(entity, entity)
        rate = getattr(self.cfg.problems, f"{cfg_key}_missing_rate", 0.01)
        for col in df.columns:
            if col in _STRUCTURAL_NULL_COLS:
                continue
            null_frac = df[col].null_count() / max(1, df.height)
            # tolerate injected missing rate plus a margin; flag only gross excess
            if null_frac > max(0.25, rate * 4 + 0.05):
                issues.append({
                    "entity": entity, "check": "null_fraction", "severity": "warn",
                    "column": col, "value": round(null_frac, 4),
                    "detail": f"{null_frac:.1%} null in {col}",
                })
        return issues

    def _check_id_nulls(
        self, df: pl.DataFrame, entity: str, snapshot_date: Optional[date]
    ) -> list[dict]:
        issues = []
        for col in _ID_COLS:
            if col in df.columns:
                n_null = df[col].null_count()
                if n_null > 0:
                    issues.append({
                        "entity": entity, "check": "null_ids", "severity": "error",
                        "column": col,
                        "detail": f"{n_null} null IDs in {col}",
                    })
                # also flag empty-string ids
                if df[col].dtype == pl.Utf8:
                    n_empty = df.filter(pl.col(col) == "").height
                    if n_empty > 0:
                        issues.append({
                            "entity": entity, "check": "empty_ids",
                            "severity": "error", "column": col,
                            "detail": f"{n_empty} empty-string IDs in {col}",
                        })
        return issues

    def _check_date_ranges(
        self, df: pl.DataFrame, entity: str, snapshot_date: Optional[date]
    ) -> list[dict]:
        issues = []
        if snapshot_date is None:
            snapshot_date = date.today()
        too_early = snapshot_date - timedelta(days=365 * 100)
        too_late = snapshot_date + timedelta(days=62)

        def _as_date(v):
            return v.date() if isinstance(v, datetime) else v

        for col, dtype in zip(df.columns, df.dtypes):
            if dtype not in (pl.Date, pl.Datetime):
                continue
            vals = df[col].drop_nulls()
            if vals.len() == 0:
                continue
            min_v = _as_date(vals.min())
            max_v = _as_date(vals.max())
            if min_v is not None and min_v < too_early:
                issues.append({
                    "entity": entity, "check": "date_range", "severity": "error",
                    "column": col,
                    "detail": f"min date {min_v} older than {too_early.year}",
                })
            if max_v is not None and max_v > too_late:
                issues.append({
                    "entity": entity, "check": "date_range", "severity": "error",
                    "column": col,
                    "detail": f"max date {max_v} beyond {too_late}",
                })
        return issues

    # ------------------------------------------------------------------
    # Referential integrity (structural dependency checks)
    # ------------------------------------------------------------------

    def validate_referential_integrity(
        self, entities: dict[str, pl.DataFrame]
    ) -> list[dict[str, Any]]:
        """Check child -> parent FK links across entities in one month."""
        issues: list[dict[str, Any]] = []

        clients = entities.get("client")
        applications = entities.get("application")
        card_auths = entities.get("card_auth")
        pos = entities.get("pos_balance")
        card_bal = entities.get("card_balance")
        bureau = entities.get("bureau")
        payments = entities.get("payment")

        def ids(df: Optional[pl.DataFrame], col: str) -> set:
            if df is None or col not in df.columns:
                return set()
            return set(df[col].drop_nulls().to_list())

        client_ids = ids(clients, "client_id")

        # 1. every child entity's client references a real client
        for name, df in [("application", applications), ("card_auth", card_auths),
                         ("pos_balance", pos), ("card_balance", card_bal),
                         ("bureau", bureau), ("payment", payments)]:
            child_ids = ids(df, "client_id")
            orphans = child_ids - client_ids
            if orphans:
                issues.append({
                    "entity": name, "check": "fk_client", "severity": "error",
                    "detail": f"{len(orphans)} rows reference unknown clients",
                })

        # 2. payments -> loans (every payment references an active loan)
        loan_ids = ids(pos, "loan_id")
        pay_loans = ids(payments, "loan_id")
        orphan_pay = pay_loans - loan_ids
        if orphan_pay:
            issues.append({
                "entity": "payments", "check": "fk_loan", "severity": "error",
                "detail": f"{len(orphan_pay)} payments reference unknown loan_id",
            })

        # 3. card balances -> cards (card_balance.card_id consistent with
        #    card_auths.card_id stream)
        auth_cards = ids(card_auths, "card_id")
        bal_cards = ids(card_bal, "card_id")
        # card_id only exists in auths/balances; they should overlap on
        # the cardholder population (auths are a superset of active use)
        if bal_cards and auth_cards:
            # auth cards are ephemeral event ids, so just check nonzero overlap
            if not (bal_cards & auth_cards):
                issues.append({
                    "entity": "card_balances", "check": "fk_card",
                    "severity": "warn",
                    "detail": "card_balance and card_auth card_ids do not overlap",
                })

        # 4. bureau ids unique and stable within month (no duplicate accounts)
        if bureau is not None and "bureau_id" in bureau.columns:
            n_dup = bureau.height - bureau["bureau_id"].n_unique()
            if n_dup > 0:
                issues.append({
                    "entity": "bureau", "check": "dup_bureau_id",
                    "severity": "error",
                    "detail": f"{n_dup} duplicate bureau_id in one snapshot",
                })

        return issues