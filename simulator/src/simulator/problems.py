"""Data problem injection for the simulator.

Problem -> entity scope contract
===============================
Every data problem is deliberately scoped to a SUBSET of entities, because
it models a specific real-world failure mode that only makes sense there.
Applying a problem outside its scope would corrupt data with no realistic
cause and would break the validator's invariants.

    skew             -> clients (region/income concentration), card_auths
                        (merchant mix), applications (purpose mix)
                        [generation-time, Phase 4]
    schema evolution -> BUREAU ONLY: an external vendor adds/breaks a field
                        mid-history; other tables have no vendor
                        [Phase 4]
    exact duplicate  -> payments (re-delivered payment file, ~2%),
                        applications (submission retry bug, ~0.5%)
                        [Phase 4]
    late arrival     -> card_auths (mobile batching), payments/decisions
                        (file carrier delay) - STREAMING lane
                        [Phase 5]
    duplicate replay -> card_auths, applications (at-least-once re-delivery)
                        - STREAMING lane [Phase 5]
    burst traffic    -> card_auths, applications (end-of-month closings)
                        - STREAMING lane [Phase 5]

Legacy generic noise (missing values, outliers, batch date-shift)
-----------------------------------------------------------------
Pre-Phase-4 problems kept for continuity; they are entity-agnostic (wired
by per-entity rates in config) and are scheduled to be replaced by the
scoped rubric problems above.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
from numpy.random import Generator

from simulator.randstreams import month_rng


# Entities that receive exact-duplicate injection (rubric: "another offline
# problem, e.g. 2% duplicate rate"). Only these two - duplicates on any
# other table have no realistic cause and would corrupt FK/amount math.
_DUPLICATE_ENTITIES = frozenset({"payment", "application"})


class ProblemInjector:
    """Inject data quality problems into polars DataFrames."""

    def __init__(self, config, seed: int = 42) -> None:
        self.cfg = config
        self._root_seed = int(seed)
        self._rng: Generator = np.random.default_rng(seed)

    def apply(
        self, df: pl.DataFrame, entity_type: str, month: date | None = None
    ) -> pl.DataFrame:
        """Apply all enabled problems to a DataFrame.

        entity_type: 'client', 'application', 'card_auth', 'pos_balance',
                     'card_balance', 'bureau', or 'payment'.

        Phase 0: when a month is given, this (entity, month)'s corruption
        is a pure function of the root seed - re-applying it to the same
        clean frame reproduces the same dirty frame.
        """
        if not self.cfg.problems.enabled:
            return df

        if month is not None:
            self._rng = month_rng(
                self._root_seed, f"problems:{entity_type}", month
            )

        rate = getattr(self.cfg.problems, f"{entity_type}_missing_rate", 0.01)
        df = self._inject_missing(df, rate)
        # duplicates are event-stream entities only (see _DUPLICATE_ENTITIES)
        df = self._inject_duplicates(df, entity_type)
        df = self._inject_outliers(df)
        df = self._inject_late_arrivals(df)
        return df

    def _inject_missing(self, df: pl.DataFrame, rate: float) -> pl.DataFrame:
        if rate <= 0:
            return df
        n_rows, n_cols = df.shape
        mask = self._rng.random((n_rows, n_cols)) < rate
        exclude_suffixes = ("_id", "_date", "_timestamp", "snapshot_date",
                            "payment_date", "event_timestamp")
        result = df.clone()
        for j, col in enumerate(df.columns):
            if any(col.endswith(suf) for suf in exclude_suffixes):
                continue
            if mask[:, j].any():
                null_idx = list(np.where(mask[:, j])[0])
                vals = list(result[col])
                for i in null_idx:
                    vals[i] = None
                result = result.with_columns(pl.Series(col, vals))
        return result

    def _inject_duplicates(self, df: pl.DataFrame, entity_type: str) -> pl.DataFrame:
        """Exact-duplicate injection - payments + applications only.

        Scoped by _DUPLICATE_ENTITIES; the rate is per-entity
        (payment_duplicate_rate / application_duplicate_rate) so each table
        models its own failure mode (re-delivered payment file ~2% vs
        submission retry bug ~0.5%). Clones are byte-identical, so the
        validator's FK/amount invariants stay intact.
        """
        if entity_type not in _DUPLICATE_ENTITIES:
            return df
        rate = getattr(
            self.cfg.problems, f"{entity_type}_duplicate_rate", 0.0)
        if rate <= 0:
            return df
        n_dups = max(1, int(df.height * rate))
        idx = self._rng.choice(df.height, size=n_dups, replace=False)
        return pl.concat([df, df[idx]])

    def _inject_outliers(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.cfg.problems.outlier_rate <= 0:
            return df
        numeric_cols = [c for c, t in zip(df.columns, df.dtypes)
                        if t in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]
        if not numeric_cols:
            return df
        result = df.clone()
        for col in numeric_cols:
            mask = self._rng.random(df.height) < self.cfg.problems.outlier_rate
            if mask.any():
                multiplier = np.where(self._rng.random(mask.sum()) < 0.5, 10.0, 0.1)
                vals = result[col].to_numpy().astype(np.float64).copy()
                vals[mask] *= multiplier
                result = result.with_columns(pl.Series(col, vals))
        return result

    def _inject_late_arrivals(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.cfg.problems.late_arrival_prob <= 0:
            return df
        from datetime import timedelta
        date_cols = [c for c, t in zip(df.columns, df.dtypes)
                     if t in (pl.Date, pl.Datetime)]
        if not date_cols:
            return df
        mask = self._rng.random(df.height) < self.cfg.problems.late_arrival_prob
        if not mask.any():
            return df
        result = df.clone()
        max_days = self.cfg.problems.late_arrival_max_days
        for col in date_cols:
            shifts = self._rng.integers(1, max_days + 1, size=mask.sum())
            idx_list = list(np.where(mask)[0])
            new_vals = list(result[col])
            for i, offset in zip(idx_list, shifts):
                if new_vals[i] is not None:
                    new_vals[i] = new_vals[i] + timedelta(days=int(offset))
            result = result.with_columns(pl.Series(col, new_vals))
        return result