"""Data problem injection for the simulator.

Applies controlled corruptions: missing values, duplicates, outliers,
schema drift, late arrivals, and column swaps.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.random import Generator


class ProblemInjector:
    """Inject data quality problems into polars DataFrames."""

    def __init__(self, config, seed: int = 42) -> None:
        self.cfg = config
        self._rng: Generator = np.random.default_rng(seed)

    def apply(self, df: pl.DataFrame, entity_type: str) -> pl.DataFrame:
        """Apply all enabled problems to a DataFrame.

        entity_type: 'client', 'application', 'card_auth', 'pos_balance',
                     'card_balance', 'bureau', or 'payment'.
        """
        if not self.cfg.problems.enabled:
            return df

        rate = getattr(self.cfg.problems, f"{entity_type}_missing_rate", 0.01)
        df = self._inject_missing(df, rate)
        df = self._inject_duplicates(df)
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

    def _inject_duplicates(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.cfg.problems.duplicate_prob <= 0:
            return df
        n_dups = max(1, int(df.height * self.cfg.problems.duplicate_prob))
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