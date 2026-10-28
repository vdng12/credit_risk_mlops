"""Vendor schema evolution over calendar time (bureau ONLY, Phase 4).

The external credit bureau changes its monthly drop's file schema:

    v1  [start, change_date)             no debt_ratio column, schema_version=1
    v2  [change_date, breaking_date)     debt_ratio populated,   schema_version=2
    v3  [breaking_date, ...)             credit_duration_days retired in
                                          favour of credit_duration_months,
                                          debt_ratio kept,        schema_version=3

A reader of the bureau table must therefore tolerate files with DIFFERENT
schemas inside one dataset - which is exactly the rubric's schema-evolution
problem. Only bureau is versioned: it is the only entity with an external
vendor (see problems.py scope contract).
"""

from __future__ import annotations

from datetime import date

import polars as pl

# The field the vendor adds in the non-breaking change (v2). It is derived
# from existing bureau data so v2 rows stay internally consistent.
NEW_FIELD = "debt_ratio"


class SchemaEvolutionProblem:
    def __init__(
        self,
        change_date: date | None = None,
        breaking_change_date: date | None = None,
        breaking_schema_version: int = 3,
    ) -> None:
        self.change_date = change_date
        self.breaking_change_date = breaking_change_date
        self.breaking_schema_version = int(breaking_schema_version)

    def version(self, month: date) -> int:
        if (self.breaking_change_date is not None
                and month >= self.breaking_change_date):
            return self.breaking_schema_version
        if self.change_date is not None and month >= self.change_date:
            return 2
        return 1

    def apply(self, df: pl.DataFrame, month: date) -> tuple[pl.DataFrame, int]:
        """Stamp schema_version and mutate the frame for the month's version.

        Returns (frame, version). v1 months have NO debt_ratio column at
        all - the schema difference between partitions is the signal.
        """
        version = self.version(month)
        if version == 1:
            return df.with_columns(
                pl.lit(1, dtype=pl.Int32).alias("schema_version")), 1

        out = df.with_columns(
            pl.lit(version, dtype=pl.Int32).alias("schema_version"))
        if NEW_FIELD not in out.columns:
            ratio = (
                pl.col("amt_credit_sum_overdue")
                / pl.when(pl.col("amt_credit_sum") > 0)
                      .then(pl.col("amt_credit_sum"))
                      .otherwise(1.0)
            ).alias(NEW_FIELD)
            out = out.with_columns(ratio)

        if version >= 3 and "credit_duration_days" in out.columns:
            out = out.drop("credit_duration_days").with_columns(
                (pl.col("days_credit").abs() / 30.0)
                .alias("credit_duration_months"))
        return out, version
