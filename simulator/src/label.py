"""Default label derivation (gold layer).

The simulator does NOT emit labels -- a default flag is *derived* downstream
from installment payments, arriving late. This module is that gold-layer job.

Label definition (performance window, like Home Credit's TARGET):
  * default = 1 if the loan reaches ``default_trigger_months`` consecutive
    delinquent months (days_late >= 30 * trigger) within its first
    ``observation_months`` instalments (months on book), else 0.
  * label_status
      - ``final``          the full window (min(term, W) instalments) has
                            been observed. A default seen early is NOT
                            final yet: finalising defaults early while
                            non-defaults wait W months over-samples
                            defaults in young vintages (measured: 9.6% vs
                            an 8% target).
      - ``pending``        window still open -> exclude from training
      - ``left_truncated`` loan originated before the data starts (first
                            observed instalment > 1); its window was not
                            fully observed, so it is excluded as well
This is the same quantity the simulator calibrates (state.RunHazard), so
the configured ``target_default_rate`` is what a gold job measures.
"""

from __future__ import annotations

import polars as pl

_GOLD_COLUMNS = [
    "application_id", "client_id", "loan_id",
    "submission_timestamp", "loan_amount", "loan_term_months",
    "n_payments", "max_dpd_months", "default", "label_status",
    "first_payment_date", "last_payment_date", "as_of_date",
]


def derive_default_labels_from_frames(
    payments: pl.DataFrame,
    masters: pl.DataFrame,
    as_of,
    default_trigger_months: int = 3,
    observation_months: int = 12,
) -> pl.DataFrame:
    """Join payments -> loans -> applications and compute the windowed flag."""
    if payments.height == 0 or masters.height == 0:
        return pl.DataFrame(schema={c: pl.Utf8 for c in _GOLD_COLUMNS}).head(0)

    W = observation_months
    in_win = pl.col("num_payment") <= W
    agg = (
        payments.filter(pl.col("application_id").is_not_null())
        .group_by("application_id")
        .agg([
            pl.col("loan_id").first(),
            pl.col("client_id").first(),
            pl.col("payment_date").min().alias("first_payment_date"),
            pl.col("payment_date").max().alias("last_payment_date"),
            pl.len().alias("n_payments"),
            pl.col("num_payment").min().alias("_first_inst"),
            pl.col("num_payment").filter(in_win).max().alias("_last_inst_in_win"),
            (pl.col("days_late").filter(in_win).max() / 30.0).floor()
            .alias("max_dpd_months"),
        ])
    )
    m = (masters.select(["application_id", "submission_timestamp",
                         "loan_amount", "loan_term_months"])
         .unique(subset="application_id", keep="first"))
    gold = agg.join(m, on="application_id", how="left")

    need = pl.min_horizontal(pl.col("loan_term_months").fill_null(W), pl.lit(W))
    gold = gold.with_columns([
        (pl.col("max_dpd_months").fill_null(0) >= default_trigger_months)
        .cast(pl.Int32).alias("default"),
    ]).with_columns([
        pl.when(pl.col("_first_inst") > 1).then(pl.lit("left_truncated"))
        .when(pl.col("_last_inst_in_win").fill_null(0) >= need)
        .then(pl.lit("final"))
        .otherwise(pl.lit("pending")).alias("label_status"),
        pl.lit(as_of).alias("as_of_date"),
    ])
    return gold.select(_GOLD_COLUMNS)


def derive_default_labels(
    batches,
    default_trigger_months: int = 3,
    observation_months: int = 12,
) -> pl.DataFrame:
    """Convenience wrapper: derive labels from a list of MonthBatch."""
    payments = pl.concat([
        b.payments for b in batches
        if b.payments is not None and b.payments.height > 0])
    masters = pl.concat([
        b.application_master for b in batches
        if b.application_master is not None and b.application_master.height > 0])
    as_of = max([b.month for b in batches])
    return derive_default_labels_from_frames(
        payments, masters, as_of, default_trigger_months, observation_months)