"""Calibrate child-table distributions from Home Credit data.

Fits marginal + conditional distributions for the six child tables so the
entity generators produce realistic values instead of hand-set priors:

  bureau, previous_application, POS_CASH_balance, credit_card_balance,
  installments_payments, bureau_balance

Uses polars lazy scan + aggregation to handle the large files without
loading them fully into memory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl


def lognormal_fit(s: pl.Series) -> dict:
    """Fit lognormal (mu, sigma) to a positive series (MLE on log data)."""
    vals = s.drop_nulls().filter(s > 0)
    if vals.len() == 0:
        return {"n": 0, "mu": 0.0, "sigma": 0.0}
    logs = vals.log()
    return {
        "n": int(vals.len()),
        "mu": float(logs.mean()),
        "sigma": float(logs.std()),
    }


def value_counts(path: str, col: str, normalize: bool = True) -> dict:
    """Value counts of a categorical column, optionally normalized."""
    q = (
        pl.scan_csv(path)
        .select(pl.col(col))
        .group_by(col)
        .len()
        .sort("len", descending=True)
        .collect()
    )
    total = q["len"].sum()
    return {row[col]: (row["len"] / total if normalize else row["len"])
            for row in q.iter_rows(named=True)}


def numeric_stats(path: str, col: str) -> dict:
    """Basic numeric stats + lognormal fit for a column."""
    q = (
        pl.scan_csv(path)
        .select(pl.col(col))
        .describe()
        .collect()
    )
    d = {row["statistic"]: row["value"] for row in q.iter_rows(named=True)}
    s = pl.scan_csv(path).select(pl.col(col)).collect()[col]
    return {
        "count": int(d.get("count", 0)),
        "mean": float(d.get("mean", 0)),
        "std": float(d.get("std", 0)),
        "min": float(d.get("min", 0)), 
        "max": float(d.get("max", 0)),
        "lognormal": lognormal_fit(s),
    }


def ratio_bands(path: str, num_col: str, den_col: str) -> dict:
    """Distribution of num/den split into behavior bands."""
    q = (
        pl.scan_csv(path)
        .select([
            (pl.col(num_col) / pl.col(den_col)).alias("ratio"),
        ])
        .filter(pl.col("ratio").is_not_nan())
        .collect()
    )
    r = q["ratio"].drop_nulls()
    n = r.len()
    return {
        "n": n,
        "zero": float((r == 0).sum() / n),
        "partial_low": float(((r > 0) & (r < 0.5)).sum() / n),
        "partial_high": float(((r >= 0.5) & (r < 1.0)).sum() / n),
        "full_or_more": float((r >= 1.0).sum() / n),
        "mean_positive": float(r.filter(r > 0).mean() or 0.0),
    }


def dpd_bands(path: str, dpd_col: str = "SK_DPD") -> dict:
    """Distribute SK_DPD into delinquency bands."""
    q = (
        pl.scan_csv(path)
        .select(pl.col(dpd_col))
        .collect()
    )
    s = q[dpd_col].drop_nulls()
    n = s.len()
    return {
        "n": n,
        "current_0": float((s == 0).sum() / n),
        "dpd_1_30": float(((s > 0) & (s <= 30)).sum() / n),
        "dpd_31_60": float(((s > 30) & (s <= 60)).sum() / n),
        "dpd_61_90": float(((s > 60) & (s <= 90)).sum() / n),
        "dpd_90_plus": float((s > 90).sum() / n),
    }


def days_late_bands(path: str) -> dict:
    """Payment timing: DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT."""
    q = (
        pl.scan_csv(path)
        .select([
            (pl.col("DAYS_ENTRY_PAYMENT") - pl.col("DAYS_INSTALMENT")).alias("late"),
        ])
        .collect()
    )
    s = q["late"].drop_nulls()
    n = s.len()
    return {
        "n": n,
        "early": float((s < 0).sum() / n),
        "on_time": float((s == 0).sum() / n),
        "late_1_15": float(((s > 0) & (s <= 15)).sum() / n),
        "late_16_30": float(((s > 15) & (s <= 30)).sum() / n),
        "late_30_plus": float((s > 30).sum() / n),
        "mean_late_days": float(s.filter(s > 0).mean() or 0.0),
    }


def main(data_dir: str) -> dict:
    data = Path(data_dir)
    out: dict = {}

    print("=" * 70)
    print("1/6 bureau.csv")
    b = data / "bureau.csv"
    n_b = pl.scan_csv(str(b)).select(pl.len()).collect().item()
    n_clients = pl.scan_csv(str(b)).select("SK_ID_CURR").unique().collect().height
    out["bureau"] = {
        "rows": n_b,
        "records_per_client": n_b / n_clients,
        "credit_active": value_counts(str(b), "CREDIT_ACTIVE"),
        "credit_type": value_counts(str(b), "CREDIT_TYPE"),
        "amt_credit_sum": numeric_stats(str(b), "AMT_CREDIT_SUM"),
        "amt_credit_sum_debt": numeric_stats(str(b), "AMT_CREDIT_SUM_DEBT"),
        "amt_credit_sum_overdue": numeric_stats(str(b), "AMT_CREDIT_SUM_OVERDUE"),
        "cnt_credit_prolong": value_counts(str(b), "CNT_CREDIT_PROLONG"),
        "days_credit": numeric_stats(str(b), "DAYS_CREDIT"),
    }
    print(f"  rows={n_b:,} clients={n_clients:,} records/client={n_b/n_clients:.2f}")

    print("2/6 previous_application.csv")
    p = data / "previous_application.csv"
    n_p = pl.scan_csv(str(p)).select(pl.len()).collect().item()
    n_pc = pl.scan_csv(str(p)).select("SK_ID_CURR").unique().collect().height
    out["previous_application"] = {
        "rows": n_p,
        "apps_per_client": n_p / n_pc,
        "contract_status": value_counts(str(p), "NAME_CONTRACT_STATUS"),
        "contract_type": value_counts(str(p), "NAME_CONTRACT_TYPE"),
        "amt_credit": numeric_stats(str(p), "AMT_CREDIT"),
        "amt_annuity": numeric_stats(str(p), "AMT_ANNUITY"),
        "cnt_payment": numeric_stats(str(p), "CNT_PAYMENT"),
        "rate_interest_primary": numeric_stats(str(p), "RATE_INTEREST_PRIMARY"),
    }
    print(f"  rows={n_p:,} apps/client={n_p/n_pc:.2f}")

    print("3/6 installments_payments.csv")
    i = data / "installments_payments.csv"
    out["installments_payments"] = {
        "amt_instalment": numeric_stats(str(i), "AMT_INSTALMENT"),
        "payment_ratio_bands": ratio_bands(str(i), "AMT_PAYMENT", "AMT_INSTALMENT"),
        "days_late_bands": days_late_bands(str(i)),
    }
    print(f"  payment ratio bands: {out['installments_payments']['payment_ratio_bands']}")

    print("4/6 POS_CASH_balance.csv")
    pos = data / "POS_CASH_balance.csv"
    out["pos_cash_balance"] = {
        "sk_dpd_bands": dpd_bands(str(pos), "SK_DPD"),
        "sk_dpd_def_rate": float(
            pl.scan_csv(str(pos)).select(pl.col("SK_DPD_DEF").mean()).collect().item()
        ),
        "cnt_instalment": numeric_stats(str(pos), "CNT_INSTALMENT"),
        "months_balance": numeric_stats(str(pos), "MONTHS_BALANCE"),
    }
    print(f"  dpd bands: {out['pos_cash_balance']['sk_dpd_bands']}")

    print("5/6 credit_card_balance.csv")
    c = data / "credit_card_balance.csv"
    out["credit_card_balance"] = {
        "amt_credit_limit": numeric_stats(str(c), "AMT_CREDIT_LIMIT_ACTUAL"),
        "utilization": ratio_bands(str(c), "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"),
        "sk_dpd_bands": dpd_bands(str(c), "SK_DPD"),
    }
    print(f"  utilization bands: {out['credit_card_balance']['utilization']}")

    print("6/6 bureau_balance.csv")
    bb = data / "bureau_balance.csv"
    out["bureau_balance"] = {
        "status": value_counts(str(bb), "STATUS"),
    }
    print(f"  status: {out['bureau_balance']['status']}")

    return out


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else (
        "/home/mii/.cache/kagglehub/competitions/home-credit-default-risk")
    result = main(data_dir)

    out_path = Path("simulator/calibrated_params.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")