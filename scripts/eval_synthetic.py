"""Evaluate LightGBM on synthetic data vs the real Home Credit baseline.

Runs the *exact same* feature-engineering + training pipeline (the one in
notebooks/fit.py) on two datasets:

  1. real    -- application_train.csv (the baseline)
  2. synthetic -- simulator output, mapped into the Home Credit schema and
                 labeled via the gold-layer default derivation.

Failure mode this guards against: a simulator whose default signal is
decorrelated from its features would produce AUC ~0.5 (unlearnable), so
comparing AUC tells us whether the synthetic data actually carries the
latent structure a model can exploit.

Usage:
    uv sync --extra eval
    uv run python scripts/eval_synthetic.py \
        --data /home/mii/.cache/kagglehub/competitions/home-credit-default-risk \
        --n-customers 8000 --months 24
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler, TargetEncoder
import lightgbm as lgb

# ---------------------------------------------------------------------------
# Home Credit feature set (identical to notebooks/fit.py)
# ---------------------------------------------------------------------------

FEATURES = [
    "CODE_GENDER", "DAYS_BIRTH", "CNT_CHILDREN", "CNT_FAM_MEMBERS",
    "NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE", "NAME_TYPE_SUITE",
    "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "NAME_CONTRACT_TYPE",
    "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "OWN_CAR_AGE", "NAME_HOUSING_TYPE",
    "AMT_INCOME_TOTAL", "NAME_INCOME_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE",
    "DAYS_EMPLOYED", "WEEKDAY_APPR_PROCESS_START", "HOUR_APPR_PROCESS_START",
    "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY", "REGION_POPULATION_RELATIVE",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
]

HIGH_CARD_COLS = ["ORGANIZATION_TYPE", "OCCUPATION_TYPE"]


def build_dataset(raw_df: pl.DataFrame, target: np.ndarray | None = None):
    """Same encode -> scale pipeline as notebooks/fit.py.build_dataset."""
    cat_cols = [c for c, t in zip(raw_df.columns, raw_df.dtypes)
                if c in FEATURES and t in (pl.Utf8, pl.Categorical, pl.Object)]
    num_cols = [c for c in FEATURES if c not in cat_cols]
    low_card = [c for c in cat_cols if c not in HIGH_CARD_COLS]

    cleaned = raw_df.select([c for c in FEATURES if c in raw_df.columns])
    if "DAYS_EMPLOYED" in cleaned.columns:
        cleaned = cleaned.with_columns(
            pl.when(pl.col("DAYS_EMPLOYED") == 365243)
            .then(None).otherwise(pl.col("DAYS_EMPLOYED")).alias("DAYS_EMPLOYED"))

    augmented = cleaned
    # engineered ratios (only if source cols exist)
    need = ["AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY", "DAYS_EMPLOYED", "DAYS_BIRTH"]
    if all(c in cleaned.columns for c in need):
        augmented = cleaned.with_columns([
            (pl.col("AMT_CREDIT") / pl.col("AMT_INCOME_TOTAL")).alias("CREDIT_INCOME_RATIO"),
            (pl.col("AMT_ANNUITY") / pl.col("AMT_INCOME_TOTAL")).alias("ANNUITY_INCOME_RATIO"),
            (pl.col("AMT_CREDIT") / pl.col("AMT_ANNUITY")).alias("CREDIT_TERM_MONTHS"),
            (pl.col("DAYS_EMPLOYED") / pl.col("DAYS_BIRTH")).clip(0, 1).alias("DAYS_EMPLOYED_RATIO"),
        ])
        num_cols = num_cols + ["CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO",
                               "CREDIT_TERM_MONTHS", "DAYS_EMPLOYED_RATIO"]

    X_high = augmented.select(HIGH_CARD_COLS).to_numpy()
    enc = TargetEncoder(smooth="auto")
    X_high = enc.fit_transform(X_high, target) if target is not None else enc.transform(X_high)

    dummies = augmented.select(low_card).to_dummies(columns=low_card)
    X_num = augmented.select(num_cols).to_numpy()
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X_num)

    return np.hstack([X_high, dummies.to_numpy(), X_num]).astype(np.float64)


def train_cv_auc(X: np.ndarray, y: np.ndarray, nfold: int = 5, seed: int = 42) -> float:
    """LightGBM 5-fold CV AUC (same params as fit.py)."""
    param = {"num_leaves": 31, "objective": "binary", "metric": "auc",
             "verbosity": -1}
    d = lgb.Dataset(X, label=y)
    cv = lgb.cv(param, d, num_boost_round=100, nfold=nfold, metrics="auc",
                seed=seed)
    return float(cv["valid auc-mean"][-1])


# ---------------------------------------------------------------------------
# Synthetic bridge: simulator output -> Home Credit schema + TARGET
# ---------------------------------------------------------------------------

def build_synthetic_df(
    masters: pl.DataFrame,
    clients: pl.DataFrame,
    payments: pl.DataFrame,
    as_of,
) -> pl.DataFrame:
    """Map simulator entities into the Home Credit feature columns + TARGET."""
    from simulator.label import derive_default_labels_from_frames

    gold = derive_default_labels_from_frames(payments, masters, as_of)

    df = (
        masters
        .join(clients, on="client_id", how="left")
        .join(gold.select(["application_id", "default", "label_status"]),
              on="application_id", how="left")
    )

    # NAME_TYPE_SUITE synthesized from household structure
    def suite():
        children = df["children_count"].to_numpy()
        fam = df["family_members"].to_numpy()
        status = df["family_status"].to_list()
        out = []
        for c, f, s in zip(children, fam, status):
            if c > 0:
                out.append("Children")
            elif f >= 2 and s in ("Married", "Civil marriage"):
                out.append("Family")
            else:
                out.append("Unaccompanied")
        return out

    subday = df["submission_timestamp"]
    weekday = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY",
               "FRIDAY", "SATURDAY", "SUNDAY"]

    out = pl.DataFrame({
        "CODE_GENDER": df["gender"].to_list(),
        "DAYS_BIRTH": (-df["age_years"] * 365).to_list(),
        "CNT_CHILDREN": df["children_count"].to_list(),
        "CNT_FAM_MEMBERS": df["family_members"].to_list(),
        "NAME_FAMILY_STATUS": df["family_status"].to_list(),
        "NAME_EDUCATION_TYPE": df["education_level"].to_list(),
        "NAME_TYPE_SUITE": suite(),
        "AMT_CREDIT": df["loan_amount"].to_list(),
        "AMT_ANNUITY": df["annuity"].to_list(),
        "AMT_GOODS_PRICE": df["goods_price"].to_list(),
        "NAME_CONTRACT_TYPE": df["contract_type"].to_list(),
        "FLAG_OWN_CAR": df["owns_car"].cast(pl.Int8).to_list(),
        "FLAG_OWN_REALTY": df["owns_realty"].cast(pl.Int8).to_list(),
        "OWN_CAR_AGE": df["car_age"].replace(-1, None).to_list(),
        "NAME_HOUSING_TYPE": df["housing_type"].to_list(),
        "AMT_INCOME_TOTAL": df["income_total"].to_list(),
        "NAME_INCOME_TYPE": df["income_type"].to_list(),
        "OCCUPATION_TYPE": df["occupation_type"].to_list(),
        "ORGANIZATION_TYPE": df["organization_type"].to_list(),
        "DAYS_EMPLOYED": df["days_employed"].to_list(),
        "WEEKDAY_APPR_PROCESS_START": [weekday[int(d.weekday())] for d in subday],
        "HOUR_APPR_PROCESS_START": [int(d.hour) for d in subday],
        "REGION_RATING_CLIENT": df["region_rating"].to_list(),
        "REGION_RATING_CLIENT_W_CITY": df["region_rating"].to_list(),
        "REGION_POPULATION_RELATIVE": df["region_population_relative"].to_list(),
        # EXT_SOURCE: passed through from the simulator's own columns
        # (each is an independent sigmoid of a different factor combination)
        "EXT_SOURCE_1": df["ext_source_1"].to_list(),
        "EXT_SOURCE_2": df["ext_source_2"].to_list(),
        "EXT_SOURCE_3": df["ext_source_3"].to_list(),
        # null for declined / never-paid / immature loans -- NOT 0
        "TARGET": df["default"].cast(pl.Int32).to_list(),
        "label_status": df["label_status"].to_list(),
    })
    return out


def select_training_rows(df: pl.DataFrame) -> pl.DataFrame:
    """Keep only rows a real training job may use: loans whose windowed
    label is final. Declined applications have no label (no loan), and
    pending / left-truncated labels are censored; filling them with 0 (the
    old behaviour) biased the default rate down (~4.7% vs 6.9%) and mixed
    the approval policy into the target."""
    return df.filter(pl.col("label_status") == "final")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--n-customers", type=int, default=8000)
    ap.add_argument("--months", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # ---- real baseline ----
    real = pl.read_csv(f"{args.data}/application_train.csv")
    real = real.select(FEATURES + ["TARGET"])
    y_real = real["TARGET"].to_numpy()
    X_real = build_dataset(real, target=y_real)
    print(f"real: {X_real.shape}")
    auc_real = train_cv_auc(X_real, y_real)
    print(f"REAL BASELINE       5-fold CV AUC = {auc_real:.4f}")

    # ---- synthetic - (skip the entity streams irrelevant to application-
    #      level training so a large run stays fast and memory-bounded) ----
    from simulator import SimConfig, Simulator
    cfg = SimConfig(n_customers=args.n_customers,
                    start_date=date(2020, 1, 1), seed=args.seed)
    cfg.problems.enabled = False
    cfg.entities.generate_card_auths = False
    cfg.entities.generate_cards = False
    cfg.entities.generate_bureau = False
    sim = Simulator(cfg)

    masters_parts, payments_parts = [], []
    first_clients, last_month = None, None
    for batch in sim.play_forward(args.months):
        if first_clients is None:
            first_clients = batch.clients
        if batch.application_master is not None and batch.application_master.height > 0:
            masters_parts.append(batch.application_master)
        if batch.payments is not None and batch.payments.height > 0:
            payments_parts.append(batch.payments)
        last_month = batch.month

    syn_apps = pl.concat(masters_parts)
    syn_pay = pl.concat(payments_parts)
    synth = build_synthetic_df(syn_apps, first_clients, syn_pay, last_month)
    synth = select_training_rows(synth).select(FEATURES + ["TARGET"])
    y_synth = synth["TARGET"].to_numpy()
    print(f"\nsynthetic: {synth.height} applications, "
          f"default rate={y_synth.mean():.3f}")
    X_synth = build_dataset(synth, target=y_synth)
    print(f"synthetic: {X_synth.shape}")
    auc_synth = train_cv_auc(X_synth, y_synth)
    print(f"SYNTHETIC           5-fold CV AUC = {auc_synth:.4f}")

    print("\n" + "=" * 46)
    print(f"real AUC      : {auc_real:.4f}")
    print(f"synthetic AUC : {auc_synth:.4f}")
    print(f"gap           : {auc_real - auc_synth:+.4f}")
    print("verdict: unlearnable (AUC~0.5) would mean the simulator "
          "dropped the default signal; an AUC well above 0.5 means the "
          "latent structure is preserved.")


if __name__ == "__main__":
    main()