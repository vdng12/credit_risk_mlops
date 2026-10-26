"""Calibrate structural dependencies between tables.

For each child table this fits

1. marginals          -- unconditional frequencies / lognormal params
2. conditional links  -- child outcome ~ client factor (binned decile curve
                         or GLM), with factors on the SAME N(0,1) normal-score
                         scale the simulator samples on
3. Markov transitions -- bureau_balance STATUS, conditional on the client's
                         default_propensity bucket

plus the **default curve** P(default | f6) that drives the label: the
quantile function of the out-of-fold PD score, so the simulator reproduces
the real data's discrimination instead of inheriting it from a noisy
per-month proxy.

Output: simulator/calibrated_params.json (consumed via params.Params).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, LogisticRegression, PoissonRegressor

from simulator.calibrate import FACTOR_ORDER, compute_factor_scores, normal_scores

FACTOR_IDX = {name: i for i, name in enumerate(FACTOR_ORDER)}
N_BUCKETS = 5            # f6 buckets for the conditional bureau Markov chain
DIRICHLET_KAPPA = 50.0   # pseudo-counts shrinking bucket rows to the pooled row


# ---------------------------------------------------------------------------
# Factor loading
# ---------------------------------------------------------------------------

def load_factors(data_dir: str, subsample: int | None = None, seed: int = 42):
    """Return (normal-score factors, SK_ID_CURR, info, raw PD score)."""
    df = pl.read_csv(f"{data_dir}/application_train.csv", infer_schema_length=50_000)
    if subsample is not None and subsample < df.height:
        rng = np.random.default_rng(seed)
        df = df[np.sort(rng.choice(df.height, size=subsample, replace=False))]
    raw, info = compute_factor_scores(df)
    return normal_scores(raw), df["SK_ID_CURR"].to_numpy(), info, raw[:, -1]


def factor_frame(scores: np.ndarray, sk_ids: np.ndarray) -> pl.DataFrame:
    cols = {f"f{i}": scores[:, i] for i in range(scores.shape[1])}
    cols["SK_ID_CURR"] = sk_ids
    return pl.DataFrame(cols)


# ---------------------------------------------------------------------------
# Fitting helpers (outcome ~ factor, factor ~ N(0,1))
# ---------------------------------------------------------------------------

def _clean(X, y):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[ok], y[ok]


def fit_logistic(y, X, names):
    X, y = _clean(X, y)
    p = float(y.mean())
    if not 0 < p < 1:
        return {"kind": "logistic", "intercept": 0.0,
                "coef": {f: 0.0 for f in names}, "base_rate": p}
    m = LogisticRegression(max_iter=1000).fit(X, y)
    return {"kind": "logistic", "intercept": float(m.intercept_[0]),
            "coef": {f: float(m.coef_[0][j]) for j, f in enumerate(names)},
            "base_rate": p}


def fit_poisson(y, X, names):
    X, y = _clean(X, y)
    m = PoissonRegressor(max_iter=1000, alpha=0.0).fit(X, np.maximum(y, 0))
    return {"kind": "poisson", "intercept": float(m.intercept_),
            "coef": {f: float(m.coef_[j]) for j, f in enumerate(names)},
            "base_mean": float(y.mean())}


def fit_linear(y, X, names):
    X, y = _clean(X, y)
    m = LinearRegression().fit(X, y)
    return {"kind": "linear", "intercept": float(m.intercept_),
            "coef": {f: float(m.coef_[j]) for j, f in enumerate(names)},
            "resid_sigma": float((y - m.predict(X)).std())}


def fit_binned(y, f, factor_name, n_bins: int = 10) -> dict:
    """P(y=1 | factor decile). Factor is N(0,1), so edges are ~Phi^-1(k/10)."""
    f = np.asarray(f, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(f) & np.isfinite(y)
    f, y = f[ok], y[ok]
    base = float(y.mean()) if len(y) else 0.0
    if len(y) < 10 * n_bins or y.std() == 0:
        return {"kind": "binned", "factor": factor_name,
                "edges": [-np.inf, np.inf], "rates": [base]}
    edges = np.unique(np.quantile(f, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    bins = np.clip(np.searchsorted(edges, f, side="right") - 1, 0, len(edges) - 2)
    rates = [float(y[bins == b].mean()) if (bins == b).any() else base
             for b in range(len(edges) - 1)]
    return {"kind": "binned", "factor": factor_name,
            "edges": edges.tolist(), "rates": rates}


def _sub(df: pl.DataFrame, n: int, rng) -> pl.DataFrame:
    if df.height <= n:
        return df
    return df[np.sort(rng.choice(df.height, size=n, replace=False))]


def bucket_of(f: np.ndarray, k: int = N_BUCKETS) -> np.ndarray:
    """Equal-probability buckets of an N(0,1) factor."""
    edges = norm.ppf(np.linspace(0, 1, k + 1)[1:-1])
    return np.searchsorted(edges, f, side="right")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(data_dir: str, subsample_big: int = 300_000, seed: int = 42) -> dict:
    data = Path(data_dir)
    rng = np.random.default_rng(seed)
    out: dict = {}

    print("loading factor scores (normal-score scale)...")
    scores, sk_ids, finfo, pd_raw = load_factors(data_dir, subsample=100_000, seed=seed)
    factors = factor_frame(scores, sk_ids)
    out["factors"] = {"n": int(scores.shape[0]), "info": finfo}

    # ---- default curve: P(TARGET=1 | f6) = Q_PD(Phi(f6)) -----------------
    u_grid = np.linspace(0.0025, 0.9975, 200)
    out["default_curve"] = {
        "u_grid": u_grid.tolist(),
        "pd_quantiles": np.quantile(pd_raw, u_grid).tolist(),
        "base_rate": finfo["default_propensity"]["base_rate"],
        "oof_auc": finfo["default_propensity"]["oof_auc"],
        "source": "calibrated",
    }

    # ---- bureau ----------------------------------------------------------
    print("bureau...")
    bdf = (pl.scan_csv(str(data / "bureau.csv"))
           .select(["SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "AMT_CREDIT_SUM",
                    "AMT_CREDIT_SUM_OVERDUE", "CREDIT_TYPE"])
           .collect().join(factors, on="SK_ID_CURR", how="inner"))
    freq = lambda col: {r[col]: r["freq"] for r in bdf.group_by(col).agg(
        (pl.len() / bdf.height).alias("freq")).iter_rows(named=True)}
    amt = bdf["AMT_CREDIT_SUM"].to_numpy()
    pos_amt = np.isfinite(amt) & (amt > 0)
    out_b = {
        "credit_type_freq": freq("CREDIT_TYPE"),
        "credit_active_freq": freq("CREDIT_ACTIVE"),
        "models": {
            "log_amt_credit_sum": fit_linear(
                np.log(amt[pos_amt]), bdf["f1"].to_numpy()[pos_amt, None],
                ["income_capacity"]),
            "p_overdue": fit_binned(
                (bdf["AMT_CREDIT_SUM_OVERDUE"].fill_null(0).to_numpy() > 0).astype(float),
                bdf["f6"].to_numpy(), "default_propensity"),
            "p_active": fit_binned(
                (bdf["CREDIT_ACTIVE"].to_numpy() == "Active").astype(float),
                bdf["f4"].to_numpy(), "stability"),
        },
    }
    cnt = bdf.group_by("SK_ID_CURR").len().join(factors, on="SK_ID_CURR")
    out_b["models"]["n_accounts"] = fit_poisson(
        cnt["len"].to_numpy(), cnt["f4"].to_numpy()[:, None], ["stability"])
    out["bureau"] = out_b

    # ---- previous_application -------------------------------------------
    print("previous_application...")
    pdf = (pl.scan_csv(str(data / "previous_application.csv"))
           .select(["SK_ID_CURR", "NAME_CONTRACT_STATUS", "AMT_CREDIT", "CNT_PAYMENT"])
           .collect().join(factors, on="SK_ID_CURR", how="inner"))
    pdf = _sub(pdf, subsample_big, rng)
    approved = (pdf["NAME_CONTRACT_STATUS"].to_numpy() == "Approved").astype(float)
    # Only approved contracts carry meaningful amount/term: refused/canceled
    # rows have AMT_CREDIT == 0 and CNT_PAYMENT null, which previously made
    # log(amount) bimodal (resid_sigma ~4.8 => e^4.8 = 120x spread).
    apdf = pdf.filter((pl.col("NAME_CONTRACT_STATUS") == "Approved")
                      & (pl.col("AMT_CREDIT") > 0) & (pl.col("CNT_PAYMENT") > 0))
    X12 = np.column_stack([apdf["f1"].to_numpy(), apdf["f2"].to_numpy()])
    out["previous_application"] = {"models": {
        "p_approved": fit_binned(approved, pdf["f0"].to_numpy(), "creditworthiness"),
        "log_amt_credit": fit_linear(np.log(apdf["AMT_CREDIT"].to_numpy()), X12,
                                     ["income_capacity", "loan_demand"]),
        "term_months": fit_poisson(apdf["CNT_PAYMENT"].to_numpy(),
                                   apdf["f2"].to_numpy()[:, None], ["loan_demand"]),
    }}

    # ---- installments_payments (diagnostic + payment-amount links) -------
    print("installments_payments...")
    # One instalment is often paid in several rows; aggregate first, or
    # every split payment counts as a "partial".
    idf = (pl.scan_csv(str(data / "installments_payments.csv"))
           .group_by(["SK_ID_CURR", "SK_ID_PREV", "NUM_INSTALMENT_NUMBER"])
           .agg([pl.col("AMT_INSTALMENT").max(), pl.col("AMT_PAYMENT").sum(),
                 (pl.col("DAYS_ENTRY_PAYMENT") - pl.col("DAYS_INSTALMENT")).max()
                 .alias("DAYS_LATE")])
           .collect().join(factors, on="SK_ID_CURR", how="inner"))
    idf = _sub(idf, subsample_big, rng)
    inst = np.maximum(idf["AMT_INSTALMENT"].to_numpy(), 1.0)
    pay = idf["AMT_PAYMENT"].fill_null(0).to_numpy()
    f6 = idf["f6"].to_numpy()
    out["installments_payments"] = {"models": {
        "log_amt_instalment": fit_linear(np.log(inst), idf["f2"].to_numpy()[:, None],
                                         ["loan_demand"]),
        "p_partial": fit_binned((pay < 0.99 * inst).astype(float), f6,
                                "default_propensity"),
        "p_late": fit_binned((idf["DAYS_LATE"].fill_null(0).to_numpy() > 0)
                             .astype(float), f6, "default_propensity"),
    }}

    # ---- POS_CASH_balance ------------------------------------------------
    print("POS_CASH_balance...")
    posdf = _sub(pl.scan_csv(str(data / "POS_CASH_balance.csv"))
                 .select(["SK_ID_CURR", "SK_DPD"]).collect()
                 .join(factors, on="SK_ID_CURR", how="inner"), subsample_big, rng)
    dpd = posdf["SK_DPD"].fill_null(0).to_numpy()
    out["pos_cash_balance"] = {"models": {
        "p_dpd_gt0": fit_binned((dpd > 0).astype(float), posdf["f6"].to_numpy(),
                                "default_propensity")}}

    # ---- credit_card_balance --------------------------------------------
    print("credit_card_balance...")
    cdf = _sub(pl.scan_csv(str(data / "credit_card_balance.csv"))
               .select(["SK_ID_CURR", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "SK_DPD"])
               .collect().join(factors, on="SK_ID_CURR", how="inner"), subsample_big, rng)
    lim = cdf["AMT_CREDIT_LIMIT_ACTUAL"].to_numpy().astype(float)
    okl = np.isfinite(lim) & (lim > 0)
    util = cdf["AMT_BALANCE"].to_numpy() / np.where(okl, lim, np.nan)
    out["credit_card_balance"] = {"models": {
        "log_credit_limit": fit_linear(np.log(lim[okl]), cdf["f1"].to_numpy()[okl, None],
                                       ["income_capacity"]),
        "utilization": fit_linear(np.clip(util, 0, 1.5), cdf["f5"].to_numpy()[:, None],
                                  ["risk_appetite"]),
        "p_dpd_gt0": fit_binned((cdf["SK_DPD"].fill_null(0).to_numpy() > 0).astype(float),
                                cdf["f6"].to_numpy(), "default_propensity"),
    }}

    # ---- bureau_balance: STATUS Markov chain conditional on f6 bucket ----
    print("bureau_balance (conditional Markov)...")
    b2c = bdf.select(["SK_ID_BUREAU", "f6"])
    ids = b2c["SK_ID_BUREAU"].unique().to_numpy()
    ids = ids[rng.choice(len(ids), size=min(80_000, len(ids)), replace=False)]
    bb = (pl.scan_csv(str(data / "bureau_balance.csv"))
          .filter(pl.col("SK_ID_BUREAU").is_in(ids.tolist())).collect()
          .join(b2c, on="SK_ID_BUREAU", how="inner")
          .sort(["SK_ID_BUREAU", "MONTHS_BALANCE"])
          .with_columns(pl.col("STATUS").shift(-1).over("SK_ID_BUREAU").alias("NEXT")))
    states = sorted(bb["STATUS"].unique().to_list())
    sidx = {s: i for i, s in enumerate(states)}
    S = len(states)
    bb = bb.with_columns(pl.Series("bucket", bucket_of(bb["f6"].to_numpy())))
    tr = bb.filter(pl.col("NEXT").is_not_null())
    counts = np.zeros((N_BUCKETS, S, S))
    for k, a, b, c in tr.group_by(["bucket", "STATUS", "NEXT"]).len().iter_rows():
        counts[k, sidx[a], sidx[b]] += c
    pooled = counts.sum(0)
    pooled_p = pooled / np.maximum(pooled.sum(1, keepdims=True), 1)
    by_bucket = (counts + DIRICHLET_KAPPA * pooled_p[None]) / (
        counts.sum(2, keepdims=True) + DIRICHLET_KAPPA)
    # initial (most recent) status per account, per bucket
    last = bb.filter(pl.col("MONTHS_BALANCE") == pl.col("MONTHS_BALANCE")
                     .max().over("SK_ID_BUREAU"))
    init = np.zeros((N_BUCKETS, S))
    for k, s, c in last.group_by(["bucket", "STATUS"]).len().iter_rows():
        init[k, sidx[s]] += c
    init = (init + 1.0) / (init + 1.0).sum(1, keepdims=True)
    out["bureau_balance"] = {
        "status_states": states,
        "status_transition": pooled_p.tolist(),
        "status_transition_by_bucket": by_bucket.tolist(),
        "status_init_by_bucket": init.tolist(),
        "bucket_factor": "default_propensity",
        "n_buckets": N_BUCKETS,
    }
    print(f"  states={states}  transitions={tr.height:,}")
    return out


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else (
        "/home/mii/.cache/kagglehub/competitions/home-credit-default-risk")
    result = main(data_dir)
    out_path = Path(__file__).with_name("calibrated_params.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")
