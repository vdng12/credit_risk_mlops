"""t-Copula calibration from Home Credit data.

Fits nu (degrees of freedom) and the 7x7 correlation matrix from the real
application_train.csv feature groups:

  0: creditworthiness   — EXT_SOURCE_1/2/3                      (+ = safer)
  1: income_capacity    — log income, DAYS_EMPLOYED, employment  (+ = richer)
  2: loan_demand        — log AMT_CREDIT / ANNUITY / GOODS_PRICE (+ = bigger)
  3: asset_quality      — car/realty flags, car age, housing     (+ = more assets)
  4: stability          — age, children, family, gender, edu     (+ = older)
  5: risk_appetite      — contract type, suite                   (+ = revolving)
  6: default_propensity — out-of-fold GBM PD score               (+ = riskier)

Method (semi-parametric / canonical ML):
  1. One score per group: first principal component, then **sign-anchored**
     to a reference column so "+" always means the documented direction.
     (PCA sign is arbitrary; without anchoring the factor semantics the
     entity generators rely on can silently flip.)
  2. Pseudo-observations from mid-ranks: u = rank / (n + 1).
  3. Sigma from Kendall's tau inversion, rho = sin(pi * tau / 2) — valid for
     every elliptical copula (incl. t) and robust to heavy tails — then
     repaired to the nearest correlation matrix.
  4. nu by profile likelihood with Sigma fixed. If nu runs into the upper
     bound, or a likelihood-ratio test against the Gaussian copula is not
     significant, the result is flagged ``gaussian_preferred``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from scipy.stats import chi2, kendalltau, norm, rankdata
from scipy.stats import t as tdist


# ---------------------------------------------------------------------------
# Feature group definitions: {factor_name: [columns]}
# ---------------------------------------------------------------------------

FACTOR_FEATURES = {
    # REGION_RATING_* were removed from this group: the two region columns
    # are ~0.95 correlated with each other and hijacked PC1, so the
    # "creditworthiness" factor was really "region rating" (with an
    # arbitrary sign). They still feed the PD model below.
    "creditworthiness": ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"],
    "income_capacity": [
        "AMT_INCOME_TOTAL", "DAYS_EMPLOYED", "NAME_INCOME_TYPE", "OCCUPATION_TYPE",
    ],
    "loan_demand": ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"],
    "asset_quality": [
        "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "OWN_CAR_AGE", "NAME_HOUSING_TYPE",
    ],
    "stability": [
        "DAYS_BIRTH", "CNT_CHILDREN", "CNT_FAM_MEMBERS",
        "CODE_GENDER", "NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE",
    ],
    "risk_appetite": ["NAME_CONTRACT_TYPE", "NAME_TYPE_SUITE"],
    "default_propensity": ["TARGET"],
}

FACTOR_ORDER = list(FACTOR_FEATURES.keys())

# Extra columns used only by the PD model.
PD_EXTRA_FEATURES = ["REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY",
                     "REGION_POPULATION_RELATIVE"]

# Heavy-tailed amounts: log1p before standardising so PCA is not driven by a
# handful of extreme incomes / credits.
LOG_COLS = {"AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"}

# Sign anchors: factor -> (expr producing a numeric reference, sign).
# After PCA the score is flipped if corr(score, anchor) has the wrong sign.
ANCHORS = {
    "creditworthiness": (pl.col("EXT_SOURCE_2"), +1),
    "income_capacity": (pl.col("AMT_INCOME_TOTAL").log1p(), +1),
    "loan_demand": (pl.col("AMT_CREDIT").log1p(), +1),
    "asset_quality": (pl.col("FLAG_OWN_REALTY").cast(pl.Float64), +1),
    "stability": (-pl.col("DAYS_BIRTH").cast(pl.Float64), +1),          # older
    "risk_appetite": ((pl.col("NAME_CONTRACT_TYPE") == "Revolving loans")
                      .cast(pl.Float64), +1),
}


# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------

def preprocess(df: pl.DataFrame) -> pl.DataFrame:
    """Sentinel fix + Y/N flags -> 0/1. Shared by every calibration entry."""
    out = df
    if "DAYS_EMPLOYED" in out.columns:
        out = out.with_columns(
            pl.when(pl.col("DAYS_EMPLOYED") == 365243).then(None)
            .otherwise(pl.col("DAYS_EMPLOYED")).alias("DAYS_EMPLOYED"))
    for c in ("FLAG_OWN_CAR", "FLAG_OWN_REALTY"):
        if c in out.columns and out[c].dtype == pl.Utf8:
            out = out.with_columns((pl.col(c) == "Y").cast(pl.Int64).alias(c))
    return out


def load_and_preprocess(data_dir: str) -> pl.DataFrame:
    df = preprocess(pl.read_csv(f"{data_dir}/application_train.csv",
                                infer_schema_length=50_000))
    cols = ["SK_ID_CURR"]
    for g in FACTOR_FEATURES.values():
        cols += [c for c in g if c not in cols]
    cols += [c for c in PD_EXTRA_FEATURES if c not in cols]
    df = df.select([c for c in cols if c in df.columns])
    print(f"Loaded: {df.height} rows, {df.width} columns")
    return df


# ---------------------------------------------------------------------------
# Factor score computation
# ---------------------------------------------------------------------------

def _flatten_features(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    """Impute + log + standardise numeric, one-hot categorical."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    sub = df.select([c for c in cols if c in df.columns])
    num_cols = [c for c in sub.columns if sub[c].dtype.is_numeric()]
    cat_cols = [c for c in sub.columns if c not in num_cols]
    parts = []
    if num_cols:
        num = sub.select([
            (pl.col(c).cast(pl.Float64).clip(lower_bound=0).log1p()
             if c in LOG_COLS else pl.col(c).cast(pl.Float64))
            for c in num_cols]).to_numpy()
        parts.append(SimpleImputer(strategy="median").fit_transform(num))
    if cat_cols:
        parts.append(sub.select(cat_cols).fill_null("NA")
                     .to_dummies(columns=cat_cols).to_numpy().astype(np.float64))
    if not parts:
        return np.zeros((df.height, 0))
    return StandardScaler().fit_transform(np.hstack(parts))


def _oof_pd(X: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[np.ndarray, float]:
    """Out-of-fold PD score. In-sample predict_proba overstates the PD's
    discrimination, which then propagates into the simulator's label."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pd_score = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    return pd_score, float(roc_auc_score(y, pd_score))


def compute_factor_scores(df: pl.DataFrame) -> tuple[np.ndarray, dict]:
    """Return (n, 7) raw factor scores (sign-anchored) and diagnostics.

    Scores are NOT yet on a common scale; use :func:`normal_scores` to map
    them to N(0,1) margins (the scale the simulator samples on).
    """
    from sklearn.decomposition import PCA

    df = preprocess(df)
    info: dict = {}
    cols: list[np.ndarray] = []
    for name in FACTOR_ORDER[:-1]:
        avail = [c for c in FACTOR_FEATURES[name] if c in df.columns]
        if not avail:
            raise ValueError(f"No columns available for factor '{name}'")
        mat = _flatten_features(df, avail)
        if mat.shape[1] == 1:
            s, ev = mat[:, 0], 1.0
        else:
            pca = PCA(n_components=1, random_state=0)
            s = pca.fit_transform(mat)[:, 0]
            ev = float(pca.explained_variance_ratio_[0])
        flipped = False
        if name in ANCHORS:
            expr, sign = ANCHORS[name]
            a = df.select(expr.alias("_a"))["_a"].to_numpy().astype(np.float64)
            ok = np.isfinite(a)
            if ok.sum() > 10 and np.corrcoef(s[ok], a[ok])[0, 1] * sign < 0:
                s, flipped = -s, True
        info[name] = {"n_cols": len(avail), "explained_var": ev,
                      "sign_flipped_by_anchor": flipped}
        cols.append(s)

    pd_cols = list(dict.fromkeys(
        [c for g in FACTOR_ORDER[:-1] for c in FACTOR_FEATURES[g]]
        + PD_EXTRA_FEATURES))
    X = _flatten_features(df, pd_cols)
    y = df["TARGET"].to_numpy().astype(np.float64)
    pd_score, auc = _oof_pd(X, y)
    info["default_propensity"] = {
        "n_cols": X.shape[1], "explained_var": 1.0, "model": "hgb_pd_oof",
        "oof_auc": auc, "base_rate": float(y.mean()),
    }
    cols.append(pd_score)
    return np.column_stack(cols), info


def pseudo_obs(scores: np.ndarray) -> np.ndarray:
    """Mid-rank pseudo-observations u = rank / (n+1) (ties -> average rank)."""
    n = scores.shape[0]
    return rankdata(scores, method="average", axis=0) / (n + 1.0)


def normal_scores(scores: np.ndarray) -> np.ndarray:
    """Map each column to N(0,1) margins via its ranks: z = Phi^-1(u).

    This is the scale CopulaEngine.sample_latent_factors produces, so every
    model fitted against factors must use it (a z-scored PD probability is
    right-skewed, not N(0,1), and its decile edges do not line up with the
    simulator's draws).
    """
    return norm.ppf(pseudo_obs(scores))


# ---------------------------------------------------------------------------
# t-Copula estimation
# ---------------------------------------------------------------------------

def nearest_correlation(mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    m = (mat + mat.T) / 2
    w, v = np.linalg.eigh(m)
    m = v @ np.diag(np.maximum(w, eps)) @ v.T
    d = np.sqrt(np.diag(m))
    m = m / np.outer(d, d)
    np.fill_diagonal(m, 1.0)
    return m


def tau_inversion_corr(u: np.ndarray, max_n: int = 50_000,
                       seed: int = 0) -> np.ndarray:
    """rho_ij = sin(pi/2 * tau_b(i,j)); tau_b handles ties."""
    if u.shape[0] > max_n:
        idx = np.random.default_rng(seed).choice(u.shape[0], max_n, replace=False)
        u = u[idx]
    d = u.shape[1]
    r = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            tau = kendalltau(u[:, i], u[:, j], variant="b").statistic
            r[i, j] = r[j, i] = np.sin(np.pi * tau / 2)
    return nearest_correlation(r)


def _mvt_logpdf(x: np.ndarray, nu: float, sigma: np.ndarray) -> np.ndarray:
    d = x.shape[1]
    L = np.linalg.cholesky(sigma)
    z = np.linalg.solve(L, x.T).T
    mahal = (z * z).sum(axis=1)
    logdet = 2 * np.log(np.diag(L)).sum()
    const = (gammaln((nu + d) / 2) - gammaln(nu / 2)
             - d / 2 * np.log(nu * np.pi) - 0.5 * logdet)
    return const - (nu + d) / 2 * np.log1p(mahal / nu)


def t_copula_loglik(nu: float, u: np.ndarray, sigma: np.ndarray) -> float:
    """Total log c(u; nu, Sigma) = sum[log f_t,Sigma(t) - sum_j log f_t(t_j)]."""
    t = tdist.ppf(u, df=nu)
    return float(_mvt_logpdf(t, nu, sigma).sum()
                 - tdist.logpdf(t, df=nu).sum())


def gaussian_copula_loglik(u: np.ndarray, sigma: np.ndarray) -> float:
    z = norm.ppf(u)
    L = np.linalg.cholesky(sigma)
    w = np.linalg.solve(L, z.T).T
    logdet = 2 * np.log(np.diag(L)).sum()
    return float((-0.5 * logdet - 0.5 * ((w * w).sum(1) - (z * z).sum(1))).sum())


def fit_t_copula(u: np.ndarray, nu_bounds: tuple[float, float] = (2.05, 200.0),
                 alpha: float = 0.05) -> dict:
    sigma = tau_inversion_corr(u)
    res = minimize_scalar(lambda v: -t_copula_loglik(v, u, sigma),
                          bounds=nu_bounds, method="bounded",
                          options={"xatol": 0.05})
    nu, ll_t = float(res.x), float(-res.fun)
    ll_g = gaussian_copula_loglik(u, sigma)
    lr = 2 * (ll_t - ll_g)
    # nu = inf is on the boundary -> LR ~ 0.5*chi2_0 + 0.5*chi2_1
    p_value = float(0.5 * chi2.sf(max(lr, 0.0), df=1))
    at_bound = nu > nu_bounds[1] - 1.0
    return {
        "nu": nu, "corr_matrix": sigma.tolist(),
        "loglik_t": ll_t, "loglik_gaussian": ll_g,
        "lr_stat": lr, "lr_p_value": p_value,
        "nu_at_upper_bound": bool(at_bound),
        "gaussian_preferred": bool(at_bound or p_value > alpha),
    }


# ---------------------------------------------------------------------------
# Main calibration routine
# ---------------------------------------------------------------------------

def calibrate(data_dir: str, subsample: int | None = 50_000, seed: int = 42,
              nu_bounds: tuple[float, float] = (2.05, 200.0)) -> dict:
    rng = np.random.default_rng(seed)
    df = load_and_preprocess(data_dir)
    if subsample is not None and subsample < df.height:
        df = df[np.sort(rng.choice(df.height, size=subsample, replace=False))]
        print(f"Subsampled to {subsample} rows")

    scores, factor_info = compute_factor_scores(df)
    for name, inf in factor_info.items():
        print(f"  {name}: {inf}")
    fit = fit_t_copula(pseudo_obs(scores), nu_bounds=nu_bounds)
    fit["factor_info"] = factor_info
    print(f"\nnu = {fit['nu']:.2f}  (at bound: {fit['nu_at_upper_bound']}, "
          f"LR vs Gaussian = {fit['lr_stat']:.1f}, p = {fit['lr_p_value']:.3g})")
    print(np.array2string(np.array(fit["corr_matrix"]), precision=3))
    return fit


def apply_to_config(config, calibration: dict) -> None:
    config.copula.nu = (1e6 if calibration.get("gaussian_preferred")
                        else calibration["nu"])
    config.copula.corr_matrix = calibration["corr_matrix"]


if __name__ == "__main__":
    import json
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else (
        "/home/mii/.cache/kagglehub/competitions/home-credit-default-risk")
    result = calibrate(path)
    with open("simulator/calibrated_copula.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nWrote simulator/calibrated_copula.json")
