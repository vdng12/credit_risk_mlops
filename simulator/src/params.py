"""Load calibrated structural parameters.

Reads simulator/calibrated_params.json (produced by calibrate_structures.py).
Falls back to embedded defaults when the JSON is absent.

Fallback provenance (``_FALLBACK_SOURCE``):
  * decile rates for binned links come from the earlier Home Credit fit
    logs. Decile *membership* is rank-based, so the rates are valid on the
    N(0,1) normal-score scale with edges Phi^-1(k/10).
  * ``p_approved`` / ``p_active`` rates are reversed because the
    creditworthiness / stability factors were previously fitted with the
    wrong sign (see config.CopulaConfig).
  * amount models are priors: the earlier fits logged zero amounts
    (refused applications) as log(1) = 0, which produced resid_sigma ~4.8.
  * ``default_curve`` is a logit-normal PD curve matched to HC's base rate
    (8.07%) and a typical out-of-fold AUC (0.745) for application features.
  * the conditional bureau chain tilts the pooled fitted matrix.
Re-run calibrate.py + calibrate_structures.py to replace all of it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import norm

_JSON_PATH = Path(__file__).with_name("calibrated_params.json")

_DECILE_EDGES = [-np.inf] + norm.ppf(np.arange(1, 10) / 10).tolist() + [np.inf]


def _binned(factor: str, rates: list[float]) -> dict:
    return {"kind": "binned", "factor": factor, "edges": _DECILE_EDGES,
            "rates": rates}


def _interp_rates(lo: float, hi: float) -> list[float]:
    """Geometric interpolation when only the end deciles were logged."""
    return np.geomspace(lo, hi, 10).tolist()


_STATES = ["0", "1", "2", "3", "4", "5", "C", "X"]
_POOLED_T = np.array([
    [0.86, 0.01, 0.01, 0.01, 0.00, 0.00, 0.05, 0.07],
    [0.46, 0.36, 0.08, 0.02, 0.00, 0.00, 0.08, 0.00],
    [0.24, 0.08, 0.21, 0.25, 0.08, 0.02, 0.08, 0.04],
    [0.13, 0.05, 0.06, 0.10, 0.47, 0.10, 0.05, 0.04],
    [0.03, 0.07, 0.03, 0.03, 0.07, 0.71, 0.03, 0.03],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.92, 0.03, 0.02],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00],
    [0.13, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.86],
])
_POOLED_INIT = np.array([0.14, 0.05, 0.03, 0.02, 0.01, 0.02, 0.63, 0.10])


def _tilted_chain(n_buckets: int = 5, gamma: float = 0.75):
    """Bucket-specific chains: worsening moves scaled by exp(gamma * z_k)."""
    z = norm.ppf((np.arange(n_buckets) + 0.5) / n_buckets)
    dpd = {s: i for i, s in enumerate(_STATES) if s.isdigit()}
    Ts, inits = [], []
    for zk in z:
        T = _POOLED_T.copy()
        for a, i in [(s, _STATES.index(s)) for s in _STATES]:
            for b, j in dpd.items():
                worse = (a in dpd and j > dpd[a]) or (a == "X" and j > 0)
                if worse:
                    T[i, j] *= np.exp(gamma * zk)
        Ts.append((T / T.sum(1, keepdims=True)).tolist())
        init = _POOLED_INIT.copy()
        init[1:6] *= np.exp(gamma * zk)
        inits.append((init / init.sum()).tolist())
    return Ts, inits


_T_BY_BUCKET, _INIT_BY_BUCKET = _tilted_chain()

_FALLBACK_SOURCE = "fallback (see params.py docstring)"

_FALLBACK = {
    "_source": _FALLBACK_SOURCE,
    "default_curve": {
        "kind": "logit_normal", "mu": -2.7998, "s": 0.9687,
        "base_rate": 0.0807, "oof_auc": 0.745, "source": "prior",
    },
    "bureau": {
        "credit_type_freq": {
            "Consumer credit": 0.730, "Credit card": 0.235, "Car loan": 0.016,
            "Mortgage": 0.011, "Microloan": 0.006, "Other": 0.002},
        "credit_active_freq": {"Closed": 0.626, "Active": 0.370, "Sold": 0.004},
        "models": {
            "log_amt_credit_sum": {"kind": "linear", "intercept": 11.9,
                                   "coef": {"income_capacity": 0.20},
                                   "resid_sigma": 1.4},
            "p_overdue": _binned("default_propensity",
                                 [0.0008, 0.0007, 0.0008, 0.0014, 0.0014,
                                  0.0017, 0.0021, 0.0031, 0.0048, 0.0091]),
            "p_active": _binned("stability",
                                [0.3724, 0.3795, 0.3855, 0.3785, 0.3822,
                                 0.3666, 0.3710, 0.3626, 0.3689, 0.3373]),
            "n_accounts": {"kind": "poisson", "intercept": 1.719,
                           "coef": {"stability": 0.002}},
        },
    },
    "previous_application": {
        "models": {
            # reversed: see docstring (creditworthiness sign fix)
            "p_approved": _binned("creditworthiness",
                                  [0.6231, 0.6208, 0.6134, 0.6084, 0.6132,
                                   0.6119, 0.6259, 0.6494, 0.6671, 0.6159]),
            "log_amt_credit": {"kind": "linear", "intercept": 11.5,
                               "coef": {"income_capacity": 0.15,
                                        "loan_demand": 0.45},
                               "resid_sigma": 0.85},
            "term_months": {"kind": "poisson", "intercept": 2.784,
                            "coef": {"loan_demand": 0.15}},
        },
    },
    "installments_payments": {
        "models": {
            "log_amt_instalment": {"kind": "linear", "intercept": 8.777,
                                   "coef": {"loan_demand": 0.4},
                                   "resid_sigma": 0.9},
            "p_partial": _binned("default_propensity",
                                 [0.0791, 0.0834, 0.0876, 0.0907, 0.0958,
                                  0.1012, 0.1013, 0.1114, 0.1126, 0.1304]),
        },
    },
    "pos_cash_balance": {
        "models": {"p_dpd_gt0": _binned("default_propensity",
                                        _interp_rates(0.0253, 0.0383))},
    },
    "credit_card_balance": {
        "models": {
            "log_credit_limit": {"kind": "linear", "intercept": 11.4,
                                 "coef": {"income_capacity": 0.3},
                                 "resid_sigma": 0.9},
            "utilization": {"kind": "linear", "intercept": 0.307,
                            "coef": {"risk_appetite": 0.05},
                            "resid_sigma": 0.35},
            "p_dpd_gt0": _binned("default_propensity",
                                 _interp_rates(0.0341, 0.0468)),
        },
    },
    "bureau_balance": {
        "status_states": _STATES,
        "status_transition": _POOLED_T.tolist(),
        "status_transition_by_bucket": _T_BY_BUCKET,
        "status_init_by_bucket": _INIT_BY_BUCKET,
        "bucket_factor": "default_propensity",
        "n_buckets": 5,
    },
}


_REQUIRED_KEYS = ("default_curve",)
_REQUIRED_BB = ("status_transition_by_bucket", "status_init_by_bucket")


def load() -> dict:
    if _JSON_PATH.exists():
        with open(_JSON_PATH) as f:
            data = json.load(f)
        stale = (any(k not in data for k in _REQUIRED_KEYS)
                 or any(k not in data.get("bureau_balance", {}) for k in _REQUIRED_BB))
        if not stale:
            return data
        import warnings
        warnings.warn(
            f"{_JSON_PATH.name} predates the normal-score / PD-curve schema "
            "(its links were fit on a skewed factor scale). Using the "
            "embedded fallback; re-run calibrate_structures.py.",
            stacklevel=2)
    return _FALLBACK


class Params:
    """Dot-accessible wrapper over calibrated params."""

    def __init__(self, data: dict | None = None) -> None:
        self._data = data or load()

    @property
    def source(self) -> str:
        return self._data.get("_source", "calibrated")

    def model(self, table: str, name: str) -> dict:
        return self._data[table]["models"][name]

    def markov(self, table: str) -> tuple[list[str], list[list[float]]]:
        d = self._data[table]
        return d["status_states"], d["status_transition"]

    def conditional_markov(self, table: str):
        d = self._data[table]
        return (d["status_states"], np.asarray(d["status_transition_by_bucket"]),
                np.asarray(d["status_init_by_bucket"]), d["bucket_factor"])

    def pd_curve(self, z: np.ndarray) -> np.ndarray:
        """P(default within the performance window | f6 = z)."""
        dc = self._data["default_curve"]
        z = np.asarray(z, dtype=np.float64)
        if dc.get("kind") == "logit_normal":
            return expit(dc["mu"] + dc["s"] * z)
        u = norm.cdf(z)
        return np.interp(u, dc["u_grid"], dc["pd_quantiles"])

    @property
    def factor_idx(self) -> dict[str, int]:
        return {"creditworthiness": 0, "income_capacity": 1, "loan_demand": 2,
                "asset_quality": 3, "stability": 4, "risk_appetite": 5,
                "default_propensity": 6}
