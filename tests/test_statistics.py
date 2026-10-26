"""Statistical regression tests for the simulator (no Home Credit data needed).

Run:  python -m pytest tests/test_statistics.py -q
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import roc_auc_score

from simulator import SimConfig, Simulator
from simulator.calibrate import fit_t_copula, pseudo_obs
from simulator.config import CopulaConfig
from simulator.copula import CopulaEngine
from simulator.label import derive_default_labels_from_frames
from simulator.params import Params
from simulator.state import RunHazard, solve_default_shift

SIGMA = np.array(CopulaConfig().corr_matrix)


# ---------------------------------------------------------------- copula ----

@pytest.mark.parametrize("nu", [4.0, 10.0])
def test_t_copula_recovers_nu_and_sigma(nu):
    u = pseudo_obs(CopulaEngine(nu=nu, corr_matrix=SIGMA, seed=1).sample_uniform(10_000))
    fit = fit_t_copula(u)
    assert abs(fit["nu"] - nu) / nu < 0.25
    assert np.abs(np.array(fit["corr_matrix"]) - SIGMA).max() < 0.05
    assert not fit["gaussian_preferred"]


def test_gaussian_data_is_flagged():
    u = pseudo_obs(CopulaEngine(nu=1e6, corr_matrix=SIGMA, seed=2).sample_uniform(10_000))
    assert fit_t_copula(u)["gaussian_preferred"]


def test_latent_factors_have_standard_normal_margins():
    f = CopulaEngine(nu=5, corr_matrix=SIGMA, seed=3).sample_latent_factors(50_000)
    assert np.allclose(f.mean(0), 0, atol=0.03)
    assert np.allclose(f.std(0), 1, atol=0.03)


def test_lognormal_marginal_is_exact():
    eng = CopulaEngine(seed=4)
    f = np.random.default_rng(0).standard_normal(200_000)
    x = np.log(eng.lognormal(f, log_mu=12.0, log_sigma=0.6, factor_weight=0.7))
    assert abs(x.mean() - 12.0) < 0.01 and abs(x.std() - 0.6) < 0.01
    assert abs(np.corrcoef(x, f)[0, 1] - 0.7) < 0.01


# ---------------------------------------------------------- default model ----

def test_run_hazard_matches_monte_carlo():
    rh = RunHazard(window=12, trigger=3)
    rng = np.random.default_rng(5)
    for pd_target in (0.02, 0.08, 0.3):
        h = rh.hazard(np.array([pd_target]), np.array([12]))[0]
        draws = rng.random((200_000, 12)) < h
        run = np.zeros(len(draws), int)
        hit = np.zeros(len(draws), bool)
        for m in range(12):
            run = np.where(draws[:, m], run + 1, 0)
            hit |= run >= 3
        assert abs(hit.mean() - pd_target) < 0.003


def test_default_shift_hits_target():
    z = np.random.default_rng(6).standard_normal(100_000)
    pd = Params().pd_curve(z)
    for tgt in (0.03, 0.08, 0.2):
        d = solve_default_shift(pd, np.ones_like(z), tgt)
        lp = np.log(pd / (1 - pd)) + d
        assert abs((1 / (1 + np.exp(-lp))).mean() - tgt) < 1e-6


# ----------------------------------------------------------- end to end -----

def _run(target, n=25_000, months=30, seed=7):
    cfg = SimConfig(n_customers=n, start_date=date(2020, 1, 1), seed=seed,
                    target_default_rate=target)
    cfg.problems.enabled = False
    cfg.entities.generate_card_auths = False
    cfg.entities.generate_cards = False
    cfg.entities.generate_bureau = False
    sim = Simulator(cfg)
    M, P, last, clients = [], [], None, []
    for b in sim.play_forward(months):
        M.append(b.application_master)
        P.append(b.payments)
        clients.append(b.clients)
        last = b.month
    gold = derive_default_labels_from_frames(
        pl.concat([p for p in P if p.height]), pl.concat(M), last)
    return sim, gold, clients


@pytest.fixture(scope="module")
def run08():
    return _run(0.08)


def test_windowed_default_rate_matches_knob(run08):
    _, gold, _ = run08
    final = gold.filter(pl.col("label_status") == "final")
    assert abs(final["default"].mean() - 0.08) < 0.012


def test_latent_discrimination_matches_pd_curve(run08):
    sim, gold, _ = run08
    final = gold.filter(pl.col("label_status") == "final")
    idx = {c: i for i, c in enumerate(sim._client_ids)}
    f6 = sim._client_factors[[idx[c] for c in final["client_id"]], 6]
    assert abs(roc_auc_score(final["default"], f6) - 0.745) < 0.03


def test_ext_source_is_protective(run08):
    """Real HC: higher EXT_SOURCE => lower default (AUC < 0.5)."""
    sim, gold, clients = run08
    final = gold.filter(pl.col("label_status") == "final")
    j = final.join(clients[0], on="client_id")
    for c in ("ext_source_1", "ext_source_3"):
        assert roc_auc_score(j["default"], j[c]) < 0.45


def test_client_attributes_stable_across_months(run08):
    _, _, clients = run08
    a, b = clients[0].sort("client_id"), clients[12].sort("client_id")
    for col in ("gender", "children_count", "income_total", "birth_date"):
        assert (a[col] == b[col]).all(), col
    assert ((b["age_years"] - a["age_years"]).is_between(0, 1)).all()
