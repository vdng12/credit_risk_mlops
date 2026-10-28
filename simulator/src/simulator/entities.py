"""Entity generators: produce synthetic data for all 8 entity types.

Each function takes a CopulaEngine, configuration params, and returns a
polars DataFrame with synthetic data for one time slice.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import polars as pl

from simulator.copula import CopulaEngine
from simulator.state import draw_from_model
from simulator.constants import (
    LATENT_CREDITWORTHINESS,
    LATENT_INCOME_CAPACITY,
    LATENT_LOAN_DEMAND,
    LATENT_ASSET_QUALITY,
    LATENT_STABILITY,
    LATENT_RISK_APPETITE,
    INCOME_LOG_MU,
    INCOME_LOG_SIGMA,
    CREDIT_AMOUNT_LOG_MU,
    CREDIT_AMOUNT_LOG_SIGMA,
    CARD_AUTH_AMOUNT_LOG_MU,
    CARD_AUTH_AMOUNT_LOG_SIGMA,
    CARD_LIMIT_LOG_MU,
    CARD_LIMIT_LOG_SIGMA,
)


def _rng_or_new(rng: Optional[np.random.Generator] = None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def generate_clients(
    engine: CopulaEngine,
    n: int,
    snapshot_date: date,
    client_ids: Optional[list[str]] = None,
    card_penetration: float = 0.55,
    loan_penetration: float = 0.18,
    rng: Optional[np.random.Generator] = None,
    factors: Optional[np.ndarray] = None,
    is_cardholder: Optional[np.ndarray] = None,
    is_loanholder: Optional[np.ndarray] = None,
) -> pl.DataFrame:
    """Generate n client master rows (monthly batch).

    If ``factors`` (n x 7) is provided it is used as the per-client latent
    factor matrix instead of sampling fresh ones, so a client's latent
    characteristics stay stable across months. Likewise ``is_cardholder``
    and ``is_loanholder`` may be supplied as pre-computed stable flags.
    """
    if client_ids is None:
        client_ids = [f"C{i:08d}" for i in range(n)]
    rng = _rng_or_new(rng)

    if factors is not None:
        F = np.asarray(factors, dtype=np.float64)
    else:
        F = engine.sample_latent_factors(n)
    f_credit = F[:, LATENT_CREDITWORTHINESS]
    f_income = F[:, LATENT_INCOME_CAPACITY]
    f_loan = F[:, LATENT_LOAN_DEMAND]
    f_asset = F[:, LATENT_ASSET_QUALITY]
    f_stab = F[:, LATENT_STABILITY]
    f_risk = F[:, LATENT_RISK_APPETITE]

    gender = engine.categorical(f_stab, ["F", "M"],
                                np.array([0.65, 0.35]), sensitivity=0.05)
    family_status = engine.categorical(
        f_stab,
        ["Civil marriage", "Married", "Separated", "Single / not married", "Widow"],
        np.array([0.45, 0.08, 0.06, 0.30, 0.11]), sensitivity=0.10)
    education = engine.categorical(
        f_income,
        ["Higher education", "Incomplete higher", "Lower secondary",
         "Secondary / secondary special", "Academic degree"],
        np.array([0.25, 0.10, 0.20, 0.40, 0.05]), sensitivity=0.15)
    income_type = engine.categorical(
        f_income,
        ["Working", "State servant", "Commercial associate", "Pensioner",
         "Unemployed", "Student", "Businessman", "Maternal leave"],
        np.array([0.50, 0.10, 0.15, 0.08, 0.02, 0.01, 0.10, 0.04]),
        sensitivity=0.15)
    housing = engine.categorical(
        f_asset,
        ["House / apartment", "Municipal apartment", "With parents",
         "Office apartment", "Co-op apartment", "Rented apartment"],
        np.array([0.65, 0.05, 0.15, 0.02, 0.08, 0.05]), sensitivity=0.10)
    occupation = engine.categorical(
        f_income,
        ["Laborers", "Core staff", "Accountants", "Sales staff", "Managers",
         "Drivers", "Cleaning staff", "Cooking staff", "Medicine staff",
         "IT staff", "HR staff", "Low-skill Laborers"],
        np.array([0.18, 0.10, 0.08, 0.12, 0.05, 0.06,
                  0.04, 0.04, 0.03, 0.04, 0.03, 0.05]), sensitivity=0.12)

    income = engine.lognormal(f_income, INCOME_LOG_MU, INCOME_LOG_SIGMA,
                              factor_weight=0.7)
    # credit_score: centered at 600 (real HC median) with ~150 1σ spread,
    # driven primarily by creditworthiness then stability then income.
    credit_score = np.clip(
        600 + 120 * (f_credit * 1.0 + f_stab * 0.4 + f_income * 0.25),
        300, 900).astype(np.int32)

    # EXT_SOURCE: three *independent* non-linear transforms of the factors,
    # each a sigmoid of a different linear combination, so they are
    # partially decorrelated and individually informative about default.
    ext_src_1 = 1.0 / (1.0 + np.exp(-(-1.0 + 1.8 * f_credit + 0.3 * f_income)))
    ext_src_2 = 1.0 / (1.0 + np.exp(-(-0.8 + 1.2 * f_stab + 0.4 * f_asset)))
    ext_src_3 = 1.0 / (1.0 + np.exp(-(0.2 + 1.0 * f_credit - 0.3 * f_risk)))

    age = np.clip(38 + f_stab * 10 + rng.standard_normal(n) * 5, 21, 75)
    age = age.astype(np.int32)

    children = np.where(rng.random(n) < 0.55, rng.poisson(1.5, size=n), 0)
    family_members = children + rng.choice([1, 2], size=n)

    owns_car = engine.bernoulli(f_asset, 0.35, factor_weight=0.35)
    owns_realty = engine.bernoulli(f_asset, 0.68, factor_weight=0.25)

    car_age = np.clip(rng.standard_normal(n) * 5 + f_asset * 2, -1, 60)
    car_age = np.where(owns_car, car_age, -1).astype(np.int32)

    region_rating = np.clip((6 - f_stab * 1.2).astype(np.int32), 1, 5)
    region_pop = np.exp(8 + f_stab * 0.3 + rng.standard_normal(n) * 0.5) * 1e4

    days_employed = np.clip(-3000 + f_stab * 1000 + rng.standard_normal(n) * 1500,
                            -14600, 0).astype(np.int32)
    # uniform day within the birth year, so floor(age) at this snapshot == age
    birth_dates = [snapshot_date - timedelta(days=int((a + r) * 365.25))
                   for a, r in zip(age, rng.uniform(0.01, 0.99, n))]

    if is_cardholder is None:
        is_cardholder = engine.bernoulli(f_risk, card_penetration, factor_weight=0.25)
    else:
        is_cardholder = np.asarray(is_cardholder, dtype=bool)
    if is_loanholder is None:
        is_loanholder = engine.bernoulli(f_loan, loan_penetration, factor_weight=0.30)
    else:
        is_loanholder = np.asarray(is_loanholder, dtype=bool)

    org_type = ["Self-employed" if s == "Businessman" else "Other"
                for s in income_type]

    return pl.DataFrame({
        "client_id": client_ids,
        "snapshot_date": [snapshot_date] * n,
        "gender": gender.tolist(),
        "birth_date": birth_dates,
        "age_years": age.tolist(),
        "children_count": children.tolist(),
        "family_members": family_members.tolist(),
        "family_status": family_status.tolist(),
        "education_level": education.tolist(),
        "housing_type": housing.tolist(),
        "income_total": income.tolist(),
        "income_type": income_type.tolist(),
        "occupation_type": occupation.tolist(),
        "organization_type": org_type,
        "employment_start_date": [
            snapshot_date + timedelta(days=int(d)) for d in days_employed],
        "days_employed": days_employed.tolist(),
        "owns_car": owns_car.tolist(),
        "owns_realty": owns_realty.tolist(),
        "car_age": car_age.tolist(),
        "region_rating": region_rating.tolist(),
        "region_population_relative": region_pop.tolist(),
        "credit_score": credit_score.tolist(),
        "ext_source_1": ext_src_1.tolist(),
        "ext_source_2": ext_src_2.tolist(),
        "ext_source_3": ext_src_3.tolist(),
        "is_cardholder": is_cardholder.tolist(),
        "is_loanholder": is_loanholder.tolist(),
        "created_at": [datetime(snapshot_date.year, snapshot_date.month,
                              snapshot_date.day)] * n,
    })


def generate_loan_applications(
    engine: CopulaEngine,
    clients: pl.DataFrame,
    client_factors: np.ndarray,
    month: date,
    params,
    monthly_app_rate: float = 0.03,
    rng: Optional[np.random.Generator] = None,
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Generate loan applications for one month (stream).

    Uses the applicant client's stored factor vector (client_factors) so an
    application's term/amount/approval are structurally dependent on the
    applicant's latent credit profile. Calibrated GLM models drive term,
    amount and approval.

    Returns (submitted, decision, approvals_digest):
      submitted -- pre-decision columns only (the stream event)
      decision  -- outcome columns (status/decision, nightly master update)
      approvals -- digest feeding the account ledger for loan origination
                   (includes application_id for label re-join)
    """
    rng = _rng_or_new(rng)
    fidx = params.factor_idx
    n_apps = max(1, int(clients.height * monthly_app_rate))
    idx = rng.choice(clients.height, size=n_apps, replace=True)
    selected = clients[idx]
    client_ids = selected["client_id"].to_numpy()

    F = np.asarray(client_factors)[idx]
    f_credit = F[:, LATENT_CREDITWORTHINESS]
    f_risk = F[:, LATENT_RISK_APPETITE]

    # Calibrated conditional models
    loan_amount = np.exp(draw_from_model(
        params.model("previous_application", "log_amt_credit"),
        F, rng, fidx))
    loan_amount = np.clip(loan_amount, 5_000, 500_000)
    term_months = draw_from_model(
        params.model("previous_application", "term_months"), F, rng, fidx)
    term_months = np.clip(term_months, 6, 72).astype(np.int32)
    approved = draw_from_model(
        params.model("previous_application", "p_approved"), F, rng, fidx)

    interest_rate = np.clip(
        5.0 + (8 - f_credit * 1.5) + rng.standard_normal(n_apps) * 1.5,
        3.0, 25.0)

    monthly_r = interest_rate / 1200
    annuity = loan_amount * monthly_r / (1 - (1 + monthly_r) ** (-term_months))

    purpose = engine.categorical(
        F[:, LATENT_LOAN_DEMAND],
        ["personal", "auto", "mortgage", "education", "business",
         "debt_consolidation", "other"],
        np.array([0.25, 0.08, 0.30, 0.01, 0.06, 0.20, 0.10]), sensitivity=0.10)

    contract = engine.categorical(
        f_risk, ["Cash loans", "Revolving loans", "Consumer loans"],
        np.array([0.55, 0.15, 0.30]), sensitivity=0.15)

    base_ts = datetime(month.year, month.month, 1)
    offsets = rng.integers(0, 28 * 24 * 3600, n_apps)
    timestamps = [base_ts + timedelta(seconds=int(o)) for o in offsets]
    decision_offsets = rng.integers(3600, 7 * 24 * 3600, n_apps)
    decision_ts = [ts + timedelta(seconds=int(o))
                   for ts, o in zip(timestamps, decision_offsets)]

    goods_price = loan_amount * np.where(
        rng.random(n_apps) < 0.4, 1.0, 0.7 + rng.random(n_apps) * 0.5)

    app_ids = [f"APP-{month.strftime('%Y%m')}-{i:06d}" for i in range(n_apps)]

    # Pre-decision view: the stream event exposes only what is observable
    # at submission time. Status/decision are NOT here.
    submitted = pl.DataFrame({
        "application_id": app_ids,
        "client_id": client_ids.tolist(),
        "submission_timestamp": timestamps,
        "loan_amount": loan_amount.tolist(),
        "loan_term_months": term_months.tolist(),
        "interest_rate": interest_rate.tolist(),
        "purpose": purpose.tolist(),
        "annuity": annuity.tolist(),
        "goods_price": goods_price.tolist(),
        "contract_type": contract.tolist(),
    })

    # Decision outcome: arrives later (nightly master update / CDC),
    # keyed on the same application_id.
    decision = pl.DataFrame({
        "application_id": app_ids,
        "status": ["approved" if a else "declined" for a in approved],
        "decision_timestamp": decision_ts,
        "decision_by": [f"UW-{rng.choice(['A', 'B', 'C', 'D'])}"
                        for _ in range(n_apps)],
    })

    approvals = {
        "application_id": np.array(app_ids)[approved],
        "client_id": client_ids[approved],
        "factors": F[approved],
        "term_months": term_months[approved],
        "principal": loan_amount[approved],
        "annuity": annuity[approved],
    }
    return submitted, decision, approvals


def generate_card_auths(
    engine: CopulaEngine,
    clients: pl.DataFrame,
    start_ts: datetime,
    end_ts: datetime,
    auths_per_card_per_day: float = 1.8,
    decline_rate: float = 0.12,
    merchant_categories: Optional[dict[str, float]] = None,
    rng: Optional[np.random.Generator] = None,
    card_ids: Optional[list[str]] = None,
    client_factors: Optional[np.ndarray] = None,
) -> pl.DataFrame:
    """Generate card authorization events between two timestamps (stream).

    Only cardholding clients generate auths. Not present in Home Credit,
    so fully synthetic from copula + Poisson process.

    ``card_ids`` (optional) must align with
    ``clients.filter(is_cardholder)`` row order; it supplies the real
    ledger-minted card ids so the auth stream FK-links to card balances.
    """
    rng = _rng_or_new(rng)
    if merchant_categories is None:
        merchant_categories = {
            "groceries": 0.22, "dining": 0.14, "retail": 0.18,
            "transport": 0.10, "utilities": 0.08, "entertainment": 0.09,
            "travel": 0.06, "healthcare": 0.05, "education": 0.03, "other": 0.05,
        }

    holder_mask = clients["is_cardholder"].to_numpy().astype(bool)
    cardholders = clients.filter(pl.col("is_cardholder"))
    if cardholders.height == 0:
        return pl.DataFrame(schema={
            "auth_id": pl.Utf8, "card_id": pl.Utf8, "client_id": pl.Utf8,
            "event_timestamp": pl.Datetime, "amount": pl.Float64,
            "currency": pl.Utf8, "merchant_category": pl.Utf8,
            "merchant_name": pl.Utf8, "approved": pl.Boolean,
            "decline_reason": pl.Utf8,
        })

    n_days = max(1, (end_ts - start_ts).days)
    n_events = int(cardholders.height * auths_per_card_per_day * n_days)

    # Assign each event to a random cardholder and random time
    card_idx = rng.choice(cardholders.height, size=n_events, replace=True)
    client_ids = cardholders["client_id"][card_idx].to_list()
    if card_ids is not None and len(card_ids) == cardholders.height:
        card_ids = [card_ids[i] for i in card_idx]
    else:
        card_ids = [f"CC-{cid}-{rng.integers(0, 3)}" for cid in client_ids]

    secs = rng.integers(0, max(1, int((end_ts - start_ts).total_seconds())),
                        n_events)
    timestamps = [start_ts + timedelta(seconds=int(s)) for s in secs]

    # Amount / decline driven by the *cardholder's* factors (previously a
    # fresh factor draw per event, i.e. independent of the client).
    if client_factors is not None:
        F = np.asarray(client_factors)[holder_mask][card_idx]
    else:
        F = engine.sample_latent_factors(n_events)
    f_income = F[:, LATENT_INCOME_CAPACITY]
    amount = engine.lognormal(f_income, CARD_AUTH_AMOUNT_LOG_MU,
                              CARD_AUTH_AMOUNT_LOG_SIGMA, factor_weight=0.4)

    # Merchant category multinomial
    cats = list(merchant_categories.keys())
    probs = np.array(list(merchant_categories.values()))
    probs = probs / probs.sum()
    cat_idx = rng.choice(len(cats), size=n_events, p=probs)
    categories = [cats[i] for i in cat_idx]

    # Approval: driven by decline_rate + creditworthiness
    f_credit = F[:, LATENT_CREDITWORTHINESS]
    decline_prob = decline_rate * (1.5 - f_credit * 0.4)
    decline_prob = np.clip(decline_prob, 0.01, 0.6)
    approved = rng.random(n_events) > decline_prob

    decline_reasons = ["insufficient_funds", "over_limit", "card_blocked", "expired"]
    reason_idx = rng.choice(len(decline_reasons), size=n_events)

    return pl.DataFrame({
        "auth_id": [f"AUTH-{i:09d}" for i in range(n_events)],
        "card_id": card_ids,
        "client_id": client_ids,
        "event_timestamp": timestamps,
        "amount": amount.tolist(),
        "currency": ["EGP"] * n_events,
        "merchant_category": categories,
        "merchant_name": [f"MC-{c[:6].upper()}-{rng.integers(0, 999)}"
                          for c in categories],
        "approved": approved.tolist(),
        "decline_reason": [
            decline_reasons[reason_idx[i]] if not approved[i] else None
            for i in range(n_events)],
    })


# NOTE: the former generate_pos_balances / generate_card_balances /
# generate_bureau / generate_payments helpers were removed. They drew fresh
# latent factors (or none) per row, so their output was independent of the
# client, and nothing called them: child tables come from state.AccountLedger.
