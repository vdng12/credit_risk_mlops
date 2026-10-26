"""Constants: column definitions, factor indices, distribution parameters."""

from __future__ import annotations

# Latent factor indices (aligned with CopulaConfig.corr_matrix rows/cols)
LATENT_CREDITWORTHINESS = 0
LATENT_INCOME_CAPACITY = 1
LATENT_LOAN_DEMAND = 2
LATENT_ASSET_QUALITY = 3
LATENT_STABILITY = 4
LATENT_RISK_APPETITE = 5
LATENT_DEFAULT_PROPENSITY = 6

LATENT_FACTOR_NAMES = [
    "creditworthiness",
    "income_capacity",
    "loan_demand",
    "asset_quality",
    "stability",
    "risk_appetite",
    "default_propensity",
]

N_LATENT_FACTORS = 7

# Column definitions per entity
CLIENT_COLUMNS = [
    "client_id", "snapshot_date", "gender", "birth_date", "age_years",
    "children_count", "family_members", "family_status", "education_level",
    "housing_type", "income_total", "income_type", "occupation_type",
    "organization_type", "employment_start_date", "days_employed",
    "owns_car", "owns_realty", "car_age", "region_rating",
    "region_population_relative", "credit_score", "is_cardholder",
    "is_loanholder", "created_at",
]

CARD_AUTH_COLUMNS = [
    "auth_id", "card_id", "client_id", "event_timestamp", "amount",
    "currency", "merchant_category", "merchant_name", "approved",
    "decline_reason",
]

LOAN_APPLICATION_COLUMNS = [
    "application_id", "client_id", "submission_timestamp", "loan_amount",
    "loan_term_months", "interest_rate", "purpose", "annuity",
    "goods_price", "contract_type", "status", "decision_timestamp",
    "decision_by",
]

POS_BALANCE_COLUMNS = [
    "pos_balance_id", "client_id", "loan_id", "snapshot_date",
    "months_balance", "cnt_instalment", "cnt_instalment_future",
    "sk_dpd", "sk_dpd_def",
]

CARD_BALANCE_COLUMNS = [
    "card_balance_id", "card_id", "client_id", "snapshot_date",
    "months_balance", "amt_balance", "amt_credit_limit_actual",
    "amt_drawings_atm_current", "amt_drawings_current",
    "amt_inst_min_regularity", "amt_payment_current",
    "amt_receivable_principal", "cnt_drawings_atm_current",
    "cnt_drawings_current", "sk_dpd",
]

BUREAU_COLUMNS = [
    "bureau_id", "client_id", "snapshot_date", "credit_account_id",
    "credit_type", "credit_amount", "credit_currency",
    "credit_start_date", "credit_end_date", "credit_duration_days",
    "days_credit", "credit_updated_date", "amt_credit_max_overdue",
    "amt_credit_sum", "amt_credit_sum_overdue", "cnt_credit_prolong",
    "credit_active", "days_credit_enddate_fixed", "days_enddate_fact",
]

PAYMENT_COLUMNS = [
    "payment_id", "loan_id", "client_id", "payment_date", "amount",
    "principal_portion", "interest_portion", "days_late",
    "payment_method", "num_payment",
]

# Distribution parameters for continuous columns (log-normal mu, sigma)
INCOME_LOG_MU = 12.0
INCOME_LOG_SIGMA = 0.6
CREDIT_AMOUNT_LOG_MU = 11.0
CREDIT_AMOUNT_LOG_SIGMA = 0.9
CARD_AUTH_AMOUNT_LOG_MU = 3.5
CARD_AUTH_AMOUNT_LOG_SIGMA = 1.2
CARD_LIMIT_LOG_MU = 9.5
CARD_LIMIT_LOG_SIGMA = 0.8
CREDIT_SCORE_MEAN = 650
CREDIT_SCORE_STD = 85
