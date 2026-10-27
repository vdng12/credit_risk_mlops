-- =============================================================================
-- PostgreSQL init — Lending source system (OLTP)
-- =============================================================================
-- This schema defines the SOURCE tables that the simulator writes to.
-- Changes flow out via CDC (Debezium → Redpanda) for real-time consumption.
-- Analytical/batch entities (pos_balances, card_balances, bureau, master)
-- go to SeaweedFS S3 as parquet files, NOT to Postgres.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ===========================================================================
-- Core OLTP tables (written by simulator sinks, consumed via CDC)
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- clients — customer master (monthly snapshot, upserted)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS clients (
    client_id           TEXT PRIMARY KEY,
    snapshot_date       DATE NOT NULL,
    gender              TEXT,
    birth_date          DATE,
    age_years           INTEGER,
    children_count      INTEGER,
    family_members      INTEGER,
    family_status       TEXT,
    education_level     TEXT,
    housing_type        TEXT,
    income_total        DOUBLE PRECISION,
    income_type         TEXT,
    occupation_type     TEXT,
    organization_type   TEXT,
    employment_start_date DATE,
    days_employed       INTEGER,
    owns_car            BOOLEAN,
    owns_realty         BOOLEAN,
    car_age             DOUBLE PRECISION,
    region_rating       TEXT,
    region_population_relative DOUBLE PRECISION,
    credit_score        INTEGER,
    is_cardholder       BOOLEAN,
    is_loanholder       BOOLEAN,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_clients_snapshot ON clients(snapshot_date DESC);

-- ---------------------------------------------------------------------------
-- loan_applications — pre-decision submissions (stream + CDC)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS loan_applications (
    application_id      TEXT PRIMARY KEY,
    client_id           TEXT,
    submission_timestamp TIMESTAMPTZ,
    loan_amount         DOUBLE PRECISION,
    loan_term_months    INTEGER,
    interest_rate       DOUBLE PRECISION,
    purpose             TEXT,
    annuity             DOUBLE PRECISION,
    goods_price         DOUBLE PRECISION,
    contract_type       TEXT
);

CREATE INDEX IF NOT EXISTS idx_loan_apps_client ON loan_applications(client_id);

-- ---------------------------------------------------------------------------
-- application_decisions — decision outcomes (nightly CDC batch)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS application_decisions (
    application_id      TEXT PRIMARY KEY,
    status              TEXT NOT NULL,
    decision_timestamp  TIMESTAMPTZ,
    decision_by         TEXT
);

CREATE INDEX IF NOT EXISTS idx_app_decisions_status ON application_decisions(status);

-- ---------------------------------------------------------------------------
-- payments — instalment payments (transactional)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS payments (
    payment_id          TEXT PRIMARY KEY,
    loan_id             TEXT,
    client_id           TEXT,
    application_id      TEXT,
    payment_date        DATE NOT NULL,
    amount              DOUBLE PRECISION,
    principal_portion   DOUBLE PRECISION,
    interest_portion    DOUBLE PRECISION,
    days_late           INTEGER,
    payment_method      TEXT,
    num_payment         INTEGER
);

CREATE INDEX IF NOT EXISTS idx_payments_loan ON payments(loan_id);
CREATE INDEX IF NOT EXISTS idx_payments_date ON payments(payment_date DESC);

-- ---------------------------------------------------------------------------
-- card_authorizations — card auth event stream (also stored for CDC replay)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS card_authorizations (
    auth_id             TEXT PRIMARY KEY,
    card_id             TEXT,
    client_id           TEXT,
    event_timestamp     TIMESTAMPTZ,
    amount              DOUBLE PRECISION,
    currency            TEXT,
    merchant_category   TEXT,
    merchant_name       TEXT,
    approved            BOOLEAN,
    decline_reason      TEXT
);

CREATE INDEX IF NOT EXISTS idx_card_auths_client ON card_authorizations(client_id);
CREATE INDEX IF NOT EXISTS idx_card_auths_timestamp ON card_authorizations(event_timestamp DESC);

-- ===========================================================================
-- HOT PATH (CDC): Debezium captures writes to the tables above and publishes
-- change events to Redpanda topics. Downstream consumers (Flink, dbt) route
-- events to either real-time pipelines or the object store (SeaweedFS S3).
-- See docs/infra_ops_20261025.md for the full data flow.
-- ===========================================================================
--
-- Tables NOT in this schema (live in S3 as parquet, not in source PG):
--   application_master  — submitted + decision joined (gold master)
--   pos_balances         — monthly instalment balance snapshots
--   card_balances        — monthly card balance snapshots
--   bureau               — external bureau monthly drops