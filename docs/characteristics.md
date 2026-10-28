# Dataset Characteristics

A digital lender (Vietnamese market, VND currency). Borrowers apply for
instalment loans and credit cards; the bank originates loans, tracks monthly
instalment payments, days-past-due (DPD), and card authorizations; an external
credit bureau publishes a monthly account-status extract per borrower. The
simulator generates this world with a **t-copula over 7 latent factors**
(creditworthiness, income capacity, loan demand, asset quality, stability,
risk appetite, default propensity), calibrated from Home Credit data.

The ML target: **default** - a borrower is labelled defaulted if they have 3
consecutive delinquent months within the first 12 instalments of a loan.

---

## Offline Data

Batch-consumed tables (historical + reference data), stored as Parquet on
SeaweedFS S3 (plus local write-through archive). Every entity is written per
month; payments and decisions additionally appear in the streaming lane as
nightly files. Target volume: **>= 100GB across all tables combined**.

**Object-store rule**: S3 holds every BUSINESS ARTIFACT (decision/payment
files, dims, snapshots, gold). The only entities that never go to S3 are the
pure real-time event streams - `loan_applications` and `card_authorizations`
(they stream through CDC and live in the parquet write-through archive
only). Payments / decisions / clients are dual-lane: a stored file on S3
AND a CDC stream.

| Table | Grain | Key Columns | Cadence |
|---|---|---|---|
| clients | one row per client per monthly snapshot | client_id, snapshot_date, age_years, income_total, income_type, occupation_type, region_rating, region_population_relative, credit_score, ext_source_1/2/3, is_cardholder, is_loanholder | monthly |
| application_master | one row per application (submitted + decision joined) | application_id, client_id, submission_timestamp, decision_timestamp, status, loan_amount, loan_term_months, interest_rate, purpose, annuity, goods_price, contract_type | monthly (gold) |
| pos_balances | one row per active loan per month | pos_balance_id, client_id, loan_id, snapshot_date, months_balance, cnt_instalment, cnt_instalment_future, sk_dpd, sk_dpd_def | monthly |
| card_balances | one row per card per month | card_balance_id, card_id, client_id, snapshot_date, amt_balance, amt_credit_limit_actual, amt_drawings_current, amt_payment_current, cnt_drawings_current, sk_dpd | monthly |
| bureau | one row per bureau account per month (external vendor drop) | bureau_id, client_id, snapshot_date, credit_type, credit_amount, credit_start_date, days_credit, amt_credit_sum, amt_credit_sum_overdue, amt_credit_max_overdue, status | monthly (21-day lag) |
| application_decisions | one row per application outcome (dual-lane with the streaming table) | application_id, status (approved/declined), decision_timestamp, decision_by | nightly file |
| payments | one row per instalment payment (dual-lane with the streaming table) | payment_id, loan_id, client_id, application_id, payment_date, amount, principal_portion, interest_portion, days_late, payment_method, num_payment | nightly file (02:00 UTC) |

## Streaming Data

Recurring events consumed by a stream processor (Flink) via CDC topics
(`lending.public.*`), published by Debezium from the postgres-source OLTP
tables.

| Event | Grain | Key Columns | Cadence |
|---|---|---|---|
| loan_applications | one row per submission (pre-decision only) | application_id, client_id, submission_timestamp, loan_amount, loan_term_months, interest_rate, purpose, annuity, goods_price, contract_type | per-day |
| application_decisions | one row per application outcome | application_id, status (approved/declined), decision_timestamp, decision_by | nightly |
| card_authorizations | one row per authorization attempt (incl. declined) | auth_id, card_id, client_id, event_timestamp, amount, currency, merchant_category, merchant_name, approved, decline_reason | per-day (high volume) |
| payments | one row per instalment payment | payment_id, loan_id, client_id, application_id, payment_date, amount, principal_portion, interest_portion, days_late, payment_method, num_payment | nightly file, then streamed |
| clients | client master changes (new signups / updates) | client_id, snapshot_date, income_total, region_rating, is_cardholder, is_loanholder, ... | nightly/monthly master updates |

The division between the two: **offline tables are consumed by batch jobs**
(Spark / DP pipelines, historical and reference data, Parquet on S3), while
**streaming tables are consumed by stream processors** (Flink, via CDC topics).
Cadence is orthogonal - payments and decisions are written as nightly files
*and* streamed through CDC; balances and bureau are monthly snapshots consumed
only by batch.

---

## Required Data Problems

How each rubric-required problem manifests in the credit domain.

### Skew (offline)

- **Region concentration**: `region_rating` / `region_population_relative` in
  `clients` - a dominant region (top-region ratio, configurable) captures most
  of the population.
- **Heavy-tailed income**: `income_total` in `clients` follows a log-normal /
  Pareto-like margin, so a small share of clients hold a large share of
  income (top-1% income concentration).
- **Merchant concentration**: `merchant_category` in `card_authorizations` is
  dominated by groceries (22%), retail (18%), dining (14%).
- **Purpose concentration**: `purpose` in `loan_applications` is dominated by
  mortgage (30%), personal (25%), debt_consolidation (20%).
- **Effect**: any join or aggregation on region, income, merchant category, or
  loan purpose will be skewed; joins on income-binned features run hot on a
  few buckets.

### Schema evolution (offline)

- Mid-history, the external **bureau vendor adds a new column** (e.g.
  `debt_ratio`), and later a breaking change renames/retires a column.
- Earlier bureau months lack the new field (v1, null legacy column); later
  months carry it (v2); a breaking change introduces v3 with a per-row
  `schema_version`.
- **Effect**: any reader of the bureau table must tolerate files with
  different schemas in the same dataset and version-aware handling.

### Duplicates (offline)

- **~2% of `payments` rows** are duplicated (the payment file is re-delivered
  at-least-once); duplicates share the same `payment_id` and content hash.
- **~0.5% of `loan_applications` rows** are duplicated (a retry bug in the
  submission service).
- **Effect**: aggregations (total repayment, application counts, default-rate
  denominators) overcount unless deduplicated by content hash.

### Late arrivals (streaming)

- **Card authorizations** from mobile wallets arrive **5-45 minutes** after
  the event (the app batches events and flushes on connectivity restore).
- **Payment files / decision files** can land late (carrier delay).
- **Effect**: a short processing window (e.g. 1-minute) misses these events
  unless late-data / watermark handling is configured.

### Duplicates (streaming)

- **~1.5% of streamed bundles** (auths / applications) are **replayed** by the
  producer (at-least-once delivery): same event IDs and content hashes, new
  ingestion timestamps.
- **Effect**: counts and status updates are wrong unless idempotent /
  content-hash dedup is applied.

### Burst traffic (streaming)

- **Monthly / end-of-day spikes**: application and authorization volume
  multiplies periodically (e.g. end-of-month closings, payroll days) by a
  configured multiplier.
- **Effect**: the stream pipeline must autoscale / buffer or it drops or
  delays events under the spike.

### Additional problems (suggested)

- **High cardinality**: `bureau_id`, `auth_id`, `payment_id` are effectively
  unique per row ($\mathrm{unique}/\mathrm{total} \approx 1.0$) - joins on these keys are
  high-cardinality joins.
- **Data drift (economic regime)**: the default-propensity curve shifts over
  calendar time (a deterministic controller multiplies the PD logit during a
  "downturn" window) - baseline-calibrated before, drifted after. Every loan
  carries drift metadata (`drift_enabled`, `drift_scenario`, `drift_phase`,
  `drift_factor`), and downstream features/labels drift *emerge* from the
  shifted PD.
- **Out-of-order events**: card authorizations from the same card can arrive
  out of chronological order (couples with late arrival).