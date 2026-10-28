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

## Cadence model: daily, nightly, monthly

A lending bank does not emit all data on one schedule. Delivery is split
into three rhythms with distinct entity sets:

| Rhythm | What happens | Entities |
|---|---|---|
| **daily (hot)** | card authorizations and loan submissions stream continuously, day by day | loan_applications, card_authorizations |
| **nightly** | the day's payment file and application-decision outcomes land (carrier-style file) | payments, application_decisions |
| **monthly close** | the bank closes the month: balances, the external bureau drop (with its 21-day vendor lag), the client master snapshot and the gold application master | clients, application_master, pos_balances, card_balances, bureau |

Two consumption lanes serve these rhythms:

- **Streaming lane** - rows land in the postgres-source OLTP tables and are
  re-published by Debezium CDC to Redpanda topics (`lending.public.*`) for
  stream processors (Flink). The CDC messages are keyed by the OLTP primary
  keys and carry the Debezium change-envelope (before/after/source).
- **Batch lane** - artifacts land as Parquet on SeaweedFS S3 (immutable
  cold facts) plus a local write-through archive for every entity. Batch
  jobs (Spark / DP pipelines) read these.

Cadence and lane are **orthogonal**: a nightly file both lands on S3 (the
bank keeps the file) and streams through CDC (consumers see the events),
while a monthly snapshot is batch-consumed only.

**Object-store rule**: S3 holds every business artifact - decision /
payment files, dims, snapshots, gold. The only entities that never go to S3
are the pure real-time event streams (`loan_applications`,
`card_authorizations`): they live in the parquet archive and stream through
CDC. Payments / decisions / clients are dual-lane (file on S3 AND stream).

---

## Offline Data

Batch-consumed tables (historical + reference data, Parquet on SeaweedFS
S3 plus local write-through archive). Target volume: **>= 100GB across all
tables combined**.

### clients

One row per client per monthly snapshot. The client master: demographics
(gender, age, children, family), employment and income, assets (car,
realty), region rating and population share, external credit scores
(ext_source_1/2/3), card/loan holder flags. Attribute values are drawn once
from the latent factors and stay stable across months; only the snapshot
date and derived age move. Written monthly; also streams as a dimension.

### application_master

One row per application (submitted view joined with its decision outcome).
The **gold layer**: submission timestamp, requested amount / term / rate,
purpose, contract type, annuity, goods price, plus the decision status / by
/ timestamp. Written monthly; gold, no CDC. Key for label re-join.

### pos_balances

One row per active loan per month. Instalment balance panel (Home Credit's
POS_CASH_balance style): months remaining, instalment counts, and the
days-past-due level (`sk_dpd`) with the >90-day default flag. Monthly
snapshot; batch-only.

### card_balances

One row per card per month. Card balance panel: current balance, credit
limit, drawings and payments for the period, draw count, DPD level.
Monthly snapshot; batch-only.

### bureau

One row per bureau account per month - the external vendor's extract.
Credit type and amount, account age, overdue amounts, and the STATUS
Markov-chain state (mostly closed/current, matching the real vendor mix).
Arrives monthly with a 21-day lag; batch-only.

### application_decisions

One row per application outcome (dual-lane with the streaming table):
status (approved/declined), decision timestamp, underwriter id. Nightly
file on S3, then streamed.

### payments

One row per instalment payment (dual-lane with the streaming table):
payment id, loan / client / application references, payment date (each loan
pays on its own stable due day 1..28, so dates spread across the month),
amount, principal/interest split, days late, method, instalment number.
Nightly file (02:00 UTC) on S3, then streamed.

| Table | Grain | Cadence | Lane |
|---|---|---|---|
| clients | one row per client per monthly snapshot | monthly | batch + stream (dim) |
| application_master | one row per application (submitted x decision) | monthly (gold) | batch |
| pos_balances | one row per active loan per month | monthly | batch |
| card_balances | one row per card per month | monthly | batch |
| bureau | one row per bureau account per month | monthly (21-day lag) | batch |
| application_decisions | one row per outcome | nightly | batch + stream |
| payments | one row per instalment | nightly (02:00 UTC) | batch + stream |

---

## Streaming Data

Recurring events consumed by stream processors (Flink) via CDC topics
(`lending.public.*`), published by Debezium from the postgres-source OLTP
tables.

### loan_applications

One row per submission, **pre-decision only** (what is observable at
submission time): requested amount, term, rate, purpose, contract type,
annuity, goods price, application/client/ids, submission timestamp.
Per-day, high-enough volume to stream continuously; the outcome arrives
later in `application_decisions` keyed on the same application id.

### card_authorizations

One row per authorization attempt (approved and declined): auth id, card /
client, event timestamp, amount (VND), merchant category/name, approval
flag, decline reason. True per-transaction events: per-day, high volume,
spread across the month by street time.

### application_decisions / payments

The same tables as the batch lane (dual-lane): decisions nightly, payments
nightly - both re-published through CDC so consumers see the file's events
as they land.

### clients

Client master changes (new signups / master updates) stream as a dimension
for enrichment; the full snapshot is the monthly batch artifact.

| Event | Grain | Cadence |
|---|---|---|
| loan_applications | one row per submission (pre-decision) | per-day |
| application_decisions | one row per outcome | nightly |
| card_authorizations | one row per authorization attempt | per-day (high volume) |
| payments | one row per instalment | nightly, then streamed |
| clients | master changes | nightly / monthly updates |

---

## Required Data Problems

How each rubric-required problem manifests in the credit domain.

### Skew (offline)

- **Region concentration**: `region_rating` / `region_population_relative`
  in `clients` - a dominant region (top-region ratio, configurable)
  captures most of the population.
- **Heavy-tailed income**: `income_total` in `clients` follows a log-normal /
  Pareto-like margin, so a small share of clients hold a large share of
  income (top-1% income concentration).
- **Merchant concentration**: `merchant_category` in `card_authorizations`
  is dominated by groceries (22%), retail (18%), dining (14%).
- **Purpose concentration**: `purpose` in `loan_applications` is dominated
  by mortgage (30%), personal (25%), debt_consolidation (20%).
- **Effect**: any join or aggregation on region, income, merchant category,
  or loan purpose will be skewed; joins on income-binned features run hot
  on a few buckets.

### Schema evolution (offline)

- Mid-history, the external **bureau vendor adds a new column** (e.g.
  `debt_ratio`), and later a breaking change renames/retires a column.
- Earlier bureau months lack the new field (v1, null legacy column); later
  months carry it (v2); a breaking change introduces v3 with a per-row
  `schema_version`.
- **Effect**: any reader of the bureau table must tolerate files with
  different schemas in the same dataset and version-aware handling.

### Duplicates (offline)

- **~2% of `payments` rows** are duplicated (the payment file is
  re-delivered at-least-once); duplicates share the same `payment_id` and
  content hash.
- **~0.5% of `loan_applications` rows** are duplicated (a retry bug in the
  submission service).
- **Effect**: aggregations (total repayment, application counts,
  default-rate denominators) overcount unless deduplicated by content hash.

### Late arrivals (streaming)

- **Card authorizations** from mobile wallets arrive **5-45 minutes** after
  the event (the app batches events and flushes on connectivity restore).
- **Payment files / decision files** can land late (carrier delay).
- **Effect**: a short processing window (e.g. 1-minute) misses these events
  unless late-data / watermark handling is configured.

### Duplicates (streaming)

- **~1.5% of streamed bundles** (auths / applications) are **replayed** by
  the producer (at-least-once delivery): same event IDs and content hashes,
  new ingestion timestamps.
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
  unique per row ($\mathrm{unique}/\mathrm{total} \approx 1.0$) - joins on
  these keys are high-cardinality joins.
- **Data drift (economic regime)**: the default-propensity curve shifts over
  calendar time (a deterministic controller multiplies the PD logit during
  a "downturn" window) - baseline-calibrated before, drifted after. Every
  loan carries drift metadata (`drift_enabled`, `drift_scenario`,
  `drift_phase`, `drift_factor`), and downstream features/labels drift
  *emerge* from the shifted PD.
- **Out-of-order events**: card authorizations from the same card can arrive
  out of chronological order (couples with late arrival).
