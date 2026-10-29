import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Data Generator Architecture Overview

    The simulator produces a synthetic lending world via a **t-copula over 7 latent factors**
    (creditworthiness, income_capacity, loan_demand, asset_quality, stability, risk_appetite,
    default_propensity), calibrated from Home Credit data.

    ```
    t-copula (7 latent factors, N(0,1) margins)
            |
            v
    per-client factor vector f in R^7   (drawn once, stable for life)
            |-- client / application features      f_0..f_5 -> marginals
            |-- child-table links                  f         -> binned / GLM curves
            |-- bureau STATUS                      Markov chain, matrix by f_6 bucket
            `-- loan default                       f_6 -> PD curve -> per-loan hazard
                                                            -> DPD path -> windowed label
    ```

    Nine entity types are produced across **3 cadence rhythms** and delivered through
    **2 consumption lanes** (batch + streaming).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### The 7 Latent Factors

    Each client draws a latent vector $f \in \mathbb{R}^7$ once and keeps it for life.

    | idx | Factor | "+" means | Source |
    |---|---|---|---|
    | 0 | creditworthiness | safer | PC1 of EXT_SOURCE_1/2/3 |
    | 1 | income_capacity | richer | PC1 of log income, days employed, income type, occupation |
    | 2 | loan_demand | bigger loan | PC1 of log credit / annuity / goods price |
    | 3 | asset_quality | more assets | PC1 of car/realty flags, car age, housing |
    | 4 | stability | older | PC1 of age, children, family, gender, education |
    | 5 | risk_appetite | revolving | PC1 of contract type, suite |
    | 6 | default_propensity | riskier | out-of-fold GBM PD score (HC TARGET) |

    **Tail dependence**: the t-copula's $\nu$ controls joint extremes ($\nu$ small = stronger tail dependence).
    Current calibration hit $\nu=30$ (meaning near-Gaussian). Calibration is provisional pending re-run
    on real HC data.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### The 9 Entities

    | Entity | Grain | Golden source | Aliases |
    |---|---|---|---|
    | **clients** | 1 row per client per monthly snapshot | Generator (stable from f) | dim_client, client master |
    | **loan_applications** | 1 row per submission (pre-decision) | Hot stream | -- |
    | **card_authorizations** | 1 row per auth attempt | Hot stream | -- |
    | **application_decisions** | 1 row per outcome | Nightly file | -- |
    | **payments** | 1 row per instalment | Nightly file (02:00 UTC) | -- |
    | **application_master** | 1 row per application (submitted x decision) | Monthly gold (derived) | fact_application |
    | **pos_balances** | 1 row per active loan per month | Monthly close | fact_pos_cash_month |
    | **card_balances** | 1 row per card per month | Monthly close | fact_card_month |
    | **bureau** | 1 row per bureau account per month | Monthly close (21-day lag) | fact_bureau_credit |

    **Key design**: Unlike the old HC star-schema (dim_client + many facts), the generator
    produces **flat entities** that map to source OLTP tables. The star-schema decomposition
    is a downstream transformation (dbt layer), not a generation concern.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Cadence Model: Daily / Nightly / Monthly

    A lending bank does not emit all data on one schedule. Delivery is split
    into three rhythms with distinct entity sets:

    | Rhythm | What happens | Entities |
    |---|---|---|
    | **daily (hot)** | card auths and loan submissions stream continuously, day by day | loan_applications, card_authorizations |
    | **nightly** | the day's payment file and decision outcomes land (carrier-style file) | payments, application_decisions |
    | **monthly close** | the bank closes the month: balances, the external bureau drop (21-day lag), the client master snapshot, the gold application master | clients, application_master, pos_balances, card_balances, bureau |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Lane Architecture: Batch + Streaming

    Two consumption lanes serve these rhythms. **Cadence and lane are orthogonal**:

    * **Batch lane** -- artifacts land as Parquet on SeaweedFS S3 (immutable cold facts)
      plus a local write-through archive for every entity. Batch jobs (Spark / DP) read these.
    * **Streaming lane** -- rows land in the postgres-source OLTP tables and are re-published
      by Debezium CDC to Redpanda topics (`lending.public.*`) for stream processors (Flink).

    ### Object-store rule

    - S3 holds every business artifact: clients, application_master, decisions,
      payments, pos_balances, card_balances, bureau.
    - Pure-stream-only entities (loan_applications, card_authorizations) live in the
      parquet archive and stream through CDC -- they **never** go to S3.
    - Payments, decisions, and clients are **dual-lane** (file on S3 AND stream).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Sink Routing (canonical)

    | Entity | Local archive | S3 | Postgres OLTP (-> CDC) | Notes |
    |---|---|---|---|---|
    | clients | Y | Y | Y | Monthly snapshot + dim stream |
    | loan_applications | Y | N | Y | Hot stream, never S3 |
    | card_authorizations | Y | N | Y | Hot stream, never S3 |
    | application_decisions | Y | Y | Y | Nightly file + stream |
    | payments | Y | Y | Y | Nightly file + stream |
    | application_master | Y | Y | Y (gold) | Gold layer, no CDC |
    | pos_balances | Y | Y | N | Monthly close, batch only |
    | card_balances | Y | Y | N | Monthly close, batch only |
    | bureau | Y | Y | N | Monthly close (21d lag), batch only |
    """)
    return


@app.cell
def _(mo):
    mo.md(r'''
    ### LiveStage: the 3-rhythm dispatcher

    The generator's core loop uses **SimClock** boundary flags to dispatch each entity
    to the correct lane at the right time:
    ''')

    mo.mermaid('''graph TB
        subgraph CLOCK["SimClock (month boundary)"]
            DAY["per-day loop"]
            NIGHT["day-end (02:00 UTC)"]
            MONTH["month-end close"]
        end

        subgraph HOT["Daily hot stream"]
            LA["loan_applications<br/>(pre-decision)"]
            CA["card_authorizations"]
        end

        subgraph NIGHTLY["Nightly batch"]
            DEC["application_decisions"]
            PAY["payments"]
        end

        subgraph MONTHLY["Monthly close"]
            CL["clients (snapshot)"]
            AM["application_master (gold)"]
            PB["pos_balances"]
            CB["card_balances"]
            BU["bureau (21d lag)"]
        end

        DAY -->|"continuous"| HOT
        NIGHT -->|"file-based"| NIGHTLY
        MONTH -->|"snapshot"| MONTHLY

        subgraph OFFLINE["Offline Path (batch)"]
            PQ["Parquet archive<br/>outputs/generated/"]
            S3["SeaweedFS S3<br/>risk-data bucket"]
        end

        subgraph ONLINE["Online Path (stream)"]
            PG["Postgres OLTP<br/>lending_db"]
            DC["Debezium CDC"]
            RP["Redpanda<br/>lending.public.*"]
        end

        HOT --> ONLINE
        HOT --> PQ
        NIGHTLY --> ONLINE
        NIGHTLY --> OFFLINE
        MONTHLY --> OFFLINE
        AM --> PG
    ''')
    return


@app.cell
def _(mo):
    mo.md(r'''
    ## Entity Relationships
    ''')

    mo.mermaid('''erDiagram
        CLIENT["clients (monthly snapshot)"] {
            int client_id PK
            string gender
            int age
            int children
            string family_status
            string education_type
            string income_type
            string occupation
            string region
            float ext_source_1
            float ext_source_2
            float ext_source_3
            date snapshot_date
        }

        LOAN_APP["loan_applications (pre-decision)"] {
            int application_id PK
            int client_id FK
            float amount_credit
            float amount_annuity
            float amount_goods_price
            string contract_type
            string purpose
            string channel
            timestamp submitted_at
        }

        DECISION["application_decisions"] {
            int application_id PK,FK
            string status
            string decision_by
            timestamp decision_timestamp
        }

        PAYMENT["payments"] {
            int payment_id PK
            int application_id FK
            int instalment_num
            float amount
            float principal_split
            float interest_split
            int days_late
            date payment_date
        }

        APP_MASTER["application_master (gold)"] {
            int application_id PK
            int client_id FK
            float amount_credit
            float amount_annuity
            string contract_type
            string status
            int loan_term
            date disbursement_date
            float interest_rate
            string drift_phase
            int label_default
        }

        POS_BAL["pos_balances"] {
            int application_id PK,FK
            int month
            int sk_dpd
            int instalments_remaining
            int default_flag
        }

        CARD["card_authorizations"] {
            int auth_id PK
            int client_id FK
            float amount
            string merchant_category
            string merchant_name
            string approval_flag
            string decline_reason
            timestamp event_timestamp
        }

        CARD_BAL["card_balances"] {
            int card_id PK
            int client_id FK
            int month
            float balance
            float credit_limit
            float drawings_amount
            int sk_dpd
        }

        BUREAU["bureau"] {
            int bureau_id PK
            int client_id FK
            int month
            string credit_type
            float amount
            int days_overdue
            string status_markov
            string schema_version
        }

        CLIENT ||--o{ LOAN_APP : "applies"
        CLIENT ||--o{ CARD : "swipes"
        CLIENT ||--o{ CARD_BAL : "holds"
        CLIENT ||--o{ BUREAU : "bureau"
        LOAN_APP ||--o| DECISION : "outcome"
        LOAN_APP ||--o{ PAYMENT : "repays"
        LOAN_APP ||--o| APP_MASTER : "gold"
        LOAN_APP ||--o| POS_BAL : "loan"
    ''')
    return


@app.cell
def _(mo):
    mo.md(r'''
    ## Generator Pipeline

    The data flows through a clean pipeline with problems injected lane-specifically:
    ''')

    mo.mermaid('''graph LR
        subgraph GEN["Generator (core.py)"]
            COP["CopulaEngine<br/>t-copula sampling"]
            FACT["per-client factors f in R^7"]
            ENT["Entity generators<br/>clients, apps, auths,<br/>payments, bureau<br/>cards, balances"]
            STATE["AccountLedger<br/>RunHazard<br/>DPD path"]
        end

        subgraph BATCH["Offline Path"]
            direction TB
            OFFLINE_PROBS["Offline problems<br/>* skew (region/income)<br/>* schema evolution (bureau v1/v2/v3)<br/>* exact duplicates (2% pay, 0.5% apps)"]
            SINK_BATCH["Parquet archive + S3"]
        end

        subgraph STREAM["Online Path"]
            direction TB
            STREAM_PROBS["Streaming problems<br/>* late arrival (5-45 min)<br/>* duplicate replay (1.5%)<br/>* burst traffic (month-end Nx)"]
            PG["Postgres OLTP<br/>6 tables"]
            DC["Debezium CDC"]
            RP["Redpanda topics<br/>lending.public.*"]
        end

        GEN -->|"MonthBatch"| BATCH
        GEN -->|"LiveStage<br/>(cadence dispatch)"| STREAM
        STATE -->|"default label<br/>windowed derivation"| APP_MASTER
    ''')

    mo.md(r'''
    ### Run lifecycle

    | Step | What | Lane |
    |---|---|---|
    | **backfill** | months [start, live_start) -> batch path | offline |
    | **bootstrap** | dimensions (clients, cards, loans) -> postgres once | online |
    | **live** | daily hot + nightly + monthly close -> both lanes | dual |
    | **resume** | deterministic replay from seed = byte-identical | both |
    ''')
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Gold Layer: application_master

    The `application_master` is the **gold fact table** -- one row per application
    with submitted terms + decision outcome + loan status + default label.

    ### Default definition

    Default = 1 if the borrower has **3 consecutive delinquent months** within
    the first 12 instalments of a loan. Delinquency is determined by the
    **RunHazard** model: each loan has a per-instalment i.i.d. probability
    of delinquency, inverted from the calibrated PD curve ($f_6 \to P(\text{default})$).

    ```
    PD_curve:  f_6 --> P_base(default)
    logit_shift:        brentq so approval-weighted mean = target 8.07%
    hazard_inversion:   g_n^{-1}(u) per loan -> month-by-month DPD path
    ```

    ### Gold fields (key)

    | Field | Source | Notes |
    |---|---|---|
    | application_id | Generator | Stable across cadences |
    | client_id | Generator | References client snapshot |
    | amount_credit, amount_annuity | Loan terms | From entity generator |
    | contract_type | Product type | Cash vs Revolving |
    | status | Decision + lifecycle | Approved -> Active -> Closed/Defaulted |
    | loan_term | Contract | 6-60 months |
    | interest_rate | Pricing model | Conditional on $f_0$ (creditworthiness) |
    | drift_phase | Drift controller | baseline / pre_drift / post_drift |
    | drift_factor | Controller | Multiplier on PD logit |
    | label_default | Windowed derivation | 1 if 3 delinquent months in first 12 |
    | sk_dpd_max | Maximum DPD | Days past due at peak |
    """)
    return


@app.cell
def star_schema(mo):
    mo.vstack([
        mo.md(r'''
    ## Star-Schema: Dim-Fact Decomposition [DEFERRED]

    *This is a forward-looking design for the downstream analytics layer (dbt / Spark).
    Not implemented in the generator -- the generator produces flat OLTP entities
    and the star-schema is a downstream transformation.*
    '''),
        mo.md(r'''
    ### Derived Dimensions

    | Dimension | Grain | Source entities | Type |
    |---|---|---|---|
    | **dim_client** | 1 row per client (SCD type-2) | clients (monthly snapshot) | Slowly-changing |
    | **dim_date** | 1 row per calendar day | Generated from date range | Static |
    | **dim_product** | 1 row per product type | loan_applications (contract_type, purpose) | Static |
    | **dim_region** | 1 row per region | clients (region_rating, population_share) | Static |
    | **dim_merchant** | 1 row per merchant category + name | card_authorizations | Static |
    | **dim_bureau_status** | 1 row per STATUS code | bureau (STATUS Markov states) | Static |
    | **dim_application_outcome** | 1 row per status value | application_decisions + application_master | Static |
    '''),
        mo.md(r'''
    ### Derived Facts

    | Fact | Grain | Source entities | Measures |
    |---|---|---|---|
    | **fact_application** | 1 row per application | loan_applications + application_decisions + application_master | amount_credit, amount_annuity, amount_goods_price, interest_rate, loan_term, sk_dpd_max, label_default |
    | **fact_payment** | 1 row per instalment | payments | amount, principal_split, interest_split, days_late |
    | **fact_card_auth** | 1 row per authorization | card_authorizations | auth_amount, approval_flag |
    | **fact_pos_balance_month** | 1 row per (loan, month) | pos_balances | sk_dpd, instalments_remaining, default_flag |
    | **fact_card_balance_month** | 1 row per (card, month) | card_balances | balance, credit_limit, drawings_amount, sk_dpd |
    | **fact_bureau_account_month** | 1 row per (bureau, month) | bureau | amount_overdue, credit_amount, status |
    '''),
        mo.mermaid('''erDiagram
        DIM_CLIENT["dim_client"] {
            int client_key PK
            int client_id UK "natural key"
            string gender
            int age
            string family_status
            string education_type
            string income_type
            string occupation_type
            string region
            float ext_source_1
            float ext_source_2
            float ext_source_3
            date valid_from
            date valid_to
            bool is_current
        }
        DIM_DATE["dim_date"] {
            int date_key PK
            date full_date
            int day
            int month
            int year
            int quarter
            string month_name
            bool is_weekend
        }
        DIM_PRODUCT["dim_product"] {
            int product_key PK
            string contract_type
            string purpose
            string term_bracket
            string channel
        }
        DIM_REGION["dim_region"] {
            int region_key PK
            string region_code
            int region_rating
            float region_population_relative
        }
        DIM_MERCHANT["dim_merchant"] {
            int merchant_key PK
            string merchant_name
            string merchant_category
        }
        DIM_BUREAU_STATUS["dim_bureau_status"] {
            int status_key PK
            string status_code
            string status_label
        }
        DIM_OUTCOME["dim_application_outcome"] {
            int outcome_key PK
            string decision_status
            string lifecycle_state
        }

        FACT_APP["fact_application"] {
            int application_id PK
            int client_key FK
            int product_key FK
            int region_key FK
            int outcome_key FK
            int date_submitted_key FK
            float amount_credit
            float amount_annuity
            float amount_goods_price
            float interest_rate
            int loan_term
            int sk_dpd_max
            string drift_phase
            float drift_factor
            int label_default
        }
        FACT_PAY["fact_payment"] {
            int payment_id PK
            int application_id FK
            int client_key FK
            int date_payment_key FK
            float amount
            float principal_split
            float interest_split
            int days_late
            int instalment_number
        }
        FACT_AUTH["fact_card_auth"] {
            int auth_id PK
            int client_key FK
            int merchant_key FK
            int date_auth_key FK
            float auth_amount
            string approval_flag
            string decline_reason
        }
        FACT_POS["fact_pos_balance_month"] {
            int application_id FK
            int client_key FK
            int month_key FK
            int sk_dpd
            int instalments_remaining
            int default_flag
        }
        FACT_CARD["fact_card_balance_month"] {
            int card_id PK
            int client_key FK
            int month_key FK
            float balance
            float credit_limit
            float drawings_amount
            int sk_dpd
        }
        FACT_BUREAU["fact_bureau_account_month"] {
            int bureau_id PK
            int client_key FK
            int status_key FK
            int month_key FK
            string credit_type
            float credit_amount
            float amount_overdue
            int days_overdue
            string schema_version
        }
        DIM_CLIENT ||--o{ FACT_APP : ""
        DIM_CLIENT ||--o{ FACT_PAY : ""
        DIM_CLIENT ||--o{ FACT_AUTH : ""
        DIM_CLIENT ||--o{ FACT_POS : ""
        DIM_CLIENT ||--o{ FACT_CARD : ""
        DIM_CLIENT ||--o{ FACT_BUREAU : ""
        DIM_DATE ||--o{ FACT_APP : "submitted"
        DIM_DATE ||--o{ FACT_PAY : "paid"
        DIM_DATE ||--o{ FACT_AUTH : "auth"
        DIM_DATE ||--o{ FACT_POS : "month"
        DIM_DATE ||--o{ FACT_CARD : "month"
        DIM_DATE ||--o{ FACT_BUREAU : "month"
        DIM_PRODUCT ||--o{ FACT_APP : ""
        DIM_REGION ||--o{ FACT_APP : ""
        DIM_OUTCOME ||--o{ FACT_APP : ""
        DIM_MERCHANT ||--o{ FACT_AUTH : ""
        DIM_BUREAU_STATUS ||--o{ FACT_BUREAU : ""
    '''),
        mo.callout(
            mo.md(r'''
    **Deferred.** This star-schema is the target for a future dbt / Spark transformation
    layer that consumes the generator's flat OLTP entities. The generator itself
    stays entity-oriented for three reasons:

    1. **Source fidelity** -- real source systems (LOS, LMS, Card Processor, Bureau)
       produce OLTP rows, not star-schema facts. Writing OLTP-first means the CDC stream
       mirrors production.
    2. **Lane symmetry** -- flat entities flow identically through both offline (Parquet/S3)
       and online (postgres -> CDC -> Redpanda) lanes. A star-schema would require
       different materialisation strategies per lane.
    3. **Schema evolution** -- flat entities tolerate per-column schema changes
       (bureau v1/v2/v3) more naturally than a denormalised fact table.
    '''),
            kind="info"
        )
    ])
    return


@app.cell
def dp2_pipeline(mo):
    mo.vstack([
        mo.md(r'''
    ## DP2: Bronze -> Silver Transformation Pipeline [DEFERRED]

    *This section designs the transformation from the generator's flat OLTP entities
    (Bronze) to the Dim-Fact star-schema (Silver). Not implemented in the generator
    -- these are downstream Spark / dbt jobs consuming the Parquet archive and/or
    CDC stream.*
    '''),
        mo.mermaid('''graph TB
        subgraph BRONZE["Bronze -- Raw Entities (generator output)"]
            CL["clients"]
            LA["loan_applications"]
            AD["application_decisions"]
            AM["application_master"]
            PM["payments"]
            CA["card_authorizations"]
            PB["pos_balances"]
            CB["card_balances"]
            BU["bureau"]
        end

        subgraph SILVER["Silver -- Dims + Facts"]
            DC["dim_client<br/>(SCD type-2)"]
            DD["dim_date"]
            DP["dim_product"]
            DR["dim_region"]
            DM["dim_merchant"]
            DB["dim_bureau_status"]
            DO["dim_application_outcome"]

            FA["fact_application"]
            FP["fact_payment"]
            FAUTH["fact_card_auth"]
            FPOS["fact_pos_balance_month"]
            FCARD["fact_card_balance_month"]
            FBUR["fact_bureau_account_month"]
        end

        CL --> DC
        CL --> DR

        LA --> DP
        LA --> FA
        AD --> DO
        AD --> FA
        AM --> FA

        PM -.->|"content-hash dedup"| FP
        CA -.->|"late-arrival handling"| DM
        CA -.->|"content-hash dedup"| FAUTH

        PB --> FPOS
        CB --> FCARD
        BU --> DB
        BU -.->|"schema-evolution tolerant"| FBUR

        DC --> FA
        DC --> FP
        DC --> FAUTH
        DC --> FPOS
        DC --> FCARD
        DC --> FBUR
        DD --> FA
        DD --> FP
        DD --> FAUTH
        DD --> FPOS
        DD --> FCARD
        DD --> FBUR
        DP --> FA
        DR --> FA
        DO --> FA
        DM --> FAUTH
        DB --> FBUR
    '''),
        mo.md(r'''
    ### Transformation Details

    ---

    **dim_client** -- SCD type-2 from clients snapshots

    | Step | Logic |
    |---|---|
    | Source | `clients` (monthly snapshot, one row per client per month) |
    | Dedup | `(client_id, snapshot_date)` is unique per snapshot |
    | SCD logic | Compare consecutive snapshots: if any attribute changed, close current row (`valid_to = snapshot_date - 1d`) and insert new row (`valid_from = snapshot_date`) |
    | is_current | Flag set on latest row per client_id |
    | Key challenge | First-snapshot bootstrap: all clients appear in month 1 with no prior history |

    ---

    **dim_date** -- generated calendar

    | Step | Logic |
    |---|---|
    | Source | Synthesized from `min(event_timestamp)` to `max(event_timestamp) + 90d` across all entities |
    | Columns | date_key (int YYYYMMDD), full_date, day, month, year, quarter, month_name, day_of_week, is_weekend, is_month_end |
    | Key challenge | Must cover the full date range across both offline and streaming lanes; backfill and live dates may differ |

    ---

    **dim_product** -- static lookup from loan_applications

    | Step | Logic |
    |---|---|
    | Source | Distinct `(contract_type, purpose)` pairs in loan_applications |
    | Columns | product_key (surrogate), contract_type (Cash loans / Revolving loans), purpose, term_bracket, channel |
    | Refresh | Full-refresh on backfill; append-only during live (new product types are rare) |

    ---

    **dim_region** -- static lookup from clients

    | Step | Logic |
    |---|---|
    | Source | Distinct `(region_rating, region_population_relative)` from clients |
    | Columns | region_key (surrogate), region_code, region_rating, region_population_relative |
    | Refresh | Stable region set; full-refresh on backfill only |

    ---

    **dim_merchant** -- incremental from card_authorizations

    | Step | Logic |
    |---|---|
    | Source | Distinct `(merchant_name, merchant_category)` from card_authorizations |
    | Columns | merchant_key (surrogate), merchant_name, merchant_category |
    | Refresh | Incremental: new merchants discovered during live emission; upsert on natural key |

    ---

    **dim_bureau_status** -- static lookup from bureau

    | Step | Logic |
    |---|---|
    | Source | Distinct `STATUS` values from bureau (small set: C, 0, 1, 2, 3, 4, 5, X) |
    | Columns | status_key (surrogate), status_code, status_label |
    | Refresh | One-time static mapping, never changes |

    ---

    **dim_application_outcome** -- static lookup from decisions + master

    | Step | Logic |
    |---|---|
    | Source | Distinct `(decision_status, lifecycle_state)` from application_decisions JOIN application_master |
    | Columns | outcome_key (surrogate), decision_status (Approved / Declined), lifecycle_state (Active / Closed / Defaulted / WrittenOff) |
    | Refresh | Full-refresh on backfill; append-only during live |
    '''),
        mo.md(r'''
    ---

    **fact_application** -- central fact joining 3 entities

    | Step | Logic |
    |---|---|
    | Source | `loan_applications` LEFT JOIN `application_decisions` ON application_id LEFT JOIN `application_master` ON application_id |
    | Keys | client_key -> dim_client, product_key -> dim_product, region_key -> dim_region, outcome_key -> dim_application_outcome, date_submitted_key -> dim_date |
    | Measures | amount_credit, amount_annuity, amount_goods_price, interest_rate, loan_term, sk_dpd_max, label_default, drift_phase, drift_factor |
    | Dedup | application_id is unique in application_master (gold); late-arriving decisions use upsert |
    | Window | One row per application; label_default only available after 12 instalments (window = months 1-12) |

    ---

    **fact_payment** -- per-instalment payments with dedup

    | Step | Logic |
    |---|---|
    | Source | `payments` |
    | Keys | application_id -> fact_application, client_key -> dim_client, date_payment_key -> dim_date |
    | Measures | amount, principal_split, interest_split, days_late, instalment_number |
    | Dedup | Content-hash dedup: group by (payment_id, content_hash), take first row. ~2% duplicate rate injected by generator. |
    | Out-of-order | Payments from the same loan can arrive out of chronological order; join on payment_date not ingestion time |

    ---

    **fact_card_auth** -- streaming auths with late-arrival handling

    | Step | Logic |
    |---|---|
    | Source | `card_authorizations` |
    | Keys | client_key -> dim_client, merchant_key -> dim_merchant, date_auth_key -> dim_date (from event_timestamp) |
    | Measures | auth_amount, approval_flag, decline_reason |
    | Dedup | Content-hash dedup: ~1.5% duplicate replay rate. Same event_id + content_hash, discard after first. |
    | Late arrival | Date key derived from `event_timestamp` (not ingestion_time). Events can arrive 5-45 min late; watermark handling in Flink. |
    | Burst traffic | Month-end spikes produce distinct auth_ids at generation time (no dedup collision) |

    ---

    **fact_pos_balance_month** -- monthly loan snapshot

    | Step | Logic |
    |---|---|
    | Source | `pos_balances` |
    | Keys | application_id -> fact_application, client_key -> dim_client, month_key -> dim_date |
    | Measures | sk_dpd, instalments_remaining, default_flag |
    | Dedup | One row per (application_id, month) -- idempotent overwrite on full-month re-generation |

    ---

    **fact_card_balance_month** -- monthly card snapshot

    | Step | Logic |
    |---|---|
    | Source | `card_balances` |
    | Keys | client_key -> dim_client, month_key -> dim_date |
    | Measures | balance, credit_limit, drawings_amount, sk_dpd |
    | Dedup | One row per (card_id, month) -- idempotent overwrite |

    ---

    **fact_bureau_account_month** -- bureau with schema-evolution tolerance

    | Step | Logic |
    |---|---|
    | Source | `bureau` |
    | Keys | client_key -> dim_client, status_key -> dim_bureau_status, month_key -> dim_date |
    | Measures | credit_type, credit_amount, amount_overdue, days_overdue, schema_version |
    | Schema evolution | Three versions coexist: v1 (no debt_ratio, null), v2 (adds debt_ratio), v3 (breaking rename + per-row schema_version). Reader coalesces: `COALESCE(debt_ratio_v3, debt_ratio_v2, NULL) AS debt_ratio` |
    | Dedup | One row per (bureau_id, month) -- idempotent |
    '''),
        mo.callout(
            mo.md(r'''
    **DP2 processing model.** All transformations are designed for **incremental, idempotent**
    execution on Spark Structured Streaming (online lane) or batch Spark (offline lane).
    The generator's indexed deterministic RNG guarantees that re-running a month from
    the seed produces byte-identical output -- so the Silver layer can safely
    overwrite any month partition without coordination.

    | Concern | Handling |
    |---|---|
    | Late-arriving decisions | Upsert into fact_application on application_id |
    | Duplicate payments | content-hash dedup before insert |
    | Late card auths | Watermark by event_timestamp, not ingestion_time |
    | Schema evolution | Coalesce per schema_version, union across versions |
    | SCD type-2 | Compare snapshot deltas; close + insert on change |
    '''),
            kind="info"
        )
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Data Problems (Rubric Coverage)

    ### Offline (batch) problems

    | Problem | How injected | Entities affected |
    |---|---|---|
    | **Skew** | Region concentration (top_region_ratio), heavy-tailed income (Pareto/log-normal), merchant/purpose concentration | clients, card_authorizations, loan_applications |
    | **Schema evolution** | Bureau v1 (no debt_ratio) -> v2 (adds debt_ratio, non-breaking) -> v3 (per-row schema_version, breaking rename) | bureau |
    | **Exact duplicate** | Per-month Bernoulli: payments 2%, applications 0.5%; byte-identical clones (all invariants preserved), content-hash for dedup | payments, loan_applications |
    | **High cardinality** | bureau_id, auth_id, payment_id are effectively unique per row (unique/total $\approx$ 1.0) | bureau, card_authorizations, payments |

    ### Streaming problems

    | Problem | How injected | Entities affected |
    |---|---|---|
    | **Late arrival** | Event timestamp shifted into the past (5-45 min), ingestion=now; `late` flag + delay_seconds recorded | card_authorizations, application_decisions, payments |
    | **Duplicate replay** | Topic-level re-emit after postgres insert: original event timestamps, new ingestion timestamps | card_authorizations, loan_applications |
    | **Burst traffic** | Periodic Nx volume multiplier (e.g. month-end closings, payroll days); pure tick rule with distinct IDs | loan_applications, card_authorizations |

    ### Drift scenario

    A deterministic controller shifts the PD logit over calendar time:

    $$
    \text{drift\_factor}(\text{date}) =
    \begin{cases}
    1.0 & (\text{baseline}) \\
    1.0 \to 2.0 & (\text{ramp over 90 days}) \\
    2.0 & (\text{post\_drift, sustained})
    \end{cases}
    $$

    $$
    P_{\text{eff}}(f_6, t) = \operatorname{logit}^{-1}\bigl(\operatorname{logit}(P_{\text{base}}(f_6)) \times \text{factor}(t)\bigr)
    $$

    Every loan carries drift metadata (`drift_enabled`, `drift_scenario`,
    `drift_phase`, `drift_factor`). The shift preserves ranking -- default
    propensity order stays the same, but default rates increase.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r'''
    ## Source Systems -> Generator -> Sinks

    The generator simulates the bank's source systems: a Loan Origination System (LOS),
    Loan Servicing System (LMS), Card Processor, Credit Bureau, and Scoring APIs.
    Unlike the old model of writing star-schema facts directly, the generator writes to
    **source OLTP tables**, and the downstream pipeline (Debezium CDC -> Bronze -> Silver -> Gold)
    mirrors real production data engineering.
    ''')

    mo.mermaid('''graph TB
        subgraph SOURCE["Simulated Source Systems"]
            LOS["LOS<br/>loan_applications<br/>application_decisions"]
            LMS["LMS<br/>payments<br/>pos_balances"]
            CARD["Card Processor<br/>card_authorizations<br/>card_balances"]
            BUR["Credit Bureau<br/>bureau"]
            SCORE["Scoring APIs<br/>EXT_SOURCE 1/2/3"]
        end

        subgraph OLTP["Postgres Source (lending_db)"]
            LA["loan_applications"]
            AD["application_decisions"]
            PM["payments"]
            CA["card_authorizations"]
            CL["clients"]
            AM["application_master (gold)"]
        end

        subgraph CDC["Debezium CDC -> Redpanda"]
            RP["lending.public.loan_applications<br/>lending.public.application_decisions<br/>lending.public.payments<br/>lending.public.card_authorizations<br/>lending.public.clients"]
        end

        subgraph S3S["SeaweedFS S3 (batch)"]
            S3C["clients/*.parquet"]
            S3D["application_decisions/*.parquet"]
            S3P["payments/*.parquet"]
            S3B["pos_balances/*.parquet"]
            S3CB["card_balances/*.parquet"]
            S3BU["bureau/*.parquet"]
            S3AM["application_master/*.parquet"]
        end

        subgraph ARCHIVE["Local Archive<br/>outputs/generated/<run>"]
            AR_ALL["All 9 entities<br/>arrow/parquet"]
        end

        LOS --> CL
        LOS --> LA
        LOS --> AD
        LMS --> PM
        LMS --> AM
        CARD --> CA
        CARD --> CL
        BUR --> CL
        SCORE --> CL

        CL -->|monthly| S3C
        LA -->|never S3| ARCHIVE
        CA -->|never S3| ARCHIVE
        AD -->|nightly| S3D
        PM -->|nightly| S3P
        AM -->|monthly| S3AM
        AM -->|gold| AM

        CL --> OLTP
        LA --> OLTP
        AD --> OLTP
        PM --> OLTP
        CA --> OLTP
        AM --> OLTP

        OLTP --> CDC
        CDC --> RP
    ''')
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md(r'''
    ## Dataset Characteristics

    ### Volume targets

    The generator targets >= 100 GB across all tables combined. Key characteristics:

    | Attribute | Detail |
    |---|---|
    | Customers | Configurable (10k-500k), ~8% default rate |
    | Currency | VND (Vietnamese market) |
    | Language | VND amounts, Vietnamese merchant / purpose categories |
    | Loan term | 6-60 months instalment loans + revolving credit cards |
    | Default | 3 consecutive delinquent months within first 12 instalments |

    ### Required data problems across both lanes

    | Problem | Lane | Entity | Detection |
    |---|---|---|---|
    | Skew | offline | clients, auths, apps | top-region ratio, Gini on income |
    | Schema evolution | offline | bureau | null_count per column, schema_version histogram |
    | Duplicate (exact) | offline | payments, apps | content-hash duplicates |
    | Late arrival | streaming | auths, decisions | event_time vs ingestion_time |
    | Duplicate (replay) | streaming | auths, apps | same content_hash, new ingestion_time |
    | Burst traffic | streaming | apps, auths | volume ratio per tick |
    | Data drift | both | loans | PSI of PD-score vs baseline window |
    | Out-of-order | streaming | auths | same card, descending event_time |
    '''),
        mo.md('''
    ### Granularity note

    The old HC star-schema design (this notebook's previous content) decomposed
    CSV sources into dims + facts. The generator takes a **different approach**:
    it produces flat OLTP-style entities that mirror real source systems.
    The star-schema is a downstream dbt / Spark transformation -- not a
    generation concern.
    ''')
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Default Label Derivation

    The default label is derived from the RunHazard model's monthly delinquency
    path, not from a single TARGET column.

    ### How

    ```
    For each loan:
      1. Draw i.i.d. per-month delinquency indicator u ~ Uniform(0,1)
      2. Default if g_n^{-1}(u) (hazard inversion) triggers delinquency
      3. Month-by-month DPD path accumulates:
         - Delinquent month -> DPD counter increments
         - Non-delinquent -> DPD resets to 0
      4. After 12 instalments: label = 1 if DPD counter $\ge$ 3 consecutive
      5. Training uses only the 'final' window row per loan
    ```

    ### Key properties

    | Property | Value |
    |---|---|
    | Base default rate | 8.07% (calibrated to Home Credit) |
    | AUC ($f_6 \to$ default) | $\sim$0.745 |
    | EXT_SOURCE protection | AUC < 0.45 (lower default risk with higher score) |
    | Logit shift | brentq so approval-weighted mean matches target |
    | Ranking preservation | Shift is multiplicative on logit scale |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Drift Scenario: Economic Downturn

    A deterministic controller shifts the default-propensity logit over calendar
    time, causing **covariate shift** in features derived from PD-related columns
    and a genuine **label shift**.

    ### Controller

    ```
    drift_mode: gradual (ramp) or abrupt (step)

    drift_start_date: 2022-06-01
    baseline:           2020-01-01 to 2022-05-31  (factor = 1.0)
    ramp:               2022-06-01 to 2022-08-29  (factor = 1.0 -> 2.0)
    post_drift:         2022-08-30 onward          (factor = 2.0)
    ```

    ### Impact

    | Metric | Baseline | Post-drift |
    |---|---|---|
    | Approval-weighted default rate | ~8% | ~16% (2x factor) |
    | PD score distribution | Calibrated | Shifted (PSI measurable) |
    | Feature distributions | Stable | DPD, bureau STATUS, payment behavior drift emergently |
    | Ranking (AUC) | ~0.745 | Preserved (ordering unchanged) |

    Every loan carries drift metadata fields for downstream drift-detection pipelines:
    `drift_enabled`, `drift_scenario`, `drift_phase`, `drift_factor`.
    """)
    return


@app.cell
def _(mo):
    mo.md(r'''
    ## Generator Implementation (source layout)

    The generator lives at `simulator/src/simulator/` (standard src-layout package).

    | Module | Responsibility |
    |---|---|
    | `core.py` | `Simulator` (stateful iterator over months), `MonthBatch` |
    | `copula.py` | `CopulaEngine` (t-copula sampling, factor draws) |
    | `config.py` | `SimConfig`, `CopulaConfig`, `SinkConfig` (dataclasses + env overrides) |
    | `entities.py` | Per-entity generators (clients, applications, auths, payments, bureau, cards, balances) |
    | `state.py` | `AccountLedger`, `RunHazard` (loan lifecycle, DPD path, hazard inversion) |
    | `label.py` | Windowed default label derivation (3 delinquent months in first 12) |
    | `sinks.py` | `ParquetSink`, `S3Sink`, `PostgresSink` + `SinkManager` |
    | `livestage.py` | `LiveStage` -- SimClock dispatcher with 3-rhythm + 2-lane routing |
    | `paths/` | `OfflinePath`, `OnlinePath` -- routing constants and lane logic |
    | `calibrate.py` | Copula + structure calibration from real Home Credit data |
    | `simclock.py` | `SimClock` -- restart-proof date predicates |
    | `problem_injector.py` | Lane-scoped problem injection (offline: skew/evolution/dup; streaming: late/replay/burst) |
    ''')

    mo.md(r'''
    ### Configuration precedence

    ```
    defaults < calibration artifacts < YAML scenario < env (SIM_*) < CLI flags
    ```

    ```yaml
    # configs/simulator/challenges_v1.yaml
    seed: 42
    run_id: challenges_v1
    time:
      start_date: 2020-01-01
      bureau_drop_lag_days: 21
      payment_file_hour_utc: 2
    entities:
      n_customers: 50000
      target_default_rate: 0.08
    copula:
      nu: 30.0
    drift:
      enabled: true
      scenario: economic_downturn
      drift_start_date: 2022-06-01
      default_propensity_multiplier: 2.0
    sinks:
      enable_s3: true
      enable_postgres: true
    ```
    ''')
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md(r'''
    ## Summary: Old Design vs New Reality

    | Aspect | Old design (HC star-schema) | New generator design |
    |---|---|---|
    | **Data source** | Kaggle CSV files (application_train.csv + 6 child CSVs) | t-copula with 7 latent factors -> 9 entity generators |
    | **Entities** | 1 client table + 6 fact tables (star-schema) | 9 flat OLTP entities (clients, apps, auths, payments, decisions, master, pos, card, bureau) |
    | **Cadence** | Everything monthly | 3 rhythms: daily hot / nightly / monthly close |
    | **Lanes** | Single batch output | 2 lanes: offline (parquet+S3) + online (postgres->CDC->Redpanda) |
    | **Label** | TARGET column (given) | Windowed default: 3 delinquent months in first 12 instalments |
    | **Drift** | None | Deterministic PD-logit controller, per-row metadata |
    | **Schema evolution** | None | Bureau v1/v2/v3 with schema_version |
    | **Duplicates** | None | Payments 2%, applications 0.5% (exact byte-identical clones) |
    | **Streaming problems** | None | Late arrival, duplicate replay, burst traffic |
    | **Default rate** | ~8% (fixed) | Configurable via logit shift, preserved through drift |
    | **Reproducibility** | Read-once CSV | Indexed deterministic RNG: f(seed, entity, month, day, slot) |
    '''),
        mo.md(r'''
    ### References

    | Doc | Description |
    |---|---|
    | `docs/characteristics.md` | Full dataset characteristics: entities, cadence, lanes, problems |
    | `docs/statistics.md` | Statistical method: t-copula, hazard inversion, calibration |
    | `docs/adr/0001-entity-cadence-split.md` | ADR: split into 3 cadences with explicit lane routing |
    | `docs/datgentor_impl_note_20261026.md` | Implementation note: packaging, sinks, COLD-first routing |
    | `docs/datgentor_improv_plan_20261028.md` | Improvement plan: Phase 0-7, config, problems, drift |
    | `docs/datgentor_challenges_issues_20261029.md` | Implementation issues: PK collisions, lane-scoped problems |
    | `AGENTS.md` | Project overview, k3d infrastructure, commands |
    ''')
    ])
    return


if __name__ == "__main__":
    app.run()
