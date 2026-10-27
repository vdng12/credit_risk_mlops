# AGENTS.md

## Project

**Credit Risk MLOps** — an MLOps pipeline for a banking/lending default‑risk
system. Three layers: Data → ML → Agent.

Python 3.13+, managed with `uv` (workspace monorepo). Notebooks use marimo.

Current state: LightGBM POC baseline in `notebooks/fit.py` (5-fold CV + held-out
test), a t-copula synthetic data generator in `simulator/`, and a K8s kind
cluster with 6 services for the ML platform stack.

---

## Data generator (`simulator/`)

Standalone uv project at `simulator/` with its own `pyproject.toml` and `uv.lock`.
Produces synthetic lending data across 8 entity types through a **t-copula with
7 latent factors**.

### Architecture

```ascii
t-copula (7 latent factors, N(0,1) margins)
        │
        ▼
per-client factor vector f ∈ R⁷   (drawn once, stable for life)
        ├── client / application features      f₀…f₅  → marginals
        ├── child-table links                  f       → binned / GLM curves
        ├── bureau STATUS                      Markov chain, matrix by f₆ bucket
        └── loan default                       f₆ → PD curve → per-loan hazard
                                                       → DPD path → windowed label
```

### Latent factors

| idx | Factor | "+" means | Source |
|---|---|---|---|
| 0 | creditworthiness | safer | PC1 of EXT_SOURCE_1/2/3 |
| 1 | income_capacity | richer | PC1 of log income, days employed, income type, occupation |
| 2 | loan_demand | bigger loan | PC1 of log credit / annuity / goods price |
| 3 | asset_quality | more assets | PC1 of car/realty flags, car age, housing |
| 4 | stability | older | PC1 of age, children, family, gender, education |
| 5 | risk_appetite | revolving | PC1 of contract type, suite |
| 6 | default_propensity | riskier | out-of-fold GBM PD score (HC TARGET) |

### Output paths

The simulator splits output into two lanes — this is the real DE ingest wiring:

```
simulator (t-copula) ──► MonthBatch
                            │
                            ├──► HOT (stream)
                            │    ├── loan_applications     → Redpanda topic: loan_applications
                            │    └── card_authorizations   → Redpanda topic: card_authorizations
                            │
                            └──► COLD (batch, nightly dump)
                                 ├── clients               → SeaweedFS S3 bucket (parquet)
                                 ├── application_decisions → SeaweedFS S3 + Postgres lending_db
                                 ├── application_master    → Postgres lending_db (gold layer)
                                 ├── pos_balances          → SeaweedFS S3 (parquet)
                                 ├── card_balances         → SeaweedFS S3 (parquet)
                                 ├── bureau                → SeaweedFS S3 (parquet)
                                 └── payments              → Postgres lending_db (relational)
```

The `output/` directory was used only for workshop validation; it is NOT the production path.

### Usage

```bash
# Run standalone
cd simulator && uv run python -m simulator --n-customers 50000 --months 12

# From workspace root
uv run python -m simulator --n-customers 100000 --months 6

# Test
uv run pytest tests/test_statistics.py -q
```

### Statistical method (docs/statistics.md)

Full documentation at `docs/statistics.md`. Key design points:

- **t-copula** separates marginals from dependence; tail dependence controlled by ν (ν small = stronger joint extremes)
- **Copula parameters** (Σ, ν) calibrated from Home Credit via Kendall's τ inversion + profile likelihood
- **Default model**: exact hazard inversion `RunHazard` with per-loan i.i.d. monthly delinquency, matched to HC's 8.07% base rate and AUC 0.745
- **Default-rate knob**: logit shift solved by `brentq` so approval-weighted mean matches target — preserves ranking, deterministic
- **Label derivation**: `default = 1` if 3 consecutive delinquent months within first 12 instalments; training uses only `final` window rows

### Calibration status

Current shipped values are **provisional** (need re-calibration on real Home Credit data).
The old fit hit ν = 30 (search bound, meaning Gaussian preferred). Two factor signs
(creditworthiness, stability) were flipped and corrected in `CopulaConfig.corr_matrix`.
Signs of factors 1,2,3,5 are unverified until anchored calibration is re-run.

---

## Infrastructure (`kind` cluster)

All services run in a local `kind` Kubernetes cluster (`risk-cluster`).

| Service | Pod IP | Host port | Notes |
|---|---|---|---|
| **postgres-source** | 10.244.0.4 | 5432 | OLTP lending DB, 20 customers seeded |
| **postgres-platform** | 10.244.0.5 | 5433 | mlflow/feast/metastore DBs |
| **seaweedfs** | 10.244.0.7 | 8333 (S3) | S3 object store, creds via env vars |
| **redpanda** | 10.244.0.6 | 9092 | Kafka broker, healthy |
| **polaris** | 10.244.0.9 | 8181 | Iceberg REST catalog, UP |
| **mlflow** | 10.244.0.10 | 5000 | MLflow tracking, S3 artifact store |

### Commands

```bash
bash ops/k8s/kind-up.sh                    # create cluster + deploy all
bash ops/k8s/apply-all.sh                  # re-deploy all manifests
kubectl get pods -o wide                   # check status

kind delete cluster --name risk-cluster    # tear down
```

### Custom image (MLflow)

```bash
docker build -t risk-mlflow:latest -f infra/k8s/mlflow/Dockerfile .
kind load docker-image risk-mlflow:latest --name risk-cluster
```

### Port mappings

| Service | Host port | Container port |
|---|---|---|
| Postgres Source | 5432 | 5432 |
| Postgres Platform | 5433 | 5432 |
| SeaweedFS S3 | 8333 | 8333 |
| SeaweedFS Filer UI | 8888 | 8888 |
| SeaweedFS Master | 9333 | 9333 |
| Redpanda | 9092 | 9092 |
| Polaris REST | 8181 | 8181 |
| Polaris Mgmt | 8182 | 8182 |
| MLflow UI | 5000 | 5000 |

---

## Repo structure

```
.
├── pyproject.toml              # Workspace root
├── uv.lock
├── configs/                    # Platform-agnostic configs
│   ├── postgres/init/          # Source OLTP schema + seed
│   └── seaweedfs/s3.json       # S3 identities (legacy)
├── infra/                      # Deployment definitions
│   ├── kind.yaml               # Cluster config
│   ├── compose/                # Old docker-compose (reference)
│   └── k8s/<service>/         # K8s manifests per service
│       └── manifest.yaml
├── ops/                        # Operational scripts
│   ├── k8s/                    # apply-all.sh, kind-up.sh
│   └── postgres/               # run-source-postgres.sh
├── simulator/                  # Standalone uv project
│   ├── pyproject.toml          # Own deps (numpy, polars, scipy, sklearn)
│   ├── uv.lock
│   └── src/                    # Package source
│       ├── __init__.py
│       ├── core.py             # Simulator (stateful iterator)
│       ├── config.py           # SimConfig, CopulaConfig, EntityConfig
│       ├── copula.py           # CopulaEngine (t-copula sampling)
│       ├── calibrate.py        # Fit copula from real data
│       ├── entities.py         # Per-entity generators
│       ├── state.py            # AccountLedger, RunHazard
│       ├── label.py            # Default label derivation
│       ├── sinks.py            # Parquet/S3/Postgres/Kafka sinks
│       └── ...
├── tests/
│   └── test_statistics.py      # Regression tests for simulator
├── scripts/
│   └── eval_synthetic.py       # LightGBM eval on synthetic vs real
├── notebooks/
│   ├── explore.py              # marimo EDA
│   └── fit.py                  # marimo LightGBM baseline
└── docs/
    ├── statistics.md           # Full statistical method doc
    ├── infra_ops_20261025.md   # Infra deployment plan
    ├── deployment_report_20261026.md  # Cluster state report
    └── data_generator_impl_note_20261026.md  # Generator impl notes
```

## Conventions

- Python 3.13+
- `uv` for dependency management (workspace monorepo)
- marimo for notebooks
- JVM-free stack (Redpanda over Kafka, no Java)
- Infrastructure is kind (local K8s) — GKE target for prod
- Two Postgres roles:
  - `postgres-source` (host :5432) — OLTP source system
  - `postgres-platform` (host :5433) — MLflow / Feast / metastore
- Data flow: simulator (t-copula) → MonthBatch → HOT (Redpanda topics) + COLD (SeaweedFS parquet + Postgres tables)
  - `output/` is for workshop validation only, not the production pipeline