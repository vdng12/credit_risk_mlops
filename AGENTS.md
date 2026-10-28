# AGENTS.md

## Project

**Credit Risk MLOps** - an MLOps pipeline for a banking/lending default-risk
system. Three layers: Data -> ML -> Agent.

Python 3.13+, managed with `uv` (workspace monorepo). Notebooks use marimo.

Current state: LightGBM POC baseline in `notebooks/fit.py` (5-fold CV + held-out
test), a t-copula synthetic data generator in `simulator/`, and a K8s k3d
cluster with 7 services for the ML platform stack.

---

## Data generator (`simulator/`)

uv workspace member at `simulator/` (src-layout package, `src/simulator/`).
Single workspace lock at the repo root. Produces synthetic lending data across
8 entity types through a **t-copula with 7 latent factors**.

### Architecture

```ascii
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

Each client draws the latent vector $\mathbf{f} \in \mathbb{R}^7$ once and
keeps it for life (subscripts $f_0,\dots,f_5$ = application-feature factors;
$f_6$ = default propensity).

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

The simulator releases everything COLD-first - no direct Kafka writes. Kafka is
streamed downstream by the Debezium CDC connector on postgres-source, which
re-publishes writes to the OLTP tables (this is the real DE ingest wiring):

```
simulator (t-copula) -> MonthBatch -> local parquet write-through (outputs/generated/)
                         |
                         +-- postgres-source lending_db (OLTP - Debezium CDC -> Redpanda)
                         |    clients               -> lending_db.clients        -> CDC topic lending.public.clients
                         |    loan_applications     -> lending_db.loan_applications -> CDC topic lending.public.loan_applications
                         |    card_authorizations   -> lending_db.card_authorizations -> CDC topic lending.public.card_authorizations
                         |    application_decisions -> lending_db.application_decisions -> CDC topic lending.public.application_decisions
                         |    payments              -> lending_db.payments            -> CDC topic lending.public.payments
                         |    application_master    -> lending_db.application_master (gold layer, no CDC)
                         |
                         +-- SeaweedFS S3 bucket (parquet snapshots)
                              clients               -> s3://risk-data/<run>/clients_<yyyy-mm>.parquet
                              application_decisions -> s3://risk-data/<run>/application_decisions_<yyyy-mm>.parquet
                              pos_balances          -> s3://risk-data/<run>/pos_balances_<yyyy-mm>.parquet
                              card_balances         -> s3://risk-data/<run>/card_balances_<yyyy-mm>.parquet
                              bureau                -> s3://risk-data/<run>/bureau_<yyyy-mm>.parquet
```

Sinks and endpoints are configured via `SinkConfig` (env `SIM_*` overrides); all
entities also land in the local file sink as a write-through archive.

The generator's local write-through archive defaults to `outputs/generated/`
(gitignored) - the production lanes are the S3/Postgres sinks above
(Kafka/Redpanda topics are CDC-derived from Postgres, not generator sinks).
The legacy `output/` directory was workshop validation only - not the
production path.

### Usage

```bash
# From workspace root (single venv + lock for the whole repo)
uv run simulator --n-customers 100000 --months 6    # console script
uv run python -m simulator --n-customers 100000 --months 6   # module form

# Test
uv run pytest tests/test_statistics.py -q
```

### Statistical method (docs/statistics.md)

Full documentation at `docs/statistics.md`. Key design points:

- **t-copula** separates marginals from dependence; tail dependence controlled by $\nu$ ($\nu$ small = stronger joint extremes)
- **Copula parameters** $(\Sigma, \nu)$ calibrated from Home Credit via Kendall's $\tau$ inversion + profile likelihood
- **Default model**: exact hazard inversion `RunHazard` with per-loan i.i.d. monthly delinquency, matched to HC's 8.07% base rate and AUC 0.745
- **Default-rate knob**: logit shift solved by `brentq` so approval-weighted mean matches target - preserves ranking, deterministic
- **Label derivation**: $\mathrm{default} = 1$ if 3 consecutive delinquent months within first 12 instalments; training uses only `final` window rows

### Calibration status

Current shipped values are **provisional** (need re-calibration on real Home Credit data).
The old fit hit $\nu = 30$ (search bound, meaning Gaussian preferred). Two factor signs
(creditworthiness, stability) were flipped and corrected in `CopulaConfig.corr_matrix`.
Signs of factors 1,2,3,5 are unverified until anchored calibration is re-run.

---

## Infrastructure (`k3d` cluster)

All services run in a local `k3d` Kubernetes cluster (`risk-cluster`).

| Service | Pod IP* | Host port | Notes |
|---|---|---|---|
| **postgres-source** | 10.42.0.9 | 5432 | OLTP lending DB; init SQL is schema-only (0 rows), data comes from the simulator |
| **postgres-platform** | 10.42.0.4 | 5433 | mlflow/feast/metastore DBs |
| **seaweedfs** | 10.42.0.7 | 8333 (S3) | S3 object store, creds via env vars |
| **redpanda** | 10.42.0.12 | 9092 | Kafka broker (CDC topics: lending.public.*) |
| **polaris** | 10.42.0.10 | 8181/8182 | Iceberg REST catalog, UP |
| **mlflow** | 10.42.0.8 | 5000 | MLflow tracking, S3 artifact store (custom image risk-mlflow) |
| **debezium** | 10.42.0.13 | - | Connect API on :8083, pod-internal only (no host `-p` mapping) |

*Pod IPs change on every cluster recreate - the stable contract is the
NodePort services (host port == nodePort), see `ops/k8s/k3d-up.sh`.

### Commands

```bash
bash ops/k8s/k3d-up.sh                    # create cluster + deploy all (wires NodePort = host port)
bash ops/k8s/apply-all.sh                  # re-deploy all manifests
bash ops/debezium/register-connector.sh    # (re)register the CDC connector after a recreate
kubectl get pods -o wide                   # check status
k3d cluster delete --name risk-cluster    # tear down
```

### Custom image (MLflow)

```bash
docker build -t risk-mlflow:latest -f infra/k8s/mlflow/Dockerfile .
k3d image import risk-mlflow:latest --name risk-cluster
# custom image is not in the registry - re-import after every cluster recreate
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
+- pyproject.toml              # Workspace root
+- uv.lock
+- configs/                    # Platform-agnostic configs
|  `- seaweedfs/s3.json       # S3 identities (legacy)
+- infra/                      # Deployment definitions
|  +- k3d.yaml               # Cluster config
|  +- postgres/init/          # Source OLTP schema (embedded in postgres-source manifest)
|  +- compose/                # Old docker-compose (reference)
|  `- k8s/<service>/         # K8s manifests per service
|     `- manifest.yaml
+- ops/                        # Operational scripts
|  +- k8s/                    # apply-all.sh, k3d-up.sh
|  +- debezium/               # register-connector.sh (CDC connector registration)
|  `- postgres/               # run-source-postgres.sh
+- simulator/                  # Workspace member (uv; src/simulator layout)
|  +- pyproject.toml          # Deps (numpy, polars, scipy, sklearn); [project.scripts] simulator
|  +- Dockerfile              # uv multi-stage image build
|  `- src/simulator/          # Package source
|     +- __init__.py
|     +- core.py             # Simulator (stateful iterator)
|     +- config.py           # SimConfig, CopulaConfig, SinkConfig
|     +- copula.py           # CopulaEngine (t-copula sampling)
|     +- calibrate.py        # Fit copula from real data
|     +- entities.py         # Per-entity generators
|     +- state.py            # AccountLedger, RunHazard
|     +- label.py            # Default label derivation
|     +- sinks.py            # Parquet/S3/Postgres sinks
|     `- ...
+- tests/
|  `- test_statistics.py      # Regression tests for simulator
+- scripts/
|  `- eval_synthetic.py       # LightGBM eval on synthetic vs real
+- notebooks/
|  +- explore.py              # marimo EDA
|  `- fit.py                  # marimo LightGBM baseline
+- docs/
   +- statistics.md           # Full statistical method doc
   +- infra_ops_20261025.md   # Infra deployment plan
   +- deployment_report_20261026.md  # Cluster state report
   +- data_generator_impl_note_20261026.md  # Generator impl notes
   `- fixlog_20261028.md      # Session log: restructure, sinks, CDC-only, k3d recovery
```

## Conventions

- Python 3.13+
- `uv` for dependency management (workspace monorepo)
- marimo for notebooks
- JVM-free stack (Redpanda over Kafka, no Java)
- Infrastructure is k3d (local K8s) - GKE target for prod
- Two Postgres roles:
  - `postgres-source` (host :5432) - OLTP source system
  - `postgres-platform` (host :5433) - MLflow / Feast / metastore
- Data flow: simulator (t-copula) -> MonthBatch -> local parquet write-through + Postgres OLTP (lending_db) + SeaweedFS S3; Kafka/Redpanda streams are Debezium CDC re-publications of the Postgres OLTP writes (no direct Kafka writes from the generator)
  - `outputs/generated/` is the gitignored write-through archive; legacy `output/` was workshop validation only