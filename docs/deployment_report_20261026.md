# Deployment Report — October 26, 2026

**Project:** Credit Risk MLOps
**Repo:** `credit-risk-mlops` (fresh start, moved from `default_risk_mlops`)
**Cluster:** kind (`risk-cluster`), Kubernetes v1.34.0

---

## 1. Current state

### Running services (kind cluster)

| Service | Status | Pod IP | Host port | Notes |
|---|---|---|---|---|
| **postgres-source** | Running | 10.244.0.4 | 5432 | 20 customers seeded, full OLTP schema |
| **postgres-platform** | Running | 10.244.0.5 | 5433 | mlflow/feast/metastore DBs created |
| **seaweedfs** | Running | 10.244.0.7 | 8333 (S3) | S3 creds via env vars (`seaweedfs/seaweedfs123`) |
| **redpanda** | Running | 10.244.0.6 | 9092 | CLUSTER HEALTHY, no topics yet |
| **polaris** | Running | 10.244.0.9 | 8181 | Iceberg REST catalog, STATUS UP |
| **mlflow** | Running | 10.244.0.10 | 5000 | UI responding 200, S3 artifact store, Postgres backend |

### Repo layout

```
configs/                    ← platform-agnostic configs
├── postgres/init/          ← source OLTP schema + seed data
└── seaweedfs/s3.json       ← S3 identity config (legacy reference)

infra/                      ← how things are deployed
├── kind.yaml               ← cluster config (port forwards for all services)
├── compose/
│   └── docker-compose.yml  ← old compose stack (reference/fallback)
└── k8s/<service>/
    └── manifest.yaml       ← K8s Deployment + Service (+ ConfigMap where needed)

ops/                        ← how you run it
├── k8s/
│   ├── apply-all.sh        ← apply manifests in dependency order
│   └── kind-up.sh          ← create cluster + apply all
└── postgres/
    └── run-source-postgres.sh  ← standalone source PG (old Docker workflow)

notebooks/                  ← ML work
├── explore.py              ← EDA (marimo)
└── fit.py                  ← LightGBM baseline (marimo)
```

---

## 2. Build log (what was created this session)

### Cluster bootstrap
- Created `ops/kind.yaml` → moved to `infra/kind.yaml`
- apiVersion: `kind.x-k8s.io/v1alpha4` (not `v1alpha2`)
- Port mappings for all 6 services + tools (pgAdmin, Redpanda Console)
- Cluster was recreated once due to accidental node kill + broken CoreDNS

### Manifest gotchas resolved

| Issue | Fix |
|---|---|
| `apiVersion` in kind config | Changed from `v1alpha2` to `kind.x-k8s.io/v1alpha4` |
| SeaweedFS `mini` not found | K8s `command` overrides both ENTRYPOINT and CMD. Fixed by using `/entrypoint.sh mini ...` as command |
| Custom image not found | `imagePullPolicy: IfNotPresent` — tells K8s to use local containerd copy instead of pulling from registry |
| MLflow `psycopg` missing | Image built with `psycopg2-binary`. MLflow URI must use `postgresql+psycopg2://` scheme |
| CoreDNS Unauthorized | Fresh cluster creation fixed it. Old cluster had stale CoreDNS auth tokens |
| DNS resolution from pods | Fresh cluster has working CoreDNS. Hostnames like `postgres-platform` resolve correctly |

### MLflow image
- Custom image `risk-mlflow:latest` built from `infra/k8s/mlflow/Dockerfile`
- Contains: Python 3.13-slim + mlflow 2.19+ + psycopg2-binary + boto3
- Loaded into kind's containerd via `kind load docker-image`
- Accessed by manifest via `imagePullPolicy: IfNotPresent`

---

## 3. How to rebuild from scratch

```bash
kind delete cluster --name risk-cluster                           # tear down
bash ops/k8s/kind-up.sh                                           # create + deploy
```

For custom image rebuild:
```bash
docker build -t risk-mlflow:latest -f infra/k8s/mlflow/Dockerfile .
kind load docker-image risk-mlflow:latest --name risk-cluster
```

---

## 4. Open items

| Item | Status | Priority |
|---|---|---|
| `simulator/` | Empty — intended replacement for legacy `data_generator/` | Not started |
| S3 bucket creation | SeaweedFS S3 works but needs `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` env + bucket creation via Admin UI or aws CLI | Low — works for dev |
| Debezium/CDC | Discussed in prior sessions, not implemented | Medium |
| K8s → Helm migration | Under consideration. 6 services, mixed Bitnami/custom chart approach | Exploring |
| MLflow experiment logging | Need to run notebooks with `mlflow.set_tracking_uri("http://localhost:5000")` | Low — infra is ready |
| GCP/Terraform | Not started | Future |
| Notebook dependency mgmt | `pyproject.toml` has minimal deps. `uv add` needed for mlflow, polars, lightgbm, etc. | Medium |