# Coursework Gap Analysis - 2026-10-28

Source rubric: `docs/rubric/rubic/EDAI K11.xlsx` (3 sheets, 100 pts each).
This repo targets the **edai-1 (50%)** data-platform sheet. Status reflects the
repo today. See `docs/data_problem_statement_20261028.md` for the problem-injection
principles extracted from the RecSys reference.

Legend: [x] done | [~] partial | [ ] missing

## edai-1 (50%) - 100 pts

| Rubric item | Pts | Status | Work items |
|---|---|---|---|
| **Engineering Fundamentals** | | | |
| Docker & Docker Compose | 2 | [x] | `simulator/Dockerfile`, compose ref in `infra/compose/` |
| Optimize Dockerfile (multistage) | 3 | [x]/[~] | uv multi-stage exists; document before/after size in `docs/` |
| **Implement Data Generator** | | | |
| Simulate offline problems (>=100GB) | - | [~] | Current sim is POC-scale; target >=100GB volume |
| - skew | 2 | [~] | t-copula skews exist but no characteristics report (skew ratios, cardinality ratios, volume) |
| - schema evolution | 2 | [ ] | Add versioned schema (v1 omits new fields; breaking change; per-row `schema_version`) |
| - another offline problem (e.g. 2% duplicate) | 2 | [ ] | Inject exact duplicates (content-hash dedup detectable) |
| - store for Bronze ingest (MinIO/PostgreSQL) | 2 | [x] | SeaweedFS S3 + postgres-source + Debezium CDC -> Redpanda |
| Simulate online problems | | | |
| - late arrivals | 2 | [ ] | Late event_timestamp vs ingestion_ts on CDC/stream lane |
| - another streaming problem (e.g. 1.5% dup) | 2 | [ ] | Duplicate replay bundles (original event time, new ingestion time) |
| **Processing Jobs** | | | |
| Spark: baseline + Spark UI docs | 2 | [ ] | |
| Spark: handle skew + explanation | 3 | [ ] | |
| Spark: handle schema evolution | 3 | [ ] | |
| Spark: other offline problem | 3 | [ ] | |
| Spark jobs in pipelines (Airflow) | 2 | [ ] | |
| Flink: baseline + Flink UI docs | 2 | [ ] | |
| Flink: handle late arrival | 3 | [ ] | |
| Flink: other streaming problem | 3 | [ ] | |
| Flink window processing demo | 2 | [ ] | |
| **Data Storage** | | | |
| Lakehouse optimization (compaction/z-order/partitioning) | 3 | [ ] | Iceberg (polaris) available for compaction/z-order |
| Datawarehouse optimization (indexing) | 3 | [ ] | |
| **Data Pipeline Orchestration (Airflow)** | | | |
| DP1 raw->bronze: ingest + validate | 4 | [~] | Sink wiring exists; no Airflow, no validate stage |
| DP2 bronze->silver/gold: ingest + validate | 4 | [ ] | gold `application_master` exists but no pipeline |
| DP3 offline feature table: ingest + validate | 4 | [ ] | No `feat_` layer |
| **Data Governance (DataHub)** | | | |
| DP1/DP2/DP3 lineage + data validation | 6 | [ ] | |
| **Schema Design** | | | |
| Visualize tables per zone | 2 | [ ] | |
| Dim table with SCD2 | 2 | [ ] | |
| Feature tables `feat_` (event_timestamp, created) | 2 | [ ] | |
| Dim<->fact relationships | 2 | [ ] | |
| Naming convention (`dim_/fact_/obt_/feat_/raw_/stg_`) | 2 | [ ] | |
| **Novel ideas** (2 x 10) | 20 | [ ] | Biggest single chunk - reserve time |

**edai-1 total: ~17 pts in hand, ~83 pts of work.**

## Quick reference - other sheets

- **edai-2 (50%)**: LLM inference platform, agentgateway (MCP/A2A), RAG+Feast, Kubeflow + MLflow, KServe, agent w/ tools, testing (coverage/mutation/property-based/load), CI/CD, NGINX gateway, IaC, observability, A/B, Vault, 2 novel ideas. *Not the current focus.*
- **Final coursework (hidden sheet, 100 pts)**: builds on edai-1/2 - real-time drift detection API, KEDA autoscale, data drift + label table in generator, distributed training, incremental data versioning, ML drift monitoring (PushGateway + retrain trigger), A/B inference split, service mesh auth, low-level ML design (5 classes), 2 novel ideas. *The drift work in the generator directly feeds this sheet later.*

## Recommended order (edai-1)

1. **Data generator hardening** (13 pts + 100GB) - skew report, schema evolution, exact duplicate, streaming late/replay. *(from `data_problem_statement_20261028.md` principles)*
2. **Schema design + naming + SCD2** (10 pts) - cheap, unblocks Bronze/Silver/Gold.
3. **Airflow DP1->DP3 with validate stages** (12 pts).
4. **Spark + Flink jobs with baseline->optimized docs** (24 pts) - the heavy lift.
5. **Storage optimization** (6 pts) - Iceberg compaction/z-order; indexing.
6. **DataHub governance** (6 pts).
7. **Novel ideas** (20 pts) - continuously look for tools/techniques beyond EDAI.