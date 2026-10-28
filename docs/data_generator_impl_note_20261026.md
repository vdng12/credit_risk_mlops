# Data Generator Implementation Note - 2026-10-26 (updated 2026-10-28)

**Scope:** the `simulator/` t-copula data generator as a uv workspace member,
packaging, sink routing (COLD-first, CDC-derived Kafka), and validation.
The 2026-10-28 update rewrites the doc for the post-restructure state: the
generator is no longer a standalone uv project, and Kafka is no longer a
direct sink (see sections 1 and 3).

---

## 1. Workspace restructure (current state)

| Item | Before (2026-10-26) | After (2026-10-28) |
|---|---|---|
| Project model | standalone uv project (own `pyproject.toml`, `uv.lock`, `.venv`) | workspace member of the root monorepo, single root `uv.lock` |
| Package layout | `simulator/src/*.py` flat (hatch `force-include` shim) | `simulator/src/simulator/*.py`, standard src layout |
| Build backend | hatchling + `force-include` rebase | `uv_build` (auto-detects the src layout, no shim) |
| Root deps | `[project.optional-dependencies]` dev/eval | `[dependency-groups]` dev/eval; `default-groups = ["dev"]` |
| Entry point | `python -m simulator` only | `[project.scripts] simulator = "simulator.__main__:main"` + module form |
| Runtime | bare `uv run simulator` | also a container image + CronJob |

Key win of `uv_build`: the package installs as `simulator` directly from
`src/simulator/`, so the hatch `force-include = { "src" = "simulator" }`
rebase hack is gone. `uv run simulator ...` and
`uv run python -m simulator ...` are the same CLI.

## 2. Packaging and the engineering image

`simulator/Dockerfile` is a two-layer `uv` build:

1. `uv sync --frozen --no-dev --no-install-workspace` - resolves and caches
   the full dependency closure from the root lockfile;
2. `uv sync --frozen --no-dev --package simulator --all-extras` - installs
   the `simulator` member itself (incl. the `sinks` extra).

`ENTRYPOINT ["simulator"]` uses the console script. Deployment is
`infra/k8s/simulator/manifest.yaml` (a CronJob); root `.dockerignore` keeps
the image lean.

## 3. Sink routing: COLD-first, CDC-derived Kafka

**Principle.** The generator writes everything COLD-first. It never writes
Kafka: the `lending.public.*` topics are produced by the Debezium CDC
connector on the postgres-source OLTP tables (the real DE ingest wiring).
This replaced the old HOT lane (direct Kafka writes from the generator),
which had created parallel orphan topics the DE never consumed.

| destination | entities |
|---|---|
| local parquet write-through, `outputs/generated/<run>/` | all 9 (clients, loan_applications, card_authorizations, application_decisions, payments, application_master, pos_balances, card_balances, bureau) |
| Postgres `lending_db` (OLTP, -> Debezium CDC) | clients, loan_applications, card_authorizations, application_decisions, payments |
| Postgres `lending_db` (gold layer, no CDC) | application_master |
| SeaweedFS S3 `risk-data` | clients, application_decisions, pos_balances, card_balances, bureau |

**Configuration.** `SinkConfig` reads env `SIM_*` (e.g. `SIM_POSTGRES_DSN`,
`SIM_S3_ENDPOINT`, `SIM_S3_BUCKET`, `SIM_OUTPUT_DIR`); per-sink toggles via
`--no-s3` / `--no-postgres`. Sink failures raise loudly - nothing is
silently swallowed (previously S3/Kafka errors were `except: pass`, so
"validation" tested nothing reached the cluster).

**`to_postgres` is schema-tolerant**: it projects each frame onto the target
table's columns. The OLTP schema is the contract - e.g. `loan_applications`
carries only pre-decision columns; outcomes live in `application_decisions`.
Dialect note: SQLAlchemy 2.1 defaults `postgresql://` DSNs to psycopg3, so
DSNs are pinned `postgresql+psycopg2`.

## 4. Validation

### Unit tests (11/11)

```bash
uv run pytest tests/test_statistics.py -q      # 11 passed
```

| Test | Verifies |
|---|---|
| `test_t_copula_recovers_nu_and_sigma` | fitting recovers known $\nu$, $\Sigma$ within tolerance |
| `test_gaussian_data_is_flagged` | Gaussian data flagged, not tail-dependent |
| `test_latent_factors_have_standard_normal_margins` | margins $\mathcal{N}(0,1)$, mean 0 std 1 |
| `test_lognormal_marginal_is_exact` | lognormal marginal mean/std exact, corr matches weight |
| `test_run_hazard_matches_monte_carlo` | hazard inversion ($g_n^{-1}$) matches simulation to $\pm 0.003$ |
| `test_default_shift_hits_target` | logit shift hits target default rate to $10^{-6}$ |
| `test_windowed_default_rate_matches_knob` | final-label default rate $\approx$ target (0.08) |
| `test_latent_discrimination_matches_pd_curve` | AUC($f_6 \rightarrow$ default) $\approx 0.745$ |
| `test_ext_source_is_protective` | EXT_SOURCE lower default (AUC $< 0.45$) |
| `test_client_attributes_stable_across_months` | client attrs stable, age advances |

### End-to-end against the live k3d stack

Seed 21, 400 customers, 2026-03-01 (validated via real host ports):

| Check | Result |
|---|---|
| Postgres `lending_db` | clients 400, loan_applications 12, card_authorizations 11286 |
| SeaweedFS S3 `risk-data` | 5 entity parquet snapshots |
| Redpanda topics | `lending.public.*` all 5 streaming, Debezium envelopes `op=c` |
| MLflow / Filer / Polaris | HTTP 200/UP |

## 5. Statistical method (docs/statistics.md)

Full method in `docs/statistics.md` - t-copula with 7 latent factors,
marginal samplers, Kendall-$\tau$ calibration of $\Sigma$, profile
likelihood for $\nu$, exact hazard inversion for the gold label
($\mathrm{default} = 1$ iff 3 consecutive delinquent months within the first
$W = 12$ instalments). The current shipped copula parameters stay
**provisional** until re-calibrated on the real Home Credit files.

## 6. Known gaps / next steps

1. **Calibration provisional** - $\nu$ hit the old search bound (30),
   meaning "Gaussian preferred"; signs of factors 1, 2, 3, 5 unverified.
   Re-run `calibrate.py` against the real Home Credit CSVs.
2. **Backfill months** are generated but not written to sinks (forward
   months only call `to_sinks` today).
3. **Label gold layer** - `derive_default_labels_from_frames` assumptions
   are tied to the windowed default definition; tested, not stress-tested
   on long histories.
4. **Eval pipeline** - `scripts/eval_synthetic.py` needs the eval dependency
   group and the real HC CSVs for the baseline.
5. **Debezium API host port** - Connect's `:8083` is pod-internal only; add
   `-p "8083:8083"` to `ops/k8s/k3d-up.sh` if host access is ever needed.

## 7. Files touched (current state)

```
pyproject.toml                                  # workspace root; [dependency-groups]; [tool.uv] default-groups
uv.lock                                         # single lock for the whole repo
simulator/pyproject.toml                        # uv_build, [project.scripts], sinks extra
simulator/Dockerfile                            # 2-layer uv image
simulator/src/simulator/*.py                    # package source (moved from src/*.py)
infra/k8s/simulator/manifest.yaml               # CronJob
ops/k8s/k3d-up.sh                               # k3d bootstrap (NodePort = host port)
outputs/generated/                              # local write-through archive (gitignored)
```