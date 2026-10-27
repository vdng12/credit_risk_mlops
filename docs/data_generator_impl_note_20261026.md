# Data Generator Implementation Note — 2026-10-26

**Scope:** Wiring the `simulator/` t-copula data generator as a standalone uv
project inside the workspace monorepo, validating the pipeline end-to-end,
and documenting the hybrid workspace layout (independent locks vs root
workspace).

---

## 1. What changed

| Item | Before | After |
|---|---|---|
| Simulator deps | declared in root `pyproject.toml` | own `pyproject.toml` at `simulator/`, own `uv.lock`, own `.venv` |
| Package layout | `simulator/*.py` flat | `simulator/src/*.py` (src layout, forced to install as `simulator` package via hatch `force-include`) |
| Root repo | single project | workspace root, `[tool.uv.workspace] members = ["simulator"]` |
| Root deps | `simulator` via path source | `simulator = { workspace = true }` |

### Why src layout with force-include

The user requested `simulator/src/*.py` (not `src/simulator/*.py`). Hatch's
`packages = ["src"]` would install a package named `src`, breaking
`import simulator`. Fixed with:

```toml
[tool.hatch.build.targets.wheel]
force-include = { "src" = "simulator" }
```

This rebases the `src/` directory into a `simulator/` package in the wheel.

### Workspace wiring

```toml
# root pyproject.toml
dependencies = ["simulator"]

[tool.uv.sources]
simulator = { workspace = true }

[tool.uv.workspace]
members = ["simulator"]
```

Key uv nuance (from guide): a workspace member must be declared in
`tool.uv.sources` with `{ workspace = true }` — a plain `simulator = { path = ... }`
fails with "must be declared as workspace sources".

---

## 2. Validation results

### Tests (11/11 passing)

```bash
$ uv run pytest tests/test_statistics.py -q
...........                                                              [100%]
11 passed in 5.54s
```

| Test | Verifies |
|---|---|
| `test_t_copula_recovers_nu_and_sigma` | fitting recovers known ν, Σ within tolerance |
| `test_gaussian_data_is_flagged` | Gaussian data flagged, not tail-dependent |
| `test_latent_factors_have_standard_normal_margins` | margins N(0,1), mean 0 std 1 |
| `test_lognormal_marginal_is_exact` | lognormal marginal mean/std exact, corr matches weight |
| `test_run_hazard_matches_monte_carlo` | hazard inversion matches simulation to ±0.003 |
| `test_default_shift_hits_target` | logit shift hits target default rate to 1e-6 |
| `test_windowed_default_rate_matches_knob` | final-label default rate ≈ target (0.08) |
| `test_latent_discrimination_matches_pd_curve` | AUC(f₆ → default) ≈ 0.745 |
| `test_ext_source_is_protective` | EXT_SOURCE lower default (AUC < 0.45) |
| `test_client_attributes_stable_across_months` | client attrs stable, age advances |

### Generator run (100k customers)

```bash
uv run python -m simulator --n-customers 100000 --months 6
```

| Metric | Value |
|---|---|
| Rows generated | 917,297 |
| Parquet files | ~60 (7 entity types × ~6 months + decisions) |
| Disk used | 37 MB |
| Wall time | ~2s (problems/sinks disabled) |

Entity types emitted: clients, application_decisions, application_master,
pos_balances, card_balances, bureau, payments.

---

## 3. Output wiring

The `__main__` CLI drives sinks; note the **hot/cold split** in
`MonthBatch.hot_events()` and `MonthBatch.cold_batches()`:

- **HOT (stream)** — loan_applications, card_authorizations → Kafka topics
- **COLD (batch)** — clients, decisions, master, balances, bureau, payments → Parquet/S3/Postgres

`Simulator.__main__` was verified to run but **sink connections (Postgres,
S3, Kafka) will hang if the services aren't reachable**. For pure file output
use the direct `SinkManager.to_parquet` path (as done in validation) or
ensure the k3d cluster services are up.

---

## 4. Known gaps / next steps

1. **Calibration provisional** — ν = 30 hit the old search bound ("Gaussian
   preferred"); signs of factors 1,2,3,5 unverified. Re-run `calibrate.py`
   against the real Home Credit CSVs for shipped parameters.
2. **Sinks integration** — validate `to_postgres` / `to_s3` / `to_kafka`
   against the running k3d stack (postgres-source :5432, seaweedfs :8333,
   redpanda :9092).
3. **Label gold layer** — `derive_default_labels_from_frames` assumptions
   tied to the windowed default definition (3 consecutive DPD months within
   first 12 instalments); currently only validated via tests.
4. **Eval pipeline** — `scripts/eval_synthetic.py` runs LightGBM on synthetic
   vs real HC data; needs the dev-extra deps and real HC csv for the baseline.

---

## 5. Files touched

```
pyproject.toml                                  # workspace root, simulator dep
uv.lock
simulator/pyproject.toml                        # NEW — standalone project
simulator/uv.lock                               # NEW
simulator/src/*.py                              # MOVED from simulator/*.py
tests/test_statistics.py                        # unchanged, passing
scripts/eval_synthetic.py                       # unchanged, exposed
output/                                          # generated parquet (gitignored?)
```