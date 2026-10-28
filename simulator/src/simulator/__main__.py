"""CLI entry point for the simulator.

Subcommands (one config story, one seed):

    simulator [overrides] generate [--config <yaml>] [--run-id X]
    simulator validate --config <yaml> --run-id X
    simulator resume [--config <yaml>] [--months N] [--backfill-until D]
    simulator calibrate --data-dir <home-credit-data>

Backwards-compatible flat form (defaults to ``generate``):

    python -m simulator --n-customers 50000 --start 2020-01-01 --months 24
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from simulator.calibrate import calibrate
from simulator.config import SimConfig
from simulator.config_loader import CALIBRATION_DIR, COPULA_ARTIFACT, load_config
from simulator.core import Simulator
from simulator.quality import QualityValidator
from simulator.report import print_report

# Validator entity key per parquet file name (validator's _ENTITY_KEYS).
_VALIDATE_ENTITIES = {
    "clients": "client",
    "applications": "application",
    "card_auths": "card_auth",
    "pos_balances": "pos_balance",
    "card_balances": "card_balance",
    "bureau": "bureau",
    "payments": "payment",
}


def _common_parser(suppress: bool = False) -> argparse.ArgumentParser:
    """Override flags shared by the main parser and the subcommands.

    Subcommand parsers use ``default=SUPPRESS`` so a flag given *before*
    the subcommand (on the main parser) survives the subparser merge
    instead of being reset to None.
    """
    p = argparse.ArgumentParser(add_help=False)
    default = argparse.SUPPRESS if suppress else None
    p.add_argument("--config", type=str, default=default)
    p.add_argument("--run-id", type=str, default=default)
    p.add_argument("--seed", type=int, default=default)
    p.add_argument("--n-customers", type=int, default=default)
    p.add_argument("--start", type=str, default=default)
    p.add_argument("--months", type=int, default=default)
    p.add_argument("--backfill-until", type=str, default=default)
    p.add_argument("--no-problems", action="store_true", default=default)
    p.add_argument("--no-s3", action="store_true", default=default)
    p.add_argument("--no-postgres", action="store_true", default=default)
    p.add_argument("--output-dir", type=str, default=default)
    return p


def _apply_overrides(cfg: SimConfig, ns: argparse.Namespace) -> dict:
    """Apply CLI overrides to the loaded config; return run-scope vars."""
    if getattr(ns, "seed", None) is not None:
        cfg.seed = ns.seed
    if getattr(ns, "n_customers", None) is not None:
        cfg.entities.n_customers = ns.n_customers
    if getattr(ns, "start", None) is not None:
        cfg.time.start_date = date.fromisoformat(ns.start)
    if getattr(ns, "no_problems", False):
        cfg.problems.enabled = False
    if getattr(ns, "no_s3", False):
        cfg.sinks.enable_s3 = False
    if getattr(ns, "no_postgres", False):
        cfg.sinks.enable_postgres = False
    if getattr(ns, "output_dir", None) is not None:
        cfg.sinks.file_output_dir = ns.output_dir
    return {
        "run_id": getattr(ns, "run_id", None) or cfg.run_id or "default",
        "months": getattr(ns, "months", None) or 12,
        "backfill_until": (
            date.fromisoformat(ns.backfill_until)
            if getattr(ns, "backfill_until", None) else None),
    }


def _run_batches(sim: Simulator, batches, run_id: str) -> None:
    for batch in batches:
        paths = batch.to_sinks(sim.sink, run_id=run_id)
        n_rows = sum(
            df.height for df in [
                batch.clients, batch.applications,
                batch.application_decisions, batch.application_master,
                batch.card_auths, batch.pos_balances, batch.card_balances,
                batch.bureau, batch.payments,
            ] if df is not None)
        print(f"  {batch.month}: {n_rows:,} rows, "
              f"{len(batch.issues)} quality issues, {len(paths)} files")


def cmd_generate(ns: argparse.Namespace) -> int:
    cfg = load_config(ns.config)
    scope = _apply_overrides(cfg, ns)
    run_id, months, backfill_until = (
        scope["run_id"], scope["months"], scope["backfill_until"])

    sim = Simulator(cfg)
    batches: list = []

    if backfill_until is not None:
        print(f"Backfilling until {backfill_until} (run_id={run_id})...")
        bf = sim.backfill(until=backfill_until)
        _run_batches(sim, bf, run_id)
        batches.extend(bf)

    print(f"Simulating {months} months forward...")
    fwd = list(sim.play_forward(months))
    _run_batches(sim, fwd, run_id)
    batches.extend(fwd)

    print()
    print_report(batches, sim)
    return 0


def cmd_resume(ns: argparse.Namespace) -> int:
    """Rebuild state to the bookmark (recompute deterministic months),
    then continue forward. The batch lane is not rewritten."""
    cfg = load_config(ns.config)
    scope = _apply_overrides(cfg, ns)
    run_id, months, backfill_until = (
        scope["run_id"], scope["months"], scope["backfill_until"])

    if backfill_until is None:
        backfill_until = _read_bookmark(run_id, cfg)
    if backfill_until is None:
        print("resume needs a resume point: pass --backfill-until (or run "
              "with postgres enabled to read the sim_state bookmark)",
              file=sys.stderr)
        return 2

    sim = Simulator(cfg)
    print(f"Rebuilding state to {backfill_until} (deterministic)...")
    sim.backfill(until=backfill_until)   # state only, no sink writes
    print(f"Continuing {months} months forward (run_id={run_id})...")
    fwd = list(sim.play_forward(months))
    _run_batches(sim, fwd, run_id)
    print_report(fwd, sim)
    return 0


def _read_bookmark(run_id: str, cfg: SimConfig) -> date | None:
    """Best-effort read of the postgres sim_state bookmark (Phase 0)."""
    try:
        import sqlalchemy as sa
    except ImportError:
        return None
    try:
        url = sa.engine.make_url(cfg.sinks.postgres_dsn)
        if url.drivername == "postgresql":
            url = url.set(drivername="postgresql+psycopg2")
        engine = sa.create_engine(url)
        with engine.connect() as conn:
            row = conn.execute(
                sa.text(
                    "SELECT last_month FROM sim_state WHERE run_id = :rid"),
                {"rid": run_id}).fetchone()
        engine.dispose()
        return date.fromisoformat(row[0]) if row else None
    except Exception:
        return None


def cmd_validate(ns: argparse.Namespace) -> int:
    cfg = load_config(ns.config)
    run_id = ns.run_id or "default"
    out = Path(cfg.sinks.file_output_dir) / run_id
    if not out.exists():
        print(f"no output for run_id={run_id} at {out}", file=sys.stderr)
        return 2

    validator = QualityValidator(cfg)
    months = sorted({p.name.split("_")[-1].removesuffix(".parquet")
                     for p in out.glob("*.parquet")})
    all_issues: list = []
    for month in months:
        entities: dict = {}
        for fname, key in _VALIDATE_ENTITIES.items():
            path = out / f"{fname}_{month}.parquet"
            if path.exists():
                import polars as pl
                entities[key] = pl.read_parquet(path)
        snapshot = date.fromisoformat(f"{month}-01")
        issues = validator.validate_batch(entities, snapshot_date=snapshot)
        all_issues.extend(issues)
        print(f"  {month}: {len(issues)} issues")

    errors = [i for i in all_issues if i["severity"] == "error"]
    print(f"\n{len(all_issues)} issues total, {len(errors)} errors")
    for issue in all_issues[:20]:
        print(f"  [{issue['severity']}] {issue['entity']} "
              f"{issue['check']}: {issue.get('detail', '')}")
    return 1 if errors else 0


def cmd_calibrate(ns: argparse.Namespace) -> int:
    if ns.data_dir is None:
        print("calibrate needs --data-dir <home-credit-data>", file=sys.stderr)
        return 2
    result = calibrate(ns.data_dir, subsample=ns.subsample, seed=ns.seed or 42)
    COPULA_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    with open(COPULA_ARTIFACT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {COPULA_ARTIFACT}")

    from simulator import calibrate_structures
    params = calibrate_structures.main(ns.data_dir)
    params_path = Path(
        calibrate_structures.__file__).with_name("calibrated_params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Wrote {params_path}")
    return 0


def main() -> None:
    common = _common_parser()
    parser = argparse.ArgumentParser(
        prog="simulator", parents=[common],
        description="t-Copula data simulator for default risk MLOps")
    sub = parser.add_subparsers(dest="command")
    sub.required = False

    gen = sub.add_parser("generate", parents=[_common_parser(suppress=True)],
                         help="backfill + forward run, writes sinks")
    gen.set_defaults(func=cmd_generate)

    val = sub.add_parser("validate", parents=[_common_parser(suppress=True)],
                         help="validate a run's parquet output")
    val.set_defaults(func=cmd_validate)

    res = sub.add_parser("resume", parents=[_common_parser(suppress=True)],
                         help="rebuild state to bookmark, continue forward")
    res.set_defaults(func=cmd_resume)

    cal = sub.add_parser("calibrate", parents=[_common_parser(suppress=True)],
                         help="fit copula + child-table params from real data")
    cal.add_argument("--data-dir", type=str, default=None)
    cal.add_argument("--subsample", type=int, default=None)
    cal.set_defaults(func=cmd_calibrate)

    ns = parser.parse_args()
    if ns.command is None:
        ns.command = "generate"
        ns.func = cmd_generate
    try:
        raise SystemExit(ns.func(ns))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()