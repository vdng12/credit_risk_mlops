"""CLI entry point for the simulator.

Usage:
    python -m simulator --n-customers 50000 --start 2020-01-01 --months 24
    python -m simulator backfill --until 2023-12-31
"""

from __future__ import annotations

import argparse
from datetime import date

from simulator.config import SimConfig
from simulator.core import Simulator
from simulator.report import print_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="t-Copula data simulator for default risk MLOps")
    parser.add_argument("--n-customers", type=int, default=50_000,
                        help="Number of synthetic customers")
    parser.add_argument("--start", type=str, default="2020-01-01",
                        help="Simulation start date (YYYY-MM-DD)")
    parser.add_argument("--months", type=int, default=12,
                        help="Number of months to simulate forward")
    parser.add_argument("--backfill-until", type=str, default=None,
                        help="Generate backfill history up to this date")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-problems", action="store_true",
                        help="Disable data problem injection")
    parser.add_argument("--output-dir", type=str, default="data/generated")
    args = parser.parse_args()

    cfg = SimConfig(
        n_customers=args.n_customers,
        start_date=date.fromisoformat(args.start),
        seed=args.seed,
    )
    cfg.problems.enabled = not args.no_problems
    cfg.sinks.file_output_dir = args.output_dir

    sim = Simulator(cfg)

    backfill_batches = []
    if args.backfill_until:
        print(f"Backfilling until {args.backfill_until}...")
        backfill_batches = sim.backfill(until=args.backfill_until)
        print(f"Generated {len(backfill_batches)} backfill months")
        print_report(backfill_batches, sim)

    print(f"\nSimulating {args.months} months forward...")
    forward_batches = []
    for batch in sim.play_forward(args.months):
        paths = batch.to_sinks(sim.sink, run_id="incremental", mode="incremental")
        forward_batches.append(batch)
        n_rows = sum(
            df.height for df in [
                batch.clients, batch.applications,
                batch.application_decisions, batch.application_master,
                batch.card_auths, batch.pos_balances, batch.card_balances,
                batch.bureau, batch.payments,
            ] if df is not None)
        n_issues = len(batch.issues)
        print(f"  {batch.month}: {n_rows:,} rows, {n_issues} quality issues, "
              f"{len(paths)} files written")

    print()
    print_report(forward_batches, sim)


if __name__ == "__main__":
    main()