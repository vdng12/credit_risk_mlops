"""Simulation reporting: structural + quality summaries.

Turns the ledger state and MonthBatch entities into a readable report,
emphasizing the structural dependencies the simulator now enforces
(account persistence, FK integrity, DPD/status evolution).
"""

from __future__ import annotations

from collections import Counter
from datetime import date
from typing import Any

import polars as pl

_ENTITY_ORDER = [
    "clients", "applications", "application_decisions", "application_master",
    "card_auths", "pos_balances", "card_balances", "bureau", "payments",
]

_HOT = {"applications", "card_auths"}
_COLD = {"clients", "application_decisions", "application_master",
         "pos_balances", "card_balances", "bureau", "payments"}


def summarize_batch(batch, simulator) -> dict[str, Any]:
    """Per-month summary: row counts, issues, structural stats."""
    names = {
        "clients": batch.clients, "applications": batch.applications,
        "application_decisions": batch.application_decisions,
        "application_master": batch.application_master,
        "card_auths": batch.card_auths, "pos_balances": batch.pos_balances,
        "card_balances": batch.card_balances, "bureau": batch.bureau,
        "payments": batch.payments,
    }
    rows = {k: (df.height if df is not None else 0) for k, df in names.items()}
    hot_rows = sum(rows[k] for k in _HOT)
    cold_rows = sum(rows[k] for k in _COLD)

    issues = batch.issues
    issue_counts = Counter(i["check"] for i in issues)
    error_count = sum(1 for i in issues if i["severity"] == "error")
    warn_count = sum(1 for i in issues if i["severity"] == "warn")

    # Structural stats
    approval_rate = 0.0
    if (batch.application_decisions is not None
            and batch.application_decisions.height > 0):
        approval_rate = (
            batch.application_decisions.filter(pl.col("status") == "approved").height
            / batch.application_decisions.height
        )

    return {
        "month": batch.month.isoformat(),
        "rows": rows,
        "hot_rows": hot_rows,
        "cold_rows": cold_rows,
        "total_rows": sum(rows.values()),
        "issues": issue_counts,
        "errors": error_count,
        "warnings": warn_count,
        "approval_rate": approval_rate,
        "ledger_loans": len(simulator.ledger.loans),
        "ledger_cards": len(simulator.ledger.cards),
        "ledger_bureau": len(simulator.ledger.bureau),
    }


def _dpd_breakdown(pos_df: pl.DataFrame | None) -> dict[str, int]:
    if pos_df is None or pos_df.height == 0 or "sk_dpd" not in pos_df.columns:
        return {"current": 0, "1_30": 0, "31_60": 0, "61_90": 0, "90_plus": 0}
    d = pos_df["sk_dpd"]
    return {
        "current": int((d == 0).sum()),
        "1_30": int(((d > 0) & (d <= 30)).sum()),
        "31_60": int(((d > 30) & (d <= 60)).sum()),
        "61_90": int(((d > 60) & (d <= 90)).sum()),
        "90_plus": int((d > 90).sum()),
    }


def summarize_run(batches, simulator) -> dict[str, Any]:
    """Aggregate summary across a whole run (backfill or forward)."""
    per_month = [summarize_batch(b, simulator) for b in batches]

    totals = {}
    for e in _ENTITY_ORDER:
        totals[e] = sum(m["rows"][e] for m in per_month)

    all_issues = Counter()
    for m in per_month:
        all_issues.update(m["issues"])

    final = batches[-1]
    last_dpd = _dpd_breakdown(final.pos_balances)
    approval = (
        [m["approval_rate"] for m in per_month if m["rows"]["applications"] > 0]
    )

    return {
        "n_months": len(batches),
        "total_rows": sum(totals.values()),
        "rows_by_entity": totals,
        "issues_by_check": dict(all_issues),
        "total_issues": sum(all_issues.values()),
        "final_ledger": {
            "loans": len(simulator.ledger.loans),
            "cards": len(simulator.ledger.cards),
            "bureau": len(simulator.ledger.bureau),
        },
        "final_dpd_breakdown": last_dpd,
        "mean_approval_rate": (sum(approval) / len(approval)) if approval else 0.0,
    }


def format_report(summary: dict[str, Any]) -> str:
    """Render a run summary as plain text."""
    lines: list[str] = []
    lines.append("=" * 62)
    lines.append("SIMULATION REPORT")
    lines.append("=" * 62)
    lines.append(f"Months simulated : {summary['n_months']}")
    lines.append(f"Total rows       : {summary['total_rows']:,}")

    lines.append("\nRows by entity (HOT = stream, COLD = batch):")
    for e, cnt in summary["rows_by_entity"].items():
        tag = "HOT " if e in {"applications", "card_auths"} else "COLD"
        lines.append(f"  {tag}  {e:20s} {cnt:>10,}")

    lines.append(f"\nFinal ledger state:")
    fb = summary["final_ledger"]
    lines.append(f"  loans  : {fb['loans']:,}")
    lines.append(f"  cards  : {fb['cards']:,}")
    lines.append(f"  bureau : {fb['bureau']:,}")

    lines.append(f"\nFinal DPD breakdown (loans):")
    db = summary["final_dpd_breakdown"]
    total_dpd = max(1, sum(db.values()))
    for k, v in db.items():
        lines.append(f"  {k:10s} {v:>6,}  ({100 * v / total_dpd:.1f}%)")

    lines.append(f"\nMean approval rate: {summary['mean_approval_rate']:.1%}")

    lines.append("\nQuality issues by check:")
    if summary["issues_by_check"]:
        for check, cnt in sorted(summary["issues_by_check"].items()):
            lines.append(f"  {check:22s} {cnt:>5}")
    else:
        lines.append("  (none)")

    lines.append("=" * 62)
    return "\n".join(lines)


def print_report(batches, simulator) -> None:
    print(format_report(summarize_run(batches, simulator)))