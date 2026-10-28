"""Content-derived identity hash (Phase 4).

A row's payload hash is a deterministic fingerprint of its content - the
gospel for "is this the same event" checks (exact-duplicate detection,
idempotent consumption). Computed at DQ-report time rather than stored, so
generated tables keep their natural schema.

Any field change (including schema_version) changes the hash, so a v1 row
and a v2 row of otherwise-identical content hash differently.
"""

from __future__ import annotations

import hashlib
import json

import polars as pl


def _canonical_value(value) -> object:
    # Dates/datetimes/Decimals -> stable strings; everything else as-is.
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def row_payload_hash(record: dict) -> str:
    payload = {k: _canonical_value(v) for k, v in record.items()}
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def frame_payload_hashes(df: pl.DataFrame) -> pl.Series:
    """One content hash per row, in row order."""
    if df.height == 0:
        return pl.Series(dtype=pl.Utf8)
    return pl.Series(
        [row_payload_hash(r) for r in df.iter_rows(named=True)],
        dtype=pl.Utf8)
