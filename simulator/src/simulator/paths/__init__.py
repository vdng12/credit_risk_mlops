"""Lane paths: offline (batch) and online (stream) writers.

The same clean month is consumed differently per lane:

- OfflinePath  - write-through parquet archive + S3 cold facts (batch)
- OnlinePath   - OLTP postgres appends, the source of the CDC stream

Routing sets live here (not in MonthBatch), so the batch/stream contract
is explicit and testable.
"""

from __future__ import annotations

from simulator.paths.offline import OfflinePath, _S3_COLD_FACTS
from simulator.paths.online import OnlinePath, _STREAM_OLTP

__all__ = ["OfflinePath", "OnlinePath", "_S3_COLD_FACTS", "_STREAM_OLTP"]
