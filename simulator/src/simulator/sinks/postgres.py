"""Postgres OLTP sink - append-only, tolerating schema projection.

The OLTP tables are the **source of the Debezium CDC stream** (Kafka /
Redpanda topics are NOT written by the generator). ``write()`` projects
onto the target table's columns so e.g. loan_applications only carries
pre-decision fields while the outcome lives in application_decisions.
"""

from __future__ import annotations

import polars as pl


class PostgresSink:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def write(
        self,
        df: pl.DataFrame,
        table_name: str,
        if_exists: str = "append",
    ) -> None:
        import sqlalchemy as sa

        # `postgresql://` defaults to the psycopg (v3) dialect in modern
        # SQLAlchemy; pin the installed driver (psycopg2-binary) when no
        # explicit `+driver` is present in the DSN.
        url = sa.engine.make_url(self.dsn)
        if url.drivername == "postgresql":
            url = url.set(drivername="postgresql+psycopg2")
        engine = sa.create_engine(url)
        pdf = df.to_pandas()
        with engine.connect() as conn:
            inspector = sa.inspect(conn)
            if table_name in inspector.get_table_names():
                existing = {c["name"] for c in inspector.get_columns(table_name)}
                pdf = pdf[[c for c in pdf.columns if c in existing]]
        pdf.to_sql(table_name, engine, if_exists=if_exists, index=False)
        engine.dispose()
