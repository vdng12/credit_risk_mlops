-- =============================================================================
-- PostgreSQL init — ML platform metadata databases
-- =============================================================================
-- Runs once at first boot of platform-postgres (the core compose service).
-- POSTGRES_DB ("platform") already exists; these add the metadata stores.
-- =============================================================================

CREATE DATABASE mlflow;
CREATE DATABASE feast;
CREATE DATABASE metastore;