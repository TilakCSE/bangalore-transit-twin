-- scripts/init_postgres.sql
-- Runs once on first postgres container start
-- Creates separate DBs for MLflow and the transit app

CREATE DATABASE mlflow;
CREATE DATABASE transit_meta;

-- MLflow needs its own user in prod; for local dev share the airflow user
GRANT ALL PRIVILEGES ON DATABASE mlflow TO airflow;
GRANT ALL PRIVILEGES ON DATABASE transit_meta TO airflow;

-- Transit metadata schema (stop info cache, route metadata)
\connect transit_meta
CREATE SCHEMA IF NOT EXISTS transit;

CREATE TABLE IF NOT EXISTS transit.gtfs_feed_log (
    id              SERIAL PRIMARY KEY,
    feed_name       VARCHAR(64) NOT NULL,
    ingestion_date  DATE NOT NULL,
    checksum        VARCHAR(64),
    rows_loaded     INTEGER,
    status          VARCHAR(32) DEFAULT 'success',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS transit.model_registry_cache (
    model_name      VARCHAR(128) PRIMARY KEY,
    version         VARCHAR(32),
    stage           VARCHAR(32),
    artifact_uri    TEXT,
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);