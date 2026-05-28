"""
Flink → Iceberg Bronze Sink  (Nessie REST catalog + S3FileIO)
─────────────────────────────────────────────────────────────────────────────
No JAR copying. No container restarts. No Hadoop classpath fights.
All JARs are pre-baked into the custom Flink image (Dockerfile.flink).

This script just submits the SQL via the Flink SQL Client CLI inside Docker.

Run:
    python3 -m stream_processing.flink.jobs.iceberg_sink
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

FLINK_REST_URL        = "http://localhost:8082"
FLINK_KAFKA_BOOTSTRAP = os.getenv("FLINK_KAFKA_BOOTSTRAP", "kafka:29092")
MINIO_ENDPOINT        = "http://minio:9000"
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
NESSIE_URI            = "http://nessie:19120/api/v1"
WAREHOUSE             = "s3://transit-twin-local/lakehouse"
TOPIC_VEHICLE_POS     = os.getenv("TOPIC_VEHICLE_POSITIONS", "vehicle-positions")
WATERMARK_LAG_SEC     = 60


def build_sql() -> str:
    return f"""
SET 'execution.runtime-mode' = 'streaming';
SET 'parallelism.default' = '2';
SET 'execution.checkpointing.interval' = '60000';
SET 'state.checkpoints.dir' = 'file:///tmp/flink-checkpoints-iceberg';

-- S3FileIO properties (AWS SDK v2, no Hadoop needed)
SET 's3.endpoint'            = '{MINIO_ENDPOINT}';
SET 's3.access-key-id'       = '{AWS_ACCESS_KEY_ID}';
SET 's3.secret-access-key'   = '{AWS_SECRET_ACCESS_KEY}';
SET 's3.path-style-access'   = 'true';

-- Nessie REST catalog: talks HTTP to Nessie container, no Hadoop config class needed
CREATE CATALOG iceberg_catalog WITH (
    'type'                      = 'iceberg',
    'catalog-impl'              = 'org.apache.iceberg.nessie.NessieCatalog',
    'io-impl'                   = 'org.apache.iceberg.aws.s3.S3FileIO',
    'uri'                       = '{NESSIE_URI}',
    'ref'                       = 'main',
    'warehouse'                 = '{WAREHOUSE}',
    's3.endpoint'               = '{MINIO_ENDPOINT}',
    's3.access-key-id'          = '{AWS_ACCESS_KEY_ID}',
    's3.secret-access-key'      = '{AWS_SECRET_ACCESS_KEY}',
    's3.path-style-access'      = 'true'
);

CREATE DATABASE IF NOT EXISTS iceberg_catalog.bronze;

USE CATALOG iceberg_catalog;
USE bronze;

CREATE TABLE IF NOT EXISTS vehicle_positions_raw (
    entity_id      STRING,
    vehicle_id     STRING,
    route_id       STRING,
    trip_id        STRING,
    latitude       DOUBLE,
    longitude      DOUBLE,
    bearing        DOUBLE,
    speed_mps      DOUBLE,
    current_status STRING,
    feed           STRING,
    event_ts       TIMESTAMP(3),
    ingestion_date STRING,
    ingested_at    BIGINT
) PARTITIONED BY (feed, ingestion_date)
WITH (
    'format-version'                  = '2',
    'write.format.default'            = 'parquet',
    'write.parquet.compression-codec' = 'snappy'
);

CREATE TABLE default_catalog.default_database.kafka_source (
    entity_id      STRING,
    vehicle_id     STRING,
    route_id       STRING,
    trip_id        STRING,
    latitude       DOUBLE,
    longitude      DOUBLE,
    bearing        DOUBLE,
    speed_mps      DOUBLE,
    current_status STRING,
    feed           STRING,
    `timestamp`    BIGINT,
    ingested_at    BIGINT,
    event_ts       AS TO_TIMESTAMP_LTZ(`timestamp`, 3),
    WATERMARK FOR event_ts
        AS event_ts - INTERVAL '{WATERMARK_LAG_SEC}' SECOND
) WITH (
    'connector'                    = 'kafka',
    'topic'                        = '{TOPIC_VEHICLE_POS}',
    'properties.bootstrap.servers' = '{FLINK_KAFKA_BOOTSTRAP}',
    'properties.group.id'          = 'flink-iceberg-sink',
    'scan.startup.mode'            = 'earliest-offset',
    'format'                       = 'json',
    'json.ignore-parse-errors'     = 'true'
);

INSERT INTO iceberg_catalog.bronze.vehicle_positions_raw
SELECT
    entity_id,
    vehicle_id,
    route_id,
    trip_id,
    latitude,
    longitude,
    bearing,
    speed_mps,
    current_status,
    feed,
    event_ts,
    DATE_FORMAT(event_ts, 'yyyy-MM-dd') AS ingestion_date,
    ingested_at
FROM default_catalog.default_database.kafka_source
WHERE vehicle_id IS NOT NULL
  AND latitude   IS NOT NULL
  AND longitude  IS NOT NULL;
"""


def get_jobmanager() -> str:
    r = subprocess.run(
        ["docker", "ps", "--filter", "name=flink-jobmanager",
         "--format", "{{.Names}}"],
        capture_output=True, text=True, check=True,
    )
    containers = [c for c in r.stdout.strip().split("\n") if c]
    if not containers:
        raise RuntimeError(
            "flink-jobmanager container not running.\n"
            "Run: docker compose ps  to check status."
        )
    return containers[0]


def wait_for_flink(timeout: int = 60) -> None:
    print("  Waiting for Flink REST API...", end="", flush=True)
    for _ in range(timeout // 2):
        try:
            r = requests.get(f"{FLINK_REST_URL}/overview", timeout=3)
            if r.status_code == 200:
                d = r.json()
                if d.get("taskmanagers", 0) > 0:
                    print(f" ready ({d['taskmanagers']} TMs, {d['slots-total']} slots)")
                    return
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(2)
    raise RuntimeError(
        f"\nFlink not reachable at {FLINK_REST_URL} after {timeout}s.\n"
        "Check: docker compose logs flink-jobmanager --tail=40"
    )


def wait_for_nessie(timeout: int = 30) -> None:
    print("  Waiting for Nessie catalog...", end="", flush=True)
    for _ in range(timeout // 2):
        try:
            r = requests.get("http://localhost:19120/api/v2/config", timeout=3)
            if r.status_code == 200:
                print(" ready")
                return
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(2)
    raise RuntimeError(
        "Nessie not reachable at localhost:19120.\n"
        "Check: docker compose logs nessie --tail=20"
    )


def submit_sql(container: str) -> None:
    sql_content = build_sql()
    local_sql   = "/tmp/transit_iceberg_nessie.sql"
    remote_sql  = "/tmp/transit_iceberg_nessie.sql"

    with open(local_sql, "w") as f:
        f.write(sql_content)

    subprocess.run(
        ["docker", "cp", local_sql, f"{container}:{remote_sql}"],
        check=True,
    )

    print(f"\n── Submitting SQL via Flink SQL Client ──────────────────────")
    result = subprocess.run(
        ["docker", "exec", container,
         "/opt/flink/bin/sql-client.sh", "embedded", "-f", remote_sql],
        capture_output=True, text=True, timeout=120,
    )

    print("\n── SQL Client output ────────────────────────────────────────")
    if result.stdout:
        print(result.stdout[-3000:])

    errors = [
        l for l in (result.stderr or "").split("\n")
        if any(k in l for k in ["ERROR", "Exception", "FAILED", "ClassNotFound"])
    ]
    if errors:
        print("\n── Errors ───────────────────────────────────────────────────")
        print("\n".join(errors))
        raise RuntimeError("SQL submission failed — see errors above")

    print("  SQL Client completed.")


def check_jobs() -> None:
    print("\n── Running jobs ─────────────────────────────────────────────")
    time.sleep(6)
    try:
        jobs = requests.get(
            f"{FLINK_REST_URL}/jobs/overview", timeout=5
        ).json().get("jobs", [])
        if not jobs:
            print("  No jobs yet — may still be starting (~10s)")
            print(f"  Check: {FLINK_REST_URL}/#/overview")
            return
        for job in jobs:
            state = job.get("state", "?")
            jid   = job.get("jid", "")
            print(f"  [{state}] {jid[:8]}...")
            if state == "RUNNING":
                print(f"\n  ✅ Iceberg sink RUNNING!")
                print(f"  Flink   : {FLINK_REST_URL}/#/job/{jid}/overview")
                print(f"  MinIO   : http://localhost:9001")
                print(f"  Nessie  : http://localhost:19120")
                print(f"  Parquet files appear after first checkpoint (~60s)")
            elif state == "FAILED":
                print(f"  ❌ FAILED — {FLINK_REST_URL}/#/job/{jid}/exceptions")
    except Exception as exc:
        print(f"  Could not query: {exc}")


def main() -> None:
    print("=" * 60)
    print("  Flink → Iceberg  (Nessie catalog + S3FileIO)")
    print("=" * 60)

    wait_for_flink()
    wait_for_nessie()

    container = get_jobmanager()
    print(f"\n  JobManager: {container}")

    submit_sql(container)
    check_jobs()


if __name__ == "__main__":
    main()