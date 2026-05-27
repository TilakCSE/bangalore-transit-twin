"""
Flink → Iceberg Bronze Sink
─────────────────────────────────────────────────────────────────────────────
Uses the Iceberg AWS Bundle (AWS SDK v2) instead of hadoop-aws (SDK v1).
This is Option B from the research doc and avoids ALL the classpath conflicts.

Strategy:
  - warehouse uses s3:// (not s3a://) — matched to S3FileIO
  - io-impl = org.apache.iceberg.aws.s3.S3FileIO (Iceberg manages S3 directly)
  - Only 3 JARs needed: iceberg-flink-runtime, iceberg-aws-bundle, kafka connector
  - Runs SQL inside the Docker container via sql-client.sh (no PyFlink classpath)

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

# ── Config ────────────────────────────────────────────────────────────────────
FLINK_REST_URL        = "http://localhost:8082"
FLINK_KAFKA_BOOTSTRAP = os.getenv("FLINK_KAFKA_BOOTSTRAP", "kafka:29092")
# s3:// matches S3FileIO (Option B — no hadoop-aws needed)
ICEBERG_WAREHOUSE     = "s3://transit-twin-local/lakehouse"
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
MINIO_ENDPOINT        = "http://minio:9000"   # internal Docker DNS
TOPIC_VEHICLE_POS     = os.getenv("TOPIC_VEHICLE_POSITIONS", "vehicle-positions")
WATERMARK_LAG_SEC     = 60

LIB_DIR = Path(__file__).resolve().parents[3] / "lib"

# Only 3 JARs — clean, no Hadoop conflict
REQUIRED_JARS = [
    "iceberg-flink-runtime-1.19-1.6.1.jar",
    "iceberg-aws-bundle-1.6.1.jar",
    "flink-sql-connector-kafka-3.0.2-1.18.jar",
    "hadoop-common-3.3.4.jar",
    "hadoop-hdfs-client-3.3.4.jar",
    "woodstox-core-5.3.0.jar",
    "stax2-api-4.2.1.jar",
    "hadoop-shaded-guava-1.1.1.jar",
    "hadoop-auth-3.3.4.jar"  # <-- The missing security module!
]

DOWNLOAD_URLS = {
    "iceberg-flink-runtime-1.19-1.6.1.jar":
        "https://repo1.maven.org/maven2/org/apache/iceberg/"
        "iceberg-flink-runtime-1.19/1.6.1/iceberg-flink-runtime-1.19-1.6.1.jar",
    "iceberg-aws-bundle-1.6.1.jar":
        "https://repo1.maven.org/maven2/org/apache/iceberg/"
        "iceberg-aws-bundle/1.6.1/iceberg-aws-bundle-1.6.1.jar",
    "hadoop-common-3.3.4.jar":
        "https://repo1.maven.org/maven2/org/apache/hadoop/"
        "hadoop-common/3.3.4/hadoop-common-3.3.4.jar",
    "hadoop-hdfs-client-3.3.4.jar":
        "https://repo1.maven.org/maven2/org/apache/hadoop/"
        "hadoop-hdfs-client/3.3.4/hadoop-hdfs-client-3.3.4.jar",
    "woodstox-core-5.3.0.jar":
        "https://repo1.maven.org/maven2/com/fasterxml/woodstox/"
        "woodstox-core/5.3.0/woodstox-core-5.3.0.jar",
    "stax2-api-4.2.1.jar":
        "https://repo1.maven.org/maven2/org/codehaus/woodstox/"
        "stax2-api/4.2.1/stax2-api-4.2.1.jar",
    "hadoop-shaded-guava-1.1.1.jar":
        "https://repo1.maven.org/maven2/org/apache/hadoop/"
        "thirdparty/hadoop-shaded-guava/1.1.1/hadoop-shaded-guava-1.1.1.jar",
    "hadoop-auth-3.3.4.jar":
        "https://repo1.maven.org/maven2/org/apache/hadoop/"
        "hadoop-auth/3.3.4/hadoop-auth-3.3.4.jar"
}



# ── SQL that runs INSIDE the Flink container ─────────────────────────────────
def build_sql() -> str:
    return f"""
SET 'execution.runtime-mode' = 'streaming';
SET 'parallelism.default' = '2';
SET 'execution.checkpointing.interval' = '60000';
SET 'state.checkpoints.dir' = 'file:///tmp/flink-checkpoints-iceberg';

-- S3FileIO (AWS SDK v2) config — uses s3.* prefix, NOT fs.s3a.*
SET 's3.endpoint' = '{MINIO_ENDPOINT}';
SET 's3.access-key-id' = '{AWS_ACCESS_KEY_ID}';
SET 's3.secret-access-key' = '{AWS_SECRET_ACCESS_KEY}';
SET 's3.path-style-access' = 'true';

-- Create Iceberg catalog with S3FileIO (avoids all Hadoop classpath issues)
CREATE CATALOG iceberg_catalog WITH (
    'type'                   = 'iceberg',
    'catalog-type'           = 'hadoop',
    'warehouse'              = '{ICEBERG_WAREHOUSE}',
    'io-impl'                = 'org.apache.iceberg.aws.s3.S3FileIO',
    's3.endpoint'            = '{MINIO_ENDPOINT}',
    's3.access-key-id'       = '{AWS_ACCESS_KEY_ID}',
    's3.secret-access-key'   = '{AWS_SECRET_ACCESS_KEY}',
    's3.path-style-access'   = 'true'
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


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_containers(name_filter: str) -> list[str]:
    r = subprocess.run(
        ["docker", "ps", "--filter", f"name={name_filter}", "--format", "{{.Names}}"],
        capture_output=True, text=True, check=True,
    )
    return [c for c in r.stdout.strip().split("\n") if c]


def download_missing_jars() -> None:
    print("\n── Checking JARs ────────────────────────────────────────────")
    for jar_name in REQUIRED_JARS:
        jar_path = LIB_DIR / jar_name
        if jar_path.exists():
            print(f"  ✅ {jar_name}")
            continue
        url = DOWNLOAD_URLS.get(jar_name)
        if not url:
            raise FileNotFoundError(
                f"Missing JAR and no download URL: {jar_path}\n"
                f"Download manually and place in {LIB_DIR}/"
            )
        print(f"  ⬇  Downloading {jar_name}...")
        subprocess.run(
            ["wget", "-q", "--show-progress", "-P", str(LIB_DIR), url],
            check=True,
        )
        print(f"  ✅ {jar_name}")


def copy_jars(containers: list[str]) -> None:
    print("\n── Copying JARs into Flink containers ───────────────────────")
    for jar_name in REQUIRED_JARS:
        jar_path = LIB_DIR / jar_name
        for container in containers:
            subprocess.run(
                ["docker", "cp", str(jar_path),
                 f"{container}:/opt/flink/lib/{jar_name}"],
                check=True, capture_output=True,
            )
        print(f"  ✅ {jar_name} → {len(containers)} container(s)")


def restart_cluster(containers: list[str], wait_sec: int = 20) -> None:
    print("\n── Restarting Flink cluster ─────────────────────────────────")
    # Restart taskmanagers first, then jobmanager
    tms = [c for c in containers if "taskmanager" in c]
    jms = [c for c in containers if "jobmanager" in c]
    for c in tms + jms:
        subprocess.run(["docker", "restart", c], check=True, capture_output=True)
        print(f"  Restarted {c}")

    print(f"  Waiting {wait_sec}s...")
    time.sleep(wait_sec)

    for _ in range(20):
        try:
            r = requests.get(f"{FLINK_REST_URL}/overview", timeout=3)
            if r.status_code == 200:
                d = r.json()
                if d.get("taskmanagers", 0) > 0 and d.get("slots-total", 0) > 0:
                    print(f"  ✅ Cluster ready — "
                          f"{d['taskmanagers']} TMs, {d['slots-total']} slots")
                    return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("Cluster did not recover. Check: docker compose logs flink-jobmanager")


def submit_sql(container: str) -> None:
    sql_content = build_sql()
    local_sql   = "/tmp/transit_iceberg_sink.sql"
    remote_sql  = "/tmp/transit_iceberg_sink.sql"

    with open(local_sql, "w") as f:
        f.write(sql_content)

    subprocess.run(
        ["docker", "cp", local_sql, f"{container}:{remote_sql}"],
        check=True,
    )

    print("\n── Running Flink SQL Client inside container ────────────────")
    result = subprocess.run(
        ["docker", "exec", container,
         "/opt/flink/bin/sql-client.sh", "embedded", "-f", remote_sql],
        capture_output=True, text=True, timeout=90,
    )

    print("\n  ── SQL Client stdout ──")
    print(result.stdout[-3000:] if result.stdout else "  (empty)")

    if result.stderr:
        error_lines = [
            l for l in result.stderr.split("\n")
            if any(k in l for k in ["ERROR", "Exception", "FAILED"])
        ]
        if error_lines:
            print("\n  ── Errors ──")
            print("\n".join(error_lines))
            raise RuntimeError("SQL execution failed — see errors above")

    print("\n  SQL Client finished.")


def verify_jobs() -> None:
    print("\n── Job status ───────────────────────────────────────────────")
    time.sleep(6)
    try:
        jobs = requests.get(
            f"{FLINK_REST_URL}/jobs/overview", timeout=5
        ).json().get("jobs", [])
        if not jobs:
            print("  No jobs yet — INSERT may still be initialising")
            print(f"  Check: {FLINK_REST_URL}/#/overview in ~10s")
            return
        for job in jobs:
            state = job.get("state", "?")
            jid   = job.get("jid", "")
            print(f"  [{state}] {jid[:8]}...")
            if state == "RUNNING":
                print(f"\n  ✅ Iceberg sink is RUNNING")
                print(f"  Flink UI : {FLINK_REST_URL}/#/job/{jid}/overview")
                print(f"  MinIO    : http://localhost:9001")
                print(f"  Files    : transit-twin-local/lakehouse/bronze/vehicle_positions_raw/")
                print(f"  (Parquet files appear after first 60s checkpoint)")
            elif state == "FAILED":
                print(f"\n  ❌ Job FAILED")
                print(f"  Details: {FLINK_REST_URL}/#/job/{jid}/exceptions")
    except Exception as exc:
        print(f"  Could not query jobs: {exc}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("  Flink → Iceberg Bronze Sink  (AWS SDK v2 / S3FileIO)")
    print("=" * 60)
    print(f"\n  Warehouse : {ICEBERG_WAREHOUSE}")
    print(f"  Kafka     : {FLINK_KAFKA_BOOTSTRAP}")
    print(f"  MinIO     : {MINIO_ENDPOINT}")
    print(f"  Strategy  : S3FileIO (no hadoop-aws, no classpath conflicts)")

    # Verify Flink REST is reachable
    try:
        d = requests.get(f"{FLINK_REST_URL}/overview", timeout=5).json()
        print(f"\n  Flink {d.get('flink-version')} | "
              f"{d.get('taskmanagers')} TMs | {d.get('slots-total')} slots")
    except Exception:
        raise RuntimeError(
            f"Flink not reachable at {FLINK_REST_URL}\n"
            "Run: make up  and wait 30s"
        )

    jm = get_containers("flink-jobmanager")
    tm = get_containers("flink-taskmanager")
    if not jm:
        raise RuntimeError("No flink-jobmanager container running")

    download_missing_jars()
    copy_jars(jm + tm)
    restart_cluster(jm + tm, wait_sec=20)
    submit_sql(jm[0])
    verify_jobs()


if __name__ == "__main__":
    main()