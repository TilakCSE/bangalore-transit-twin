"""
Flink → Iceberg Bronze Sink  (REST API submission)
─────────────────────────────────────────────────────────────────────────────
Instead of fighting PyFlink's classloader, this script:

1. Copies the required JARs into the running Flink JobManager container
2. Submits a pure SQL job via Flink's SQL Client CLI inside Docker
3. The job runs entirely inside Docker where all Hadoop classes exist

This completely bypasses the NoClassDefFoundError issues.

Run:
    python3 -m stream_processing.flink.jobs.iceberg_sink

Requirements:
    pip install requests
    Docker stack running (make up)
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
ICEBERG_WAREHOUSE = os.getenv("ICEBERG_WAREHOUSE", "s3a://transit-twin-local/lakehouse")
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
# Inside Docker, MinIO is reachable via its service name
AWS_ENDPOINT_INTERNAL = "http://minio:9000"
TOPIC_VEHICLE_POS     = os.getenv("TOPIC_VEHICLE_POSITIONS", "vehicle-positions")
WATERMARK_LAG_SEC     = 60

LIB_DIR = Path(__file__).resolve().parents[3] / "lib"

REQUIRED_JARS = [
    "iceberg-flink-runtime-1.19-1.6.1.jar",
    "iceberg-aws-bundle-1.6.1.jar",
    "flink-sql-connector-kafka-3.0.2-1.18.jar",
    "hadoop-common-3.3.4.jar",
    "hadoop-hdfs-client-3.3.4.jar",
    "hadoop-mapreduce-client-core-3.3.4.jar",
    "hadoop-auth-3.3.4.jar",
    "hadoop-aws-3.3.4.jar",
    "aws-java-sdk-bundle-1.12.262.jar",
    "hadoop-shaded-guava-1.1.1.jar",
    "commons-configuration2-2.1.1.jar",
    "commons-logging-1.2.jar",
    "woodstox-core-5.3.0.jar",
    "stax2-api-4.2.1.jar"
]


def get_container(name_filter: str) -> list[str]:
    result = subprocess.run(
        ["docker", "ps", "--filter", f"name={name_filter}", "--format", "{{.Names}}"],
        capture_output=True, text=True, check=True
    )
    return [c for c in result.stdout.strip().split("\n") if c]


def copy_jars_to_flink(containers: list[str]) -> None:
    print("\n── Copying JARs into Flink containers ───────────────────────")
    for jar_name in REQUIRED_JARS:
        jar_path = LIB_DIR / jar_name
        if not jar_path.exists():
            raise FileNotFoundError(
                f"\nMissing JAR: {jar_path}\n"
                f"Run:\n  cd lib && wget https://repo1.maven.org/maven2/org/apache/iceberg/"
                f"iceberg-flink-runtime-1.19/1.6.1/iceberg-flink-runtime-1.19-1.6.1.jar"
            )
        for container in containers:
            subprocess.run(
                ["docker", "cp", str(jar_path),
                 f"{container}:/opt/flink/lib/{jar_name}"],
                check=True, capture_output=True
            )
        print(f"  ✅ {jar_name}")


def restart_and_wait(containers: list[str], wait_sec: int = 15) -> None:
    print(f"\n── Restarting containers to load JARs ───────────────────────")
    for c in containers:
        subprocess.run(["docker", "restart", c], check=True, capture_output=True)
        print(f"  Restarted {c}")
    print(f"  Waiting {wait_sec}s for cluster to stabilise...")
    time.sleep(wait_sec)

    for attempt in range(20):
        try:
            r = requests.get(f"{FLINK_REST_URL}/overview", timeout=3)
            if r.status_code == 200:
                d = r.json()
                tm = d.get("taskmanagers", 0)
                slots = d.get("slots-total", 0)
                if tm > 0 and slots > 0:
                    print(f"  ✅ Cluster ready — {tm} task managers, {slots} slots")
                    return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("Cluster did not recover after restart")


def build_sql() -> str:
    return f"""
SET 'execution.runtime-mode' = 'streaming';
SET 'parallelism.default' = '2';
SET 'execution.checkpointing.interval' = '60000';
SET 'state.checkpoints.dir' = 'file:///tmp/flink-checkpoints-iceberg';
SET 'fs.s3a.access.key' = '{AWS_ACCESS_KEY_ID}';
SET 'fs.s3a.secret.key' = '{AWS_SECRET_ACCESS_KEY}';
SET 'fs.s3a.endpoint' = '{AWS_ENDPOINT_INTERNAL}';
SET 'fs.s3a.path.style.access' = 'true';

CREATE CATALOG iceberg_catalog WITH (
    'type'                   = 'iceberg',
    'catalog-type'           = 'hadoop',
    'warehouse'              = 's3a://transit-twin-local/lakehouse',
    's3.endpoint'            = 'http://minio:9000',
    's3.access-key-id'       = 'minioadmin',
    's3.secret-access-key'   = 'minioadmin',
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
    WATERMARK FOR event_ts AS event_ts - INTERVAL '{WATERMARK_LAG_SEC}' SECOND
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


def submit_via_sql_client(container: str) -> None:
    sql = build_sql()
    sql_path_host = "/tmp/transit_iceberg_sink.sql"
    sql_path_container = "/tmp/transit_iceberg_sink.sql"

    # Write SQL file locally then copy into container
    with open(sql_path_host, "w") as f:
        f.write(sql)

    subprocess.run(
        ["docker", "cp", sql_path_host, f"{container}:{sql_path_container}"],
        check=True
    )
    print(f"\n── Submitting via Flink SQL Client ──────────────────────────")
    print(f"  SQL file copied to container")

    # Run SQL client in detached mode — it submits the job and exits
    result = subprocess.run(
        ["docker", "exec",
         "-e", f"AWS_ACCESS_KEY_ID={AWS_ACCESS_KEY_ID}",
         "-e", f"AWS_SECRET_ACCESS_KEY={AWS_SECRET_ACCESS_KEY}",
         container,
         "/opt/flink/bin/sql-client.sh",
         "embedded", "-f", sql_path_container],
        capture_output=True, text=True, timeout=60
    )

    if result.stdout:
        print(f"\n  SQL Client output:\n{result.stdout[-2000:]}")
    if result.stderr:
        # SQL client writes info to stderr — only show if actual error
        relevant = [l for l in result.stderr.split("\n")
                    if any(k in l for k in ["ERROR", "Exception", "FAILED", "Job ID"])]
        if relevant:
            print(f"\n  Relevant stderr:\n" + "\n".join(relevant))

    if result.returncode != 0 and "ERROR" in result.stderr:
        raise RuntimeError(f"SQL Client failed:\n{result.stderr[-1000:]}")


def check_running_jobs() -> None:
    print(f"\n── Checking running jobs ─────────────────────────────────────")
    time.sleep(5)
    try:
        resp = requests.get(f"{FLINK_REST_URL}/jobs/overview", timeout=5)
        jobs = resp.json().get("jobs", [])
        if not jobs:
            print("  No jobs found yet — the INSERT job may still be starting")
            print(f"  Check: {FLINK_REST_URL}/#/overview")
            return
        for job in jobs:
            status = job.get("state", "?")
            jid    = job.get("jid", "")[:8]
            name   = job.get("name", "")[:40]
            print(f"  [{status:10s}] {jid}... {name}")
            if status == "RUNNING":
                print(f"\n  ✅ Iceberg sink RUNNING!")
                print(f"  Monitor : {FLINK_REST_URL}/#/job/{job['jid']}/overview")
                print(f"  MinIO   : http://localhost:9001")
                print(f"  Data appears after first checkpoint (~60s)")
    except Exception as exc:
        print(f"  Could not check jobs: {exc}")


def main() -> None:
    print("=" * 60)
    print("  Flink → Iceberg Bronze Sink Deployer")
    print("=" * 60)
    print(f"\n  Warehouse : {ICEBERG_WAREHOUSE}")
    print(f"  Kafka     : {FLINK_KAFKA_BOOTSTRAP}")

    # Verify Flink is up
    try:
        overview = requests.get(f"{FLINK_REST_URL}/overview", timeout=5).json()
        print(f"  Flink     : {overview.get('flink-version')} | "
              f"{overview.get('taskmanagers')} TMs | "
              f"{overview.get('slots-total')} slots")
    except Exception:
        raise RuntimeError(f"Cannot reach Flink at {FLINK_REST_URL}. Run: make up")

    # Get all Flink containers
    jm_containers = get_container("flink-jobmanager")
    tm_containers = get_container("flink-taskmanager")
    all_containers = jm_containers + tm_containers

    if not jm_containers:
        raise RuntimeError("No Flink JobManager container found")

    print(f"\n  Containers: {all_containers}")

    # Copy JARs into all Flink containers
    copy_jars_to_flink(all_containers)

    # Restart so Flink picks up the new JARs from /opt/flink/lib/
    restart_and_wait(all_containers, wait_sec=18)

    # Submit the SQL job from inside the JobManager container
    submit_via_sql_client(jm_containers[0])

    # Verify
    check_running_jobs()


if __name__ == "__main__":
    main()