"""
Flink → Iceberg Bronze Sink
─────────────────────────────────────────────────────────────────────────────
Consumes vehicle positions from Kafka and writes them permanently to the
Bronze layer of the Iceberg Lakehouse on MinIO (local) / GCS (production).

Table: bronze.vehicle_positions_raw
Partitioned by: feed, ingestion_date
Format: Parquet (Iceberg default)

This is a separate job from delay_detection.py — it runs alongside it.
Both consume the same 'vehicle-positions' topic with different consumer groups
so they don't interfere with each other.

Run:
    python3 -m stream_processing.flink.jobs.iceberg_sink

Monitor:
    http://localhost:8082  → Running Jobs → iceberg-bronze-sink
    http://localhost:9001  → MinIO → transit-twin-local/lakehouse/bronze/
"""

from __future__ import annotations

import os
from pathlib import Path

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import EnvironmentSettings, StreamTableEnvironment

# ── Config ────────────────────────────────────────────────────────────────────
FLINK_KAFKA_BOOTSTRAP = os.getenv("FLINK_KAFKA_BOOTSTRAP", "kafka:29092")
WATERMARK_LAG_SEC     = 60
ICEBERG_WAREHOUSE     = os.getenv("ICEBERG_WAREHOUSE", "s3://transit-twin-local/lakehouse")
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
AWS_ENDPOINT          = os.getenv("AWS_ENDPOINT", "http://minio:9000")

LIB_DIR = Path(__file__).resolve().parents[3] / "lib"

# All JARs needed for this job
JARS = [
    LIB_DIR / "flink-sql-connector-kafka-3.0.2-1.18.jar",
    LIB_DIR / "iceberg-flink-runtime-1.19-1.6.1.jar",
    LIB_DIR / "hadoop-aws-3.3.4.jar",
    LIB_DIR / "aws-java-sdk-bundle-1.12.262.jar",
]


def _verify_jars() -> list[str]:
    """Check all required JARs exist before submitting."""
    jar_uris = []
    missing = []
    for jar in JARS:
        if jar.exists():
            jar_uris.append(f"file://{jar}")
        else:
            missing.append(str(jar))
    if missing:
        raise FileNotFoundError(
            f"Missing JARs — download them first:\n" +
            "\n".join(f"  {j}" for j in missing)
        )
    return jar_uris


def build_iceberg_sink() -> None:
    jar_uris = _verify_jars()

    # ── Connect to Docker Flink cluster ───────────────────────────────────────
    env = StreamExecutionEnvironment.create_remote_execution_environment(
        host="localhost",
        port=8081,
        jar_files=[str(j).replace("file://", "") for j in jar_uris],
    )
    env.set_parallelism(2)
    env.enable_checkpointing(60_000)   # 60s — Iceberg commits on checkpoint
    env.get_checkpoint_config().set_checkpoint_storage_dir(
        "file:///tmp/flink-checkpoints-iceberg"
    )

    settings = EnvironmentSettings.new_instance().in_streaming_mode().build()
    t_env = StreamTableEnvironment.create(env, settings)

    # ── S3 / MinIO configuration ──────────────────────────────────────────────
    # These Hadoop S3A properties let Flink write to MinIO using the S3 protocol
    t_env.get_config().set("s3.endpoint", AWS_ENDPOINT)
    t_env.get_config().set("s3.access-key", AWS_ACCESS_KEY_ID)
    t_env.get_config().set("s3.secret-key", AWS_SECRET_ACCESS_KEY)
    t_env.get_config().set("s3.path.style.access", "true")

    # ── Create Iceberg catalog pointing at MinIO ──────────────────────────────
    # Uses Hadoop catalog (file-based, no REST catalog needed for local dev)
    t_env.execute_sql(f"""
        CREATE CATALOG iceberg_catalog WITH (
            'type'            = 'iceberg',
            'catalog-type'    = 'hadoop',
            'warehouse'       = '{ICEBERG_WAREHOUSE}',
            'property-version'= '1',
            'io-impl'         = 'org.apache.iceberg.aws.s3.S3FileIO',
            's3.endpoint'     = '{AWS_ENDPOINT}',
            's3.access-key-id'     = '{AWS_ACCESS_KEY_ID}',
            's3.secret-access-key' = '{AWS_SECRET_ACCESS_KEY}',
            's3.path-style-access' = 'true'
        )
    """)

    t_env.execute_sql("CREATE DATABASE IF NOT EXISTS iceberg_catalog.bronze")

    # ── Create Bronze Iceberg table ───────────────────────────────────────────
    # PARTITIONED BY (feed, ingestion_date) for efficient time-range queries
    # This DDL is idempotent — safe to run multiple times
    t_env.execute_sql("""
        CREATE TABLE IF NOT EXISTS iceberg_catalog.bronze.vehicle_positions_raw (
            entity_id       STRING,
            vehicle_id      STRING,
            route_id        STRING,
            trip_id         STRING,
            latitude        DOUBLE,
            longitude       DOUBLE,
            bearing         DOUBLE,
            speed_mps       DOUBLE,
            current_status  STRING,
            feed            STRING,
            event_ts        TIMESTAMP(3),
            ingestion_date  STRING,
            ingested_at     BIGINT
        ) PARTITIONED BY (feed, ingestion_date)
        WITH (
            'format-version' = '2',
            'write.format.default' = 'parquet',
            'write.parquet.compression-codec' = 'snappy'
        )
    """)

    # ── Source: vehicle positions from Kafka ──────────────────────────────────
    # Different consumer group from delay_detection.py — both read independently
    t_env.execute_sql(f"""
        CREATE TABLE kafka_vehicle_positions (
            entity_id       STRING,
            vehicle_id      STRING,
            route_id        STRING,
            trip_id         STRING,
            latitude        DOUBLE,
            longitude       DOUBLE,
            bearing         DOUBLE,
            speed_mps       DOUBLE,
            current_status  STRING,
            feed            STRING,
            `timestamp`     BIGINT,
            ingested_at     BIGINT,
            event_ts        AS TO_TIMESTAMP_LTZ(`timestamp`, 3),
            WATERMARK FOR event_ts AS event_ts - INTERVAL '{WATERMARK_LAG_SEC}' SECOND
        ) WITH (
            'connector'                    = 'kafka',
            'topic'                        = '{os.getenv("TOPIC_VEHICLE_POSITIONS", "vehicle-positions")}',
            'properties.bootstrap.servers' = '{FLINK_KAFKA_BOOTSTRAP}',
            'properties.group.id'          = 'flink-iceberg-sink',
            'scan.startup.mode'            = 'earliest-offset',
            'format'                       = 'json',
            'json.ignore-parse-errors'     = 'true'
        )
    """)

    # ── Sink: write to Iceberg Bronze table ───────────────────────────────────
    # CAST(event_ts AS DATE) as string for partition value
    table_result = t_env.execute_sql("""
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
            DATE_FORMAT(event_ts, 'yyyy-MM-dd')  AS ingestion_date,
            ingested_at
        FROM kafka_vehicle_positions
        WHERE vehicle_id IS NOT NULL
          AND latitude  IS NOT NULL
          AND longitude IS NOT NULL
    """)

    # For remote execution — get job ID and return
    job_client = table_result.get_job_client()
    if job_client:
        job_id = job_client.get_job_id()
        print(f"\n✅ Iceberg sink job submitted!")
        print(f"   Job ID  : {job_id}")
        print(f"   Monitor : http://localhost:8082/#/job/{job_id}/overview")
        print(f"   Storage : {ICEBERG_WAREHOUSE}/bronze/vehicle_positions_raw/")
        print(f"   MinIO   : http://localhost:9001")
        print(f"\n   Data will appear in MinIO after the first checkpoint (~60s)")
    else:
        print("\n✅ Job submitted. Check http://localhost:8082")


if __name__ == "__main__":
    print("Starting Iceberg Bronze Sink Job")
    print(f"  Warehouse : {ICEBERG_WAREHOUSE}")
    print(f"  Kafka     : {FLINK_KAFKA_BOOTSTRAP}")
    print(f"  MinIO     : {AWS_ENDPOINT}\n")
    build_iceberg_sink()