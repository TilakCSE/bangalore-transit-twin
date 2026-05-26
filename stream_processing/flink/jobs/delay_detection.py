"""
Flink Delay Detection & Bunching Job
─────────────────────────────────────────────────────────────────────────────
Submits to the running Docker Flink cluster (not local MiniCluster).
JobManager REST API: localhost:8081 (mapped to 8082 on host)

Run:
    python3 -m stream_processing.flink.jobs.delay_detection
"""

from __future__ import annotations

import os

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import EnvironmentSettings, StreamTableEnvironment
from pyflink.table.udf import udf
from pyflink.table import DataTypes
from pyflink.common import Configuration

# ── Constants ─────────────────────────────────────────────────────────────────
FLINK_KAFKA_BOOTSTRAP = os.getenv("FLINK_KAFKA_BOOTSTRAP", "localhost:9092")
DELAY_THRESHOLD_SEC   = 120
BUNCHING_DIST_M       = 500
BUNCHING_WINDOW_SEC   = 90
WATERMARK_LAG_SEC     = 60

JAR_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../lib/flink-sql-connector-kafka-3.0.2-1.18.jar")
)


def haversine_distance_udf():
    import math

    @udf(result_type=DataTypes.DOUBLE())
    def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6_371_000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = (math.sin(dphi / 2) ** 2
             + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return haversine


def build_pipeline() -> None:
    # ── Connect to the Docker Flink cluster ───────────────────────────────────
    # localhost:8081 is the JobManager REST port (mapped from container's 8081)
    
    config = Configuration()
    config.set_string("rest.address", "localhost")
    config.set_integer("rest.port", 8083)  # Matches your UI port
    config.set_string("pipeline.jars", f"file://{JAR_PATH}")
    env = StreamExecutionEnvironment.get_execution_environment(config)
    env.set_parallelism(2) # 2 queries × 2 parallelism = 4 slots (fits in 8)
    env.enable_checkpointing(30_000)
    env.get_checkpoint_config().set_checkpoint_storage_dir(
        os.getenv("FLINK_CHECKPOINT_DIR", "file:///tmp/flink-checkpoints")
    )

    settings = EnvironmentSettings.new_instance().in_streaming_mode().build()
    t_env = StreamTableEnvironment.create(env, settings)

    # ── Register Haversine UDF ────────────────────────────────────────────────
    t_env.create_temporary_function("haversine", haversine_distance_udf())

    # ── Source: vehicle positions ─────────────────────────────────────────────
    t_env.execute_sql(f"""
        CREATE TABLE vehicle_positions (
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
            event_ts        AS TO_TIMESTAMP_LTZ(`timestamp`, 3),
            WATERMARK FOR event_ts AS event_ts - INTERVAL '{WATERMARK_LAG_SEC}' SECOND
        ) WITH (
            'connector'                           = 'kafka',
            'topic'                               = '{os.getenv("TOPIC_VEHICLE_POSITIONS", "vehicle-positions")}',
            'properties.bootstrap.servers'        = '{FLINK_KAFKA_BOOTSTRAP}',
            'properties.group.id'                 = 'flink-delay-detector',
            'scan.startup.mode'                   = 'latest-offset',
            'format'                              = 'json',
            'json.ignore-parse-errors'            = 'true'
        )
    """)

    # ── Source: trip updates ──────────────────────────────────────────────────
    t_env.execute_sql(f"""
        CREATE TABLE trip_updates (
            trip_id         STRING,
            route_id        STRING,
            vehicle_id      STRING,
            stop_id         STRING,
            arrival_delay   INT,
            departure_delay INT,
            `timestamp`     BIGINT,
            event_ts        AS TO_TIMESTAMP_LTZ(`timestamp`, 3),
            WATERMARK FOR event_ts AS event_ts - INTERVAL '{WATERMARK_LAG_SEC}' SECOND
        ) WITH (
            'connector'                           = 'kafka',
            'topic'                               = '{os.getenv("TOPIC_TRIP_UPDATES", "trip-updates")}',
            'properties.bootstrap.servers'        = '{FLINK_KAFKA_BOOTSTRAP}',
            'properties.group.id'                 = 'flink-trip-update-reader',
            'scan.startup.mode'                   = 'latest-offset',
            'format'                              = 'json',
            'json.ignore-parse-errors'            = 'true'
        )
    """)

    # ── Sink: delay events → Kafka ────────────────────────────────────────────
    t_env.execute_sql(f"""
        CREATE TABLE delay_events (
            trip_id     STRING,
            route_id    STRING,
            vehicle_id  STRING,
            delay_sec   INT,
            alert_type  STRING,
            event_ts    TIMESTAMP(3)
        ) WITH (
            'connector'                           = 'kafka',
            'topic'                               = '{os.getenv("TOPIC_FLINK_DELAYS", "flink-delay-output")}',
            'properties.bootstrap.servers'        = '{FLINK_KAFKA_BOOTSTRAP}',
            'format'                              = 'json'
        )
    """)

    # ── Sink: bunching alerts → Kafka ─────────────────────────────────────────
    t_env.execute_sql(f"""
        CREATE TABLE bunching_alerts (
            route_id        STRING,
            vehicle_id_a    STRING,
            vehicle_id_b    STRING,
            distance_m      DOUBLE,
            window_start    TIMESTAMP(3),
            window_end      TIMESTAMP(3)
        ) WITH (
            'connector'                           = 'kafka',
            'topic'                               = '{os.getenv("TOPIC_FLINK_BUNCHING", "flink-bunching-alerts")}',
            'properties.bootstrap.servers'        = '{FLINK_KAFKA_BOOTSTRAP}',
            'format'                              = 'json'
        )
    """)

    # ── StatementSet: submit both queries as one job ──────────────────────────
    statement_set = t_env.create_statement_set()

    statement_set.add_insert_sql(f"""
        INSERT INTO delay_events
        SELECT
            tu.trip_id,
            tu.route_id,
            vp.vehicle_id,
            tu.arrival_delay                                        AS delay_sec,
            CASE
                WHEN tu.arrival_delay >  {DELAY_THRESHOLD_SEC} THEN 'DELAY'
                WHEN tu.arrival_delay < -{DELAY_THRESHOLD_SEC} THEN 'EARLY'
                ELSE 'ON_TIME'
            END                                                     AS alert_type,
            vp.event_ts
        FROM vehicle_positions vp
        JOIN trip_updates tu
            ON  vp.trip_id   = tu.trip_id
            AND vp.event_ts BETWEEN tu.event_ts - INTERVAL '2' MINUTE
                                AND tu.event_ts + INTERVAL '2' MINUTE
        WHERE ABS(tu.arrival_delay) > {DELAY_THRESHOLD_SEC}
    """)

    statement_set.add_insert_sql(f"""
        INSERT INTO bunching_alerts
        SELECT
            a.route_id,
            a.vehicle_id                                                        AS vehicle_id_a,
            b.vehicle_id                                                        AS vehicle_id_b,
            haversine(a.latitude, a.longitude, b.latitude, b.longitude)        AS distance_m,
            a.event_ts                                                          AS window_start,
            b.event_ts                                                          AS window_end
        FROM vehicle_positions a
        JOIN vehicle_positions b
            ON  a.route_id   = b.route_id
            AND a.vehicle_id < b.vehicle_id
            AND a.event_ts BETWEEN b.event_ts - INTERVAL '{BUNCHING_WINDOW_SEC}' SECOND
                               AND b.event_ts + INTERVAL '{BUNCHING_WINDOW_SEC}' SECOND
        WHERE haversine(a.latitude, a.longitude, b.latitude, b.longitude) < {BUNCHING_DIST_M}
    """)

    # ── Submit ────────────────────────────────────────────────────────────────
    # execute() submits to the remote cluster and returns immediately.
    # The job then runs indefinitely on the cluster — this Python process can exit.
    # Do NOT call .wait() — that blocks until the job finishes (never, for streaming).
    # Do NOT call env.execute() after this — that's DataStream API only.
    print(f"\n  Kafka bootstrap : {FLINK_KAFKA_BOOTSTRAP}")
    print(f"  JAR             : {JAR_PATH}")
    print(f"  Delay threshold : {DELAY_THRESHOLD_SEC}s")
    print(f"  Bunching dist   : {BUNCHING_DIST_M}m\n")
    print("Submitting job to Flink cluster...")

    table_result = statement_set.execute()

    # For remote execution, get_job_client() tells us the assigned job ID
    job_client = table_result.get_job_client()
    if job_client:
        job_id = job_client.get_job_id()
        print(f"\n✅ Job submitted successfully!")
        print(f"   Job ID : {job_id}")
        print(f"   Monitor: http://localhost:8082/#/job/{job_id}/overview")
    else:
        print("\n✅ Job submitted. Check http://localhost:8083 for status.")
        
    print("\n⏳ Keeping MiniCluster alive... Press Ctrl+C to stop.")
    table_result.wait()


if __name__ == "__main__":
    build_pipeline()