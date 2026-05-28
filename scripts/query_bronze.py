"""
Query Bronze Iceberg Layer via DuckDB
─────────────────────────────────────────────────────────────────────────────
Reads Parquet files directly from MinIO using DuckDB's httpfs extension.
No Spark, no PyIceberg, no catalog needed — runs in seconds locally.

Usage:
    python3 scripts/query_bronze.py
    python3 scripts/query_bronze.py --feed bmtc
    python3 scripts/query_bronze.py --date 2026-05-28
    python3 scripts/query_bronze.py --feed namma_metro --limit 20
"""

from __future__ import annotations

import argparse
import os
from dotenv import load_dotenv

load_dotenv()

MINIO_ENDPOINT    = os.getenv("AWS_ENDPOINT", "http://localhost:9000").replace("http://", "")
ACCESS_KEY        = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
SECRET_KEY        = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
BUCKET            = os.getenv("MINIO_BUCKET", "transit-twin-local")
PARQUET_GLOB      = f"s3://{BUCKET}/lakehouse/bronze/vehicle_positions_raw_*/data/**/*.parquet"


def get_conn():
    try:
        import duckdb
    except ImportError:
        raise ImportError("Run: pip install duckdb")

    conn = duckdb.connect()
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    conn.execute(f"""
        SET s3_endpoint='{MINIO_ENDPOINT}';
        SET s3_access_key_id='{ACCESS_KEY}';
        SET s3_secret_access_key='{SECRET_KEY}';
        SET s3_use_ssl=false;
        SET s3_url_style='path';
    """)
    return conn


def query_bronze(
    date_filter: str | None = None,
    feed_filter: str | None = None,
    limit: int = 10,
) -> None:
    conn = get_conn()

    where_clauses = []
    if date_filter:
        where_clauses.append(f"ingestion_date = '{date_filter}'")
    if feed_filter:
        where_clauses.append(f"feed = '{feed_filter}'")
    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    print(f"\n{'─'*60}")
    print(f"  Bronze layer: {PARQUET_GLOB}")
    print(f"{'─'*60}\n")

    # ── Summary by feed and date ──────────────────────────────────────────────
    print("── Row counts by feed + date ────────────────────────────────")
    result = conn.execute(f"""
        SELECT
            feed,
            ingestion_date,
            COUNT(*)                        AS row_count,
            COUNT(DISTINCT vehicle_id)      AS unique_vehicles,
            ROUND(AVG(speed_mps), 2)        AS avg_speed_mps,
            MIN(event_ts)                   AS earliest,
            MAX(event_ts)                   AS latest
        FROM read_parquet('{PARQUET_GLOB}')
        {where_sql}
        GROUP BY feed, ingestion_date
        ORDER BY ingestion_date DESC, feed
    """).df()
    print(result.to_string(index=False))

    # ── Total summary ─────────────────────────────────────────────────────────
    print("\n── Total summary ────────────────────────────────────────────")
    total = conn.execute(f"""
        SELECT
            COUNT(*)                    AS total_rows,
            COUNT(DISTINCT vehicle_id)  AS total_vehicles,
            COUNT(DISTINCT feed)        AS feeds,
            COUNT(DISTINCT route_id)    AS unique_routes,
            MIN(event_ts)               AS pipeline_start,
            MAX(event_ts)               AS pipeline_latest
        FROM read_parquet('{PARQUET_GLOB}')
        {where_sql}
    """).df()
    print(total.to_string(index=False))

    # ── Sample rows ───────────────────────────────────────────────────────────
    print(f"\n── Sample rows (limit {limit}) ──────────────────────────────")
    sample = conn.execute(f"""
        SELECT
            vehicle_id,
            route_id,
            ROUND(latitude, 4)   AS lat,
            ROUND(longitude, 4)  AS lon,
            ROUND(speed_mps, 1)  AS speed_mps,
            current_status,
            feed,
            event_ts
        FROM read_parquet('{PARQUET_GLOB}')
        {where_sql}
        ORDER BY event_ts DESC
        LIMIT {limit}
    """).df()
    print(sample.to_string(index=False))

    print(f"\n✅ Query complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query Bronze Iceberg layer via DuckDB")
    parser.add_argument("--date",  help="Filter by ingestion_date (YYYY-MM-DD)")
    parser.add_argument("--feed",  help="Filter by feed name (bmtc / namma_metro)")
    parser.add_argument("--limit", type=int, default=10, help="Sample row limit")
    args = parser.parse_args()
    query_bronze(date_filter=args.date, feed_filter=args.feed, limit=args.limit)