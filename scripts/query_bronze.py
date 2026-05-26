"""
Query Bronze Iceberg Table
─────────────────────────────────────────────────────────────────────────────
Uses DuckDB + PyIceberg to query the Bronze vehicle_positions_raw table
directly from MinIO. No Spark, no cluster needed — runs locally in seconds.

Usage:
    python3 scripts/query_bronze.py
    python3 scripts/query_bronze.py --date 2026-05-26
    python3 scripts/query_bronze.py --feed namma_metro --limit 20
"""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

load_dotenv()


def query_bronze(date_filter: str | None = None, feed_filter: str | None = None, limit: int = 10):
    try:
        import duckdb
        from pyiceberg.catalog import load_catalog
    except ImportError:
        print("Install dependencies: pip install duckdb pyiceberg[s3,pyarrow]")
        return

    # Connect to Iceberg catalog on MinIO
    catalog = load_catalog(
        "local",
        **{
            "type": "hadoop",
            "warehouse": os.getenv("ICEBERG_WAREHOUSE", "s3://transit-twin-local/lakehouse"),
            "s3.endpoint": os.getenv("AWS_ENDPOINT", "http://localhost:9000"),
            "s3.access-key-id": os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
            "s3.secret-access-key": os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"),
            "s3.path-style-access": "true",
        },
    )

    try:
        table = catalog.load_table("bronze.vehicle_positions_raw")
    except Exception as exc:
        print(f"❌ Could not load table: {exc}")
        print("   Has the Iceberg sink job run at least one checkpoint (~60s)?")
        return

    # Build scan with optional filters
    scan = table.scan(limit=limit)
    if date_filter:
        scan = table.scan(row_filter=f"ingestion_date = '{date_filter}'", limit=limit)

    # Convert to Arrow and query with DuckDB
    arrow_table = scan.to_arrow()
    conn = duckdb.connect()
    conn.register("bronze_vp", arrow_table)

    print(f"\n{'─'*60}")
    print(f"  Bronze table: bronze.vehicle_positions_raw")
    print(f"  Total rows in scan: {len(arrow_table)}")
    print(f"{'─'*60}\n")

    # Summary by feed
    print("── Rows by feed ─────────────────────────────────────────")
    result = conn.execute("""
        SELECT feed, COUNT(*) as row_count,
               MIN(event_ts) as earliest,
               MAX(event_ts) as latest
        FROM bronze_vp
        GROUP BY feed
        ORDER BY row_count DESC
    """).fetchdf()
    print(result.to_string(index=False))

    # Summary by date
    print("\n── Rows by ingestion_date ───────────────────────────────")
    result = conn.execute("""
        SELECT ingestion_date, feed, COUNT(*) as row_count
        FROM bronze_vp
        GROUP BY ingestion_date, feed
        ORDER BY ingestion_date DESC, row_count DESC
    """).fetchdf()
    print(result.to_string(index=False))

    # Sample rows
    feed_clause = f"AND feed = '{feed_filter}'" if feed_filter else ""
    print(f"\n── Sample rows {feed_clause} ─────────────────────────────")
    result = conn.execute(f"""
        SELECT vehicle_id, route_id, latitude, longitude,
               speed_mps, feed, event_ts
        FROM bronze_vp
        WHERE vehicle_id IS NOT NULL {feed_clause}
        LIMIT {limit}
    """).fetchdf()
    print(result.to_string(index=False))

    # Partition listing (shows what's on disk in MinIO)
    print("\n── Table partitions ─────────────────────────────────────")
    for partition in table.inspect.partitions().to_pydict().get("partition", []):
        print(f"  {partition}")

    print(f"\n✅ Query complete. {len(arrow_table)} rows scanned from Iceberg Bronze layer.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query Bronze Iceberg table")
    parser.add_argument("--date",  help="Filter by ingestion_date (YYYY-MM-DD)")
    parser.add_argument("--feed",  help="Filter by feed name (bmtc / namma_metro)")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    query_bronze(date_filter=args.date, feed_filter=args.feed, limit=args.limit)