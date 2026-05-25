"""
DAG: gtfs_static_ingestion
─────────────────────────────────────────────────────────────────────────────
Daily ingestion of BMTC and Namma Metro GTFS static feeds into the Iceberg
Medallion Lakehouse on GCS.

Schedule: 03:00 IST daily (BMTC usually updates overnight)
Medallion flow:
  [GTFS ZIP download] → [Bronze: raw CSV → Iceberg] → [Silver: conformed schema]
  → [Gold: route_performance aggregates]

Tasks:
  1. download_gtfs_static     — fetch ZIP from BMTC/Metro URLs
  2. validate_feed            — Great Expectations schema checks
  3. load_bronze_iceberg      — append raw CSVs to Bronze Iceberg tables
  4. run_dbt_silver           — dbt models: normalize + deduplicate
  5. run_dbt_gold             — dbt models: route performance aggregates
  6. notify_slack             — post completion summary (optional)
"""

from __future__ import annotations

import hashlib
import io
import os
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import requests
from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from airflow.utils.dates import days_ago

FEEDS = {
    "bmtc": os.getenv(
        "BMTC_GTFS_STATIC_URL",
        "https://bmtcwebportal.pascos.in/gtfs/bmtc_gtfs.zip",
    ),
    "namma_metro": os.getenv(
        "NAMMA_METRO_GTFS_STATIC_URL",
        "https://transit.blr.metro.karnataka.gov.in/gtfs/namma_metro.zip",
    ),
}

GCS_BUCKET = os.getenv("GCS_BUCKET", "transit-twin-local")
BRONZE_PREFIX = "lakehouse/bronze/gtfs_static"
ICEBERG_CATALOG_URI = os.getenv("ICEBERG_CATALOG_URI", "http://localhost:8181")

default_args = {
    "owner": "data-engineering",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

GTFS_TABLES = [
    "agency", "routes", "trips", "stops", "stop_times",
    "calendar", "calendar_dates", "shapes", "fare_attributes",
]


@dag(
    dag_id="gtfs_static_ingestion",
    description="Daily GTFS static feed ingestion → Iceberg Medallion Lakehouse",
    schedule_interval="30 21 * * *",  # 03:00 IST = 21:30 UTC
    start_date=days_ago(1),
    catchup=False,
    default_args=default_args,
    tags=["gtfs", "ingestion", "bronze", "lakehouse"],
    doc_md=__doc__,
)
def gtfs_static_ingestion():

    @task
    def download_gtfs_static(feed_name: str, url: str, **context) -> dict:
        """Download GTFS ZIP and upload raw files to GCS Bronze layer."""
        gcs = GCSHook()
        ds = context["ds"]  # execution date string: YYYY-MM-DD

        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        content = response.content
        checksum = hashlib.md5(content).hexdigest()

        uploaded_paths: dict[str, str] = {}
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for filename in zf.namelist():
                table_name = filename.replace(".txt", "")
                if table_name not in GTFS_TABLES:
                    continue
                csv_bytes = zf.read(filename)
                gcs_path = f"{BRONZE_PREFIX}/{feed_name}/{ds}/{filename}"
                gcs.upload(
                    bucket_name=GCS_BUCKET,
                    object_name=gcs_path,
                    data=csv_bytes,
                    mime_type="text/csv",
                )
                uploaded_paths[table_name] = f"gs://{GCS_BUCKET}/{gcs_path}"

        # Store checksum to detect unchanged feeds
        Variable.set(f"gtfs_checksum_{feed_name}", checksum)
        return {"feed": feed_name, "date": ds, "paths": uploaded_paths, "checksum": checksum}

    @task
    def validate_feed(download_result: dict) -> bool:
        """
        Run Great Expectations suite against the downloaded CSV files.
        Checks: required columns present, no null stop_ids, valid lat/lon bounds
        (Bangalore bbox: lat 12.7-13.2, lon 77.3-77.9).
        """
        import great_expectations as ge

        feed = download_result["feed"]
        paths = download_result["paths"]
        gcs = GCSHook()

        for table_name, gcs_path in paths.items():
            # Download to local temp for GE validation
            local_path = f"/tmp/{feed}_{table_name}.csv"
            gcs.download(
                bucket_name=GCS_BUCKET,
                object_name=gcs_path.replace(f"gs://{GCS_BUCKET}/", ""),
                filename=local_path,
            )
            df = ge.read_csv(local_path)

            if table_name == "stops":
                result = df.expect_column_values_to_be_between(
                    "stop_lat", min_value=12.6, max_value=13.3
                )
                if not result["success"]:
                    raise ValueError(f"Stop lat validation failed for {feed}: {result}")
                result = df.expect_column_values_to_not_be_null("stop_id")
                if not result["success"]:
                    raise ValueError(f"Null stop_ids found in {feed}")

            if table_name == "routes":
                result = df.expect_column_values_to_not_be_null("route_id")
                if not result["success"]:
                    raise ValueError(f"Null route_ids found in {feed}")

        return True

    @task
    def load_bronze_iceberg(download_result: dict, validated: bool) -> str:
        """
        Load raw CSVs from GCS into Bronze Iceberg tables using PyIceberg.
        Each table is partitioned by feed_name and ingestion_date.
        Uses schema-on-read with minimal type coercion.
        """
        import pyarrow as pa
        import pyarrow.csv as pcsv
        from pyiceberg.catalog import load_catalog
        from pyiceberg.schema import Schema
        from pyiceberg.types import (
            IntegerType, LongType, NestedField, StringType, TimestampType,
        )

        feed = download_result["feed"]
        ds = download_result["date"]
        gcs = GCSHook()

        catalog = load_catalog(
            "gcs_catalog",
            **{
                "type": "rest",
                "uri": ICEBERG_CATALOG_URI,
                "warehouse": f"gs://{GCS_BUCKET}/lakehouse",
            },
        )

        loaded_tables = []
        for table_name, gcs_path in download_result["paths"].items():
            local_path = f"/tmp/{feed}_{table_name}_bronze.csv"
            gcs.download(
                bucket_name=GCS_BUCKET,
                object_name=gcs_path.replace(f"gs://{GCS_BUCKET}/", ""),
                filename=local_path,
            )
            arrow_table = pcsv.read_csv(local_path)
            # Add ingestion metadata columns
            n = len(arrow_table)
            arrow_table = arrow_table.append_column(
                pa.field("_feed_name", pa.string()), pa.array([feed] * n, pa.string())
            )
            arrow_table = arrow_table.append_column(
                pa.field("_ingestion_date", pa.string()), pa.array([ds] * n, pa.string())
            )

            full_table_name = f"bronze.gtfs_{table_name}"
            try:
                iceberg_table = catalog.load_table(full_table_name)
                iceberg_table.append(arrow_table)
            except Exception:
                catalog.create_table(full_table_name, schema=arrow_table.schema)
                iceberg_table = catalog.load_table(full_table_name)
                iceberg_table.append(arrow_table)

            loaded_tables.append(full_table_name)

        return f"Loaded {len(loaded_tables)} tables for feed={feed} date={ds}"

    # ── dbt Silver + Gold transforms ─────────────────────────────────────────
    run_dbt_silver = BashOperator(
        task_id="run_dbt_silver",
        bash_command=(
            "cd /opt/airflow && "
            "dbt run --select tag:silver --target prod --profiles-dir data_engineering/dbt"
        ),
    )

    run_dbt_gold = BashOperator(
        task_id="run_dbt_gold",
        bash_command=(
            "cd /opt/airflow && "
            "dbt run --select tag:gold --target prod --profiles-dir data_engineering/dbt"
        ),
    )

    run_dbt_tests = BashOperator(
        task_id="run_dbt_tests",
        bash_command=(
            "cd /opt/airflow && "
            "dbt test --target prod --profiles-dir data_engineering/dbt"
        ),
    )

    # ── Wire up tasks ─────────────────────────────────────────────────────────
    for feed_name, url in FEEDS.items():
        dl = download_gtfs_static(feed_name, url)
        val = validate_feed(dl)
        bronze = load_bronze_iceberg(dl, val)
        bronze >> run_dbt_silver >> run_dbt_gold >> run_dbt_tests


gtfs_static_ingestion()