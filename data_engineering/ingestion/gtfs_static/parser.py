"""
GTFS Static Feed Downloader & Parser
─────────────────────────────────────────────────────────────────────────────
Downloads BMTC / Namma Metro GTFS ZIP files, parses the standard text files,
and returns typed Pandas DataFrames ready for Bronze Iceberg loading.

Used by:
  - The Airflow DAG (gtfs_static_ingestion.py)
  - Local notebooks for EDA
  - The dbt seed loader (for small reference tables)

GTFS spec reference: https://gtfs.org/documentation/schedule/reference/
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

log = structlog.get_logger(__name__)

# Bangalore bounding box for coordinate validation
BLR_BBOX = {"lat_min": 12.6, "lat_max": 13.3, "lon_min": 77.2, "lon_max": 77.9}

# Canonical dtypes for each GTFS table
GTFS_DTYPES: dict[str, dict[str, str]] = {
    "agency": {"agency_id": "str", "agency_name": "str", "agency_url": "str",
               "agency_timezone": "str", "agency_lang": "str"},
    "routes": {"route_id": "str", "agency_id": "str", "route_short_name": "str",
               "route_long_name": "str", "route_type": "Int64",
               "route_color": "str", "route_text_color": "str"},
    "trips": {"route_id": "str", "service_id": "str", "trip_id": "str",
              "trip_headsign": "str", "direction_id": "Int64", "shape_id": "str"},
    "stops": {"stop_id": "str", "stop_name": "str", "stop_lat": "float64",
              "stop_lon": "float64", "location_type": "Int64", "parent_station": "str"},
    "stop_times": {"trip_id": "str", "arrival_time": "str", "departure_time": "str",
                   "stop_id": "str", "stop_sequence": "Int64",
                   "pickup_type": "Int64", "drop_off_type": "Int64"},
    "calendar": {"service_id": "str", "monday": "Int64", "tuesday": "Int64",
                 "wednesday": "Int64", "thursday": "Int64", "friday": "Int64",
                 "saturday": "Int64", "sunday": "Int64",
                 "start_date": "str", "end_date": "str"},
    "calendar_dates": {"service_id": "str", "date": "str", "exception_type": "Int64"},
    "shapes": {"shape_id": "str", "shape_pt_lat": "float64",
               "shape_pt_lon": "float64", "shape_pt_sequence": "Int64"},
}


@dataclass
class GTFSFeed:
    feed_name: str
    tables: dict[str, pd.DataFrame]
    raw_zip_bytes: bytes

    def __repr__(self) -> str:
        summary = {k: len(v) for k, v in self.tables.items()}
        return f"GTFSFeed(feed={self.feed_name}, tables={summary})"


@retry(
    wait=wait_exponential(multiplier=1, min=3, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def download_gtfs_zip(url: str, timeout: int = 60) -> bytes:
    """Download GTFS ZIP from URL with retry on transient failures."""
    log.info("Downloading GTFS feed", url=url)
    response = requests.get(url, timeout=timeout, stream=True)
    response.raise_for_status()
    content = response.content
    log.info("Downloaded GTFS feed", size_kb=len(content) // 1024)
    return content


def parse_gtfs_zip(
    feed_name: str,
    zip_bytes: bytes,
    tables_to_load: list[str] | None = None,
) -> GTFSFeed:
    """
    Parse a GTFS ZIP from bytes into a GTFSFeed of typed DataFrames.

    Args:
        feed_name: Identifier like 'bmtc' or 'namma_metro'
        zip_bytes:  Raw bytes of the downloaded ZIP
        tables_to_load: Subset of GTFS tables to parse (None = all)

    Returns:
        GTFSFeed with one DataFrame per parsed table
    """
    target_tables = tables_to_load or list(GTFS_DTYPES.keys())
    parsed: dict[str, pd.DataFrame] = {}

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        available = {name.replace(".txt", "") for name in zf.namelist()}
        log.info("GTFS ZIP contents", feed=feed_name, available=sorted(available))

        for table_name in target_tables:
            filename = f"{table_name}.txt"
            if filename not in zf.namelist():
                log.warning("GTFS table missing from feed", table=table_name, feed=feed_name)
                continue

            raw_csv = zf.read(filename).decode("utf-8-sig")  # handle BOM
            df = pd.read_csv(
                io.StringIO(raw_csv),
                dtype="str",      # read everything as str first to avoid silent coercions
                keep_default_na=False,
                na_values=[""],
            )
            df.columns = df.columns.str.strip()  # BMTC has trailing spaces in headers

            # Apply canonical dtypes
            for col, dtype in GTFS_DTYPES.get(table_name, {}).items():
                if col in df.columns:
                    try:
                        df[col] = df[col].astype(dtype)
                    except (ValueError, TypeError) as exc:
                        log.warning("Dtype coercion failed", col=col, dtype=dtype, error=str(exc))

            # Add feed metadata columns
            df["_feed_name"] = feed_name
            parsed[table_name] = df
            log.info("Parsed GTFS table", table=table_name, rows=len(df), feed=feed_name)

    return GTFSFeed(feed_name=feed_name, tables=parsed, raw_zip_bytes=zip_bytes)


def validate_stops_bbox(feed: GTFSFeed) -> list[str]:
    """
    Validate that stop coordinates fall within Bangalore bounding box.
    Returns list of violation messages (empty = all good).
    """
    violations: list[str] = []
    stops = feed.tables.get("stops")
    if stops is None:
        return ["stops table missing"]

    out_of_bbox = stops[
        (stops["stop_lat"] < BLR_BBOX["lat_min"])
        | (stops["stop_lat"] > BLR_BBOX["lat_max"])
        | (stops["stop_lon"] < BLR_BBOX["lon_min"])
        | (stops["stop_lon"] > BLR_BBOX["lon_max"])
    ]
    if len(out_of_bbox) > 0:
        violations.append(
            f"{len(out_of_bbox)} stops outside Bangalore bbox in feed '{feed.feed_name}': "
            f"{out_of_bbox['stop_id'].tolist()[:10]}"
        )
    return violations


def load_gtfs_from_disk(feed_name: str, zip_path: str | Path) -> GTFSFeed:
    """Convenience: load GTFS from a local ZIP file (for testing / offline dev)."""
    with open(zip_path, "rb") as f:
        return parse_gtfs_zip(feed_name, f.read())