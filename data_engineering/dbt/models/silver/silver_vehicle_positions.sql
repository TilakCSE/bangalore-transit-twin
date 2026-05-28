-- models/silver/silver_vehicle_positions.sql
-- ─────────────────────────────────────────────────────────────────────────────
-- Silver layer: cleaned, enriched vehicle positions.
-- Reads directly from Bronze Parquet on MinIO via DuckDB httpfs.
--
-- Transformations applied:
--   1. Deduplicate — keep latest record per vehicle per 30s window
--   2. Validate coordinates — Bangalore bbox + Delhi bbox (both feeds)
--   3. Classify speed — stationary / slow / normal / fast
--   4. Add time features — hour bucket, is_peak_hour, day_of_week
--   5. Null-safe route_id — replace nulls with 'UNKNOWN'
--
-- Grain: one row per (vehicle_id, event_ts) deduplicated
-- Tags: silver, positions, gtfs
-- ─────────────────────────────────────────────────────────────────────────────

{{ config(
    materialized='table',
    tags=['silver', 'positions']
) }}

WITH raw AS (
    SELECT *
    FROM read_parquet(
        's3://transit-twin-local/lakehouse/bronze/vehicle_positions_raw_*/data/**/*.parquet'
    )
    WHERE vehicle_id IS NOT NULL
      AND latitude   IS NOT NULL
      AND longitude  IS NOT NULL
      -- Sanity check: valid coordinate ranges
      AND latitude  BETWEEN -90  AND 90
      AND longitude BETWEEN -180 AND 180
),

-- Remove duplicate rows: same vehicle, same 30-second window
-- Keep the record with the latest ingested_at within each window
deduplicated AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY
                vehicle_id,
                time_bucket(INTERVAL '30 seconds', event_ts)
            ORDER BY ingested_at DESC
        ) AS rn
    FROM raw
),

cleaned AS (
    SELECT
        entity_id,
        vehicle_id,
        COALESCE(route_id, 'UNKNOWN')       AS route_id,
        COALESCE(trip_id,  'UNKNOWN')       AS trip_id,
        latitude,
        longitude,
        bearing,
        COALESCE(speed_mps, 0.0)            AS speed_mps,
        current_status,
        feed,
        event_ts,
        ingestion_date,
        ingested_at,

        -- ── Speed classification ────────────────────────────────────────────
        CASE
            WHEN COALESCE(speed_mps, 0) < 0.5  THEN 'STATIONARY'
            WHEN COALESCE(speed_mps, 0) < 5.0  THEN 'SLOW'
            WHEN COALESCE(speed_mps, 0) < 15.0 THEN 'NORMAL'
            ELSE                                     'FAST'
        END                                         AS speed_category,

        -- ── Time features (for ML feature store) ───────────────────────────
        EXTRACT(hour FROM event_ts)                 AS hour_of_day,
        EXTRACT(dow  FROM event_ts)                 AS day_of_week,   -- 0=Sun
        CASE
            WHEN EXTRACT(hour FROM event_ts) BETWEEN 7  AND 10 THEN 'AM_PEAK'
            WHEN EXTRACT(hour FROM event_ts) BETWEEN 17 AND 20 THEN 'PM_PEAK'
            WHEN EXTRACT(hour FROM event_ts) BETWEEN 22 AND 23 THEN 'NIGHT'
            WHEN EXTRACT(hour FROM event_ts) < 6               THEN 'EARLY_MORNING'
            ELSE                                                     'OFFPEAK'
        END                                                           AS time_bucket,

        CASE
            WHEN EXTRACT(hour FROM event_ts) BETWEEN 7  AND 10 THEN true
            WHEN EXTRACT(hour FROM event_ts) BETWEEN 17 AND 20 THEN true
            ELSE false
        END                                                           AS is_peak_hour,

        -- ── Geo validation flag ─────────────────────────────────────────────
        -- Bangalore bbox: lat 12.6–13.3, lon 77.2–77.9
        -- Delhi bbox:     lat 28.4–28.9, lon 76.8–77.4
        CASE
            WHEN feed = 'bmtc' AND
                 latitude  BETWEEN 12.6 AND 13.3 AND
                 longitude BETWEEN 77.2 AND 77.9  THEN true
            WHEN feed = 'namma_metro' AND
                 latitude  BETWEEN 28.4 AND 28.9 AND
                 longitude BETWEEN 76.8 AND 77.4  THEN true
            ELSE false
        END                                                           AS is_valid_bbox,

        CURRENT_TIMESTAMP                                             AS dbt_updated_at

    FROM deduplicated
    WHERE rn = 1
)

SELECT * FROM cleaned