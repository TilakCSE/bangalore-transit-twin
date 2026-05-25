-- models/silver/stop_time_actuals.sql
-- ─────────────────────────────────────────────────────────────────────────────
-- Silver layer: conformed stop_times table merging static GTFS schedule
-- with actual arrival times from historical GTFS-RT trip update archives.
--
-- Grain: one row per (trip_id, stop_sequence, service_date)
-- Sources: bronze.gtfs_stop_times (static schedule)
--          bronze.gtfs_rt_trip_updates (actual arrivals, archived from Kafka)
--
-- Partitioned by: service_date (for Iceberg time-travel efficiency)
-- Tags: silver, gtfs, schedule
-- ─────────────────────────────────────────────────────────────────────────────

{{ config(
    materialized='incremental',
    incremental_strategy='merge',
    unique_key=['trip_id', 'stop_id', 'service_date'],
    partition_by={'field': 'service_date', 'data_type': 'date'},
    tags=['silver', 'gtfs', 'schedule']
) }}

WITH raw_schedule AS (
    SELECT
        st.trip_id,
        st.stop_id,
        st.stop_sequence,
        st.arrival_time        AS scheduled_arrival_time,
        st.departure_time      AS scheduled_departure_time,
        st.pickup_type,
        st.drop_off_type,
        t.route_id,
        t.service_id,
        t.direction_id,
        t.shape_id,
        st._feed_name,
        st._ingestion_date
    FROM {{ source('bronze', 'gtfs_stop_times') }} st
    LEFT JOIN {{ source('bronze', 'gtfs_trips') }} t
        ON st.trip_id = t.trip_id
        AND st._feed_name = t._feed_name
    WHERE st.trip_id IS NOT NULL
      AND st.stop_id IS NOT NULL
),

actual_arrivals AS (
    -- Archived GTFS-RT trip updates (written by Flink → Iceberg sink)
    SELECT
        trip_id,
        stop_id,
        arrival_delay        AS actual_arrival_delay_sec,
        departure_delay      AS actual_departure_delay_sec,
        DATE(event_ts)       AS service_date,
        event_ts             AS actual_event_ts
    FROM {{ source('bronze', 'gtfs_rt_trip_updates_archive') }}
    WHERE trip_id IS NOT NULL

    {% if is_incremental() %}
        -- Only process new data since last run
        AND DATE(event_ts) > (SELECT MAX(service_date) FROM {{ this }})
    {% endif %}
),

joined AS (
    SELECT
        s.trip_id,
        s.stop_id,
        s.stop_sequence,
        s.route_id,
        s.service_id,
        s.direction_id,
        s.shape_id,
        s._feed_name                                        AS feed_name,
        COALESCE(a.service_date, CURRENT_DATE)              AS service_date,
        s.scheduled_arrival_time,
        s.scheduled_departure_time,
        a.actual_arrival_delay_sec,
        a.actual_departure_delay_sec,
        a.actual_event_ts,
        -- Derived: was this trip delayed?
        CASE
            WHEN a.actual_arrival_delay_sec > 120  THEN 'DELAYED'
            WHEN a.actual_arrival_delay_sec < -60  THEN 'EARLY'
            WHEN a.actual_arrival_delay_sec IS NULL THEN 'UNKNOWN'
            ELSE 'ON_TIME'
        END                                                  AS punctuality_status,
        -- Feature: time-of-day bucket for ML
        CASE
            WHEN CAST(SPLIT_PART(s.scheduled_arrival_time, ':', 1) AS INT) BETWEEN 6  AND 9  THEN 'AM_PEAK'
            WHEN CAST(SPLIT_PART(s.scheduled_arrival_time, ':', 1) AS INT) BETWEEN 17 AND 20 THEN 'PM_PEAK'
            WHEN CAST(SPLIT_PART(s.scheduled_arrival_time, ':', 1) AS INT) BETWEEN 22 AND 23 THEN 'NIGHT'
            WHEN CAST(SPLIT_PART(s.scheduled_arrival_time, ':', 1) AS INT) < 6        THEN 'EARLY_MORNING'
            ELSE 'OFFPEAK'
        END                                                  AS time_of_day_bucket,
        CURRENT_TIMESTAMP                                    AS _dbt_updated_at
    FROM raw_schedule s
    LEFT JOIN actual_arrivals a
        ON s.trip_id = a.trip_id
        AND s.stop_id = a.stop_id
)

SELECT * FROM joined