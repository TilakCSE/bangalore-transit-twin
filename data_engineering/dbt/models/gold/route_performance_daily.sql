-- models/gold/route_performance_daily.sql
-- ─────────────────────────────────────────────────────────────────────────────
-- Gold layer: daily route-level performance KPIs.
-- Consumed by: Unity HUD overlay, REST API /api/v1/routes, LLM RAG agent.
--
-- Grain: one row per (route_id, service_date, feed_name)
-- Tags: gold, performance, kpi
-- ─────────────────────────────────────────────────────────────────────────────

{{ config(
    materialized='table',
    partition_by={'field': 'service_date', 'data_type': 'date'},
    tags=['gold', 'performance', 'kpi']
) }}

WITH base AS (
    SELECT
        route_id,
        feed_name,
        service_date,
        time_of_day_bucket,
        punctuality_status,
        actual_arrival_delay_sec,
        trip_id
    FROM {{ ref('stop_time_actuals') }}
    WHERE service_date >= DATEADD('day', -90, CURRENT_DATE)  -- rolling 90-day window
      AND route_id IS NOT NULL
),

aggregated AS (
    SELECT
        route_id,
        feed_name,
        service_date,
        -- Volume
        COUNT(DISTINCT trip_id)                                          AS total_trips,
        COUNT(*)                                                         AS total_stop_observations,
        -- Punctuality KPIs
        ROUND(
            100.0 * SUM(CASE WHEN punctuality_status = 'ON_TIME' THEN 1 ELSE 0 END)
            / NULLIF(COUNT(*), 0), 2
        )                                                                AS on_time_pct,
        ROUND(
            100.0 * SUM(CASE WHEN punctuality_status = 'DELAYED' THEN 1 ELSE 0 END)
            / NULLIF(COUNT(*), 0), 2
        )                                                                AS delayed_pct,
        -- Delay statistics (in seconds)
        ROUND(AVG(actual_arrival_delay_sec), 1)                         AS avg_delay_sec,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP
            (ORDER BY actual_arrival_delay_sec), 1)                     AS median_delay_sec,
        ROUND(PERCENTILE_CONT(0.9) WITHIN GROUP
            (ORDER BY actual_arrival_delay_sec), 1)                     AS p90_delay_sec,
        MAX(actual_arrival_delay_sec)                                   AS max_delay_sec,
        -- Peak vs off-peak breakdown
        ROUND(AVG(CASE WHEN time_of_day_bucket = 'AM_PEAK'
            THEN actual_arrival_delay_sec END), 1)                      AS avg_delay_am_peak_sec,
        ROUND(AVG(CASE WHEN time_of_day_bucket = 'PM_PEAK'
            THEN actual_arrival_delay_sec END), 1)                      AS avg_delay_pm_peak_sec,
        -- Reliability score (0–100): lower delay variance = higher score
        GREATEST(0, ROUND(
            100 - STDDEV(actual_arrival_delay_sec) / 60.0, 1
        ))                                                               AS reliability_score
    FROM base
    GROUP BY route_id, feed_name, service_date
)

SELECT
    *,
    -- 7-day rolling average on-time % for trend lines
    ROUND(AVG(on_time_pct) OVER (
        PARTITION BY route_id, feed_name
        ORDER BY service_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ), 2)                                                               AS on_time_pct_7d_avg,
    CURRENT_TIMESTAMP                                                   AS _dbt_updated_at
FROM aggregated
ORDER BY service_date DESC, on_time_pct ASC