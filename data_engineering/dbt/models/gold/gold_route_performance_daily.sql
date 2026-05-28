-- models/gold/gold_route_performance_daily.sql
-- ─────────────────────────────────────────────────────────────────────────────
-- Gold layer: daily route-level KPIs.
-- Consumed by: FastAPI /api/v1/routes, Unity HUD overlay, LLM RAG agent.
--
-- Grain: one row per (route_id, feed, ingestion_date)
-- Tags: gold, performance, kpi
-- ─────────────────────────────────────────────────────────────────────────────

{{ config(
    materialized='table',
    tags=['gold', 'performance', 'kpi']
) }}

WITH silver AS (
    SELECT * FROM {{ ref('silver_vehicle_positions') }}
    WHERE is_valid_bbox = true
      AND route_id != 'UNKNOWN'
),

aggregated AS (
    SELECT
        route_id,
        feed,
        ingestion_date,

        -- ── Volume ───────────────────────────────────────────────────────────
        COUNT(DISTINCT vehicle_id)              AS unique_vehicles,
        COUNT(*)                                AS total_observations,

        -- ── Speed metrics ─────────────────────────────────────────────────────
        ROUND(AVG(speed_mps), 2)                AS avg_speed_mps,
        ROUND(AVG(speed_mps) * 3.6, 2)         AS avg_speed_kmh,
        ROUND(MAX(speed_mps) * 3.6, 2)         AS max_speed_kmh,
        ROUND(PERCENTILE_CONT(0.5)
            WITHIN GROUP (ORDER BY speed_mps)
            * 3.6, 2)                           AS median_speed_kmh,

        -- ── Congestion indicator ──────────────────────────────────────────────
        ROUND(
            100.0 * SUM(CASE WHEN speed_category = 'STATIONARY' THEN 1 ELSE 0 END)
            / NULLIF(COUNT(*), 0), 1
        )                                       AS stationary_pct,

        ROUND(
            100.0 * SUM(CASE WHEN speed_category = 'SLOW' THEN 1 ELSE 0 END)
            / NULLIF(COUNT(*), 0), 1
        )                                       AS slow_pct,

        -- ── Peak hour breakdown ───────────────────────────────────────────────
        ROUND(AVG(CASE WHEN time_bucket = 'AM_PEAK'
            THEN speed_mps END) * 3.6, 2)      AS avg_speed_am_peak_kmh,
        ROUND(AVG(CASE WHEN time_bucket = 'PM_PEAK'
            THEN speed_mps END) * 3.6, 2)      AS avg_speed_pm_peak_kmh,

        -- ── Activity window ───────────────────────────────────────────────────
        MIN(event_ts)                           AS first_seen,
        MAX(event_ts)                           AS last_seen,
        ROUND(
            DATEDIFF('minute', MIN(event_ts), MAX(event_ts)) / 60.0
        , 1)                                    AS active_hours,

        -- ── Reliability score (0–100) ─────────────────────────────────────────
        -- Higher = more consistent speed (less variance = more reliable service)
        GREATEST(0, ROUND(
            100 - (STDDEV(speed_mps) / NULLIF(AVG(speed_mps), 0)) * 50
        , 1))                                   AS reliability_score,

        CURRENT_TIMESTAMP                       AS dbt_updated_at

    FROM silver
    GROUP BY route_id, feed, ingestion_date
)

SELECT
    *,
    -- 7-day rolling average speed for trend lines in Unity
    ROUND(AVG(avg_speed_kmh) OVER (
        PARTITION BY route_id, feed
        ORDER BY ingestion_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ), 2)                                       AS avg_speed_kmh_7d,

    -- Rank routes by performance within each feed (1 = best)
    RANK() OVER (
        PARTITION BY feed, ingestion_date
        ORDER BY avg_speed_kmh DESC
    )                                           AS speed_rank_today

FROM aggregated
ORDER BY ingestion_date DESC, feed, avg_speed_kmh ASC