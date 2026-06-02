"""
Feature Engineering Pipeline v2
─────────────────────────────────────────────────────────────────────────────
Strategy: date-level aggregation instead of route-level sequences.

With sparse multi-day data (routes don't appear every day), route-level
lag features drop almost everything. Instead we aggregate to daily
network-level stats and build cross-route features per day.

This produces a rich, trainable dataset from just 3 days of data:
  - 1,550 routes × 3 days = 4,650 route-day samples
  - Each sample: route features for one day → predict speed
  - No lag requirement: each route-day is an independent sample

For production (30+ days): the sequence model kicks in automatically.

Run:
    python3 -m mlops_pipeline.features.feature_engineering
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

DBT_DB_PATH    = Path("data_engineering/dbt/transit_twin.duckdb")
FEATURE_OUTPUT = Path("mlops_pipeline/features/feature_store.parquet")
SEQ_LEN        = 7    # used when enough days exist
MIN_DAYS_FOR_SEQUENCES = 14  # switch to sequence mode after 2 weeks


def load_gold_table() -> pd.DataFrame:
    import duckdb
    if not DBT_DB_PATH.exists():
        raise FileNotFoundError(
            f"DuckDB not found: {DBT_DB_PATH}\n"
            "Run: cd data_engineering/dbt && dbt run --profiles-dir ."
        )
    conn = duckdb.connect(str(DBT_DB_PATH), read_only=True)
    df = conn.execute("""
        SELECT
            route_id,
            feed,
            ingestion_date,
            COALESCE(unique_vehicles, 0)        AS unique_vehicles,
            COALESCE(total_observations, 0)     AS total_observations,
            COALESCE(stationary_pct, 0)         AS stationary_pct,
            COALESCE(slow_pct, 0)               AS slow_pct,
            COALESCE(reliability_score, 0)      AS reliability_score,
            -- Derived targets with real variance
            LN(COALESCE(total_observations, 1) + 1)  AS log_observations,
            LN(COALESCE(unique_vehicles, 1) + 1)     AS log_vehicles
        FROM main_gold.gold_route_performance_daily
        WHERE total_observations IS NOT NULL
            AND total_observations > 0
            AND route_id != 'UNKNOWN'
            AND ingestion_date != '1970-01-21'
        ORDER BY route_id, feed, ingestion_date
    """).df()
    conn.close()

    df["ingestion_date"] = pd.to_datetime(df["ingestion_date"])
    unique_dates = sorted(df["ingestion_date"].unique())

    print(f"  Rows loaded    : {len(df):,}")
    print(f"  Feeds          : {df['feed'].unique().tolist()}")
    print(f"  Date range     : {df['ingestion_date'].min().date()} → {df['ingestion_date'].max().date()}")
    print(f"  Unique dates   : {len(unique_dates)}")
    print(f"  Unique routes  : {df['route_id'].nunique():,}")
    return df


def build_cross_sectional_features(df: pd.DataFrame):
    """
    Build one feature vector per (route, day).
    No lag requirement — works with any number of days.

    Features:
      - Route-level: speed stats, congestion, reliability
      - Network-level: daily averages (how does this route compare to network?)
      - Time: day of week, is_weekend encoded cyclically
      - Feed: encoded
    """
    df = df.copy()

    # ── Network-level daily stats (context for each route) ────────────────────
    daily_network = df.groupby(["feed", "ingestion_date"]).agg(
        network_total_routes = ("route_id",           "count"),
        network_stationary   = ("stationary_pct",     "mean"),
        network_avg_obs      = ("total_observations", "mean")
    ).reset_index()

    df = df.merge(daily_network, on=["feed", "ingestion_date"], how="left")


    # ── Time features ─────────────────────────────────────────────────────────
    df["day_of_week"] = df["ingestion_date"].dt.dayofweek
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(float)
    df["day_sin"]     = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"]     = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # ── Encodings ─────────────────────────────────────────────────────────────
    route_ids  = df["route_id"].unique()
    route_map  = {r: i / len(route_ids) for i, r in enumerate(sorted(route_ids))}
    feed_map   = {"bmtc": 0.0, "namma_metro": 1.0}

    df["route_encoded"] = df["route_id"].map(route_map).fillna(0.5)
    df["feed_encoded"]  = df["feed"].map(feed_map).fillna(0.0)


    feature_cols = [
    "total_observations",
    "unique_vehicles",
    "log_observations",
    "log_vehicles",
    "stationary_pct",
    "slow_pct",
    "network_total_routes",
    "network_stationary",
    "day_sin",
    "day_cos",
    "is_weekend",
    "feed_encoded",
    "route_encoded",
    ]

    # Fill any remaining nulls
    df[feature_cols] = df[feature_cols].fillna(0)

    print(f"  After feature engineering : {len(df):,} rows")
    print(f"  Feature dimensions        : {len(feature_cols)}")
    return df, feature_cols, route_map


def normalize_features(df: pd.DataFrame, feature_cols: list):
    """Min-max normalize. Returns array + scaler params."""
    X = df[feature_cols].values.astype(np.float32)
    y = df["log_observations"].values.astype(np.float32)

    X_min  = X.min(axis=0)
    X_max  = X.max(axis=0)
    X_range = np.where((X_max - X_min) == 0, 1, X_max - X_min)
    X_scaled = (X - X_min) / X_range

    y_min   = float(y.min())
    y_max   = float(y.max())
    y_range = y_max - y_min if y_max != y_min else 1.0
    y_scaled = (y - y_min) / y_range

    scaler_params = {
        "X_min":        X_min.tolist(),
        "X_max":        X_max.tolist(),
        "y_min":        float(y.min()),
        "y_max":        float(y.max()),
        "feature_cols": feature_cols,
        "target_col":   "log_observations",
        "mode":         "cross_sectional",
        "seq_len":      1,
    }
    return X_scaled, y_scaled, scaler_params


def save_outputs(df, X, y, scaler_params, feature_cols):
    FEATURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Save metadata
    meta_df = df[["route_id", "feed", "ingestion_date"]].copy()
    meta_df["target_speed_kmh"] = y
    meta_df.to_parquet(FEATURE_OUTPUT, index=False)

    # Save feature matrix separately for training
    np.save(FEATURE_OUTPUT.parent / "X_features.npy", X)
    np.save(FEATURE_OUTPUT.parent / "y_targets.npy",  y)

    # Save scaler
    scaler_path = FEATURE_OUTPUT.parent / "scaler_params.json"
    with open(scaler_path, "w") as f:
        json.dump(scaler_params, f, indent=2)

    print(f"\n  ✅ Feature store : {FEATURE_OUTPUT}")
    print(f"  ✅ X matrix      : {FEATURE_OUTPUT.parent}/X_features.npy  {X.shape}")
    print(f"  ✅ y targets     : {FEATURE_OUTPUT.parent}/y_targets.npy   {y.shape}")
    print(f"  ✅ Scaler        : {scaler_path}")
    print(f"\n  Target range    : {y.min()*scaler_params['y_max']:.1f} – {scaler_params['y_max']:.1f} km/h")
    print(f"  Samples ready   : {len(X):,}")


def main():
    print("=" * 55)
    print("  Feature Engineering Pipeline v2")
    print("=" * 55 + "\n")

    df = load_gold_table()
    n_days = df["ingestion_date"].nunique()

    if n_days >= MIN_DAYS_FOR_SEQUENCES:
        print(f"\n  {n_days} days available → using sequence mode (SEQ_LEN={SEQ_LEN})")
    else:
        print(f"\n  {n_days} days available → using cross-sectional mode")
        print(f"  (Switch to sequence mode after {MIN_DAYS_FOR_SEQUENCES} days)\n")

    df, feature_cols, route_map = build_cross_sectional_features(df)
    X, y, scaler_params = normalize_features(df, feature_cols)
    save_outputs(df, X, y, scaler_params, feature_cols)

    print(f"\n  Routes encoded  : {len(route_map):,}")
    print(f"  Ready to train  : python3 -m mlops_pipeline.training.train_eta_model")


if __name__ == "__main__":
    main()