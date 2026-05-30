"""
Feature Engineering Pipeline
─────────────────────────────────────────────────────────────────────────────
Reads the Gold DuckDB table and builds supervised learning sequences
for the LSTM ETA / speed prediction model.

Input:  data_engineering/dbt/transit_twin.duckdb → gold.gold_route_performance_daily
Output: mlops_pipeline/features/feature_store.parquet

Feature set per route per day:
  - avg_speed_kmh (target for next-day prediction)
  - stationary_pct, slow_pct (congestion signals)
  - avg_speed_am_peak_kmh, avg_speed_pm_peak_kmh
  - reliability_score
  - day_of_week (0-6), is_weekend
  - lag_1_speed, lag_2_speed, lag_3_speed (autoregressive features)
  - rolling_7d_speed (trend)

Run:
    python3 -m mlops_pipeline.features.feature_engineering
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DBT_DB_PATH    = Path("data_engineering/dbt/transit_twin.duckdb")
FEATURE_OUTPUT = Path("mlops_pipeline/features/feature_store.parquet")
SEQ_LEN        = 7   # use 7 days of history to predict the next day
TARGET_COL     = "avg_speed_kmh"


def load_gold_table() -> pd.DataFrame:
    import duckdb
    if not DBT_DB_PATH.exists():
        raise FileNotFoundError(
            f"DuckDB file not found: {DBT_DB_PATH}\n"
            "Run dbt first: cd data_engineering/dbt && dbt run --profiles-dir ."
        )
    conn = duckdb.connect(str(DBT_DB_PATH), read_only=True)
    df = conn.execute("""
        SELECT
            route_id,
            feed,
            ingestion_date,
            avg_speed_kmh,
            avg_speed_am_peak_kmh,
            avg_speed_pm_peak_kmh,
            stationary_pct,
            slow_pct,
            reliability_score,
            unique_vehicles,
            total_observations
        FROM main_gold.gold_route_performance_daily
        WHERE avg_speed_kmh IS NOT NULL
          AND route_id != 'UNKNOWN'
        ORDER BY route_id, feed, ingestion_date
    """).df()
    conn.close()
    print(f"Loaded {len(df):,} rows from Gold table")
    print(f"Feeds: {df['feed'].unique()}")
    print(f"Date range: {df['ingestion_date'].min()} → {df['ingestion_date'].max()}")
    print(f"Unique routes: {df['route_id'].nunique()}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ingestion_date"] = pd.to_datetime(df["ingestion_date"])
    df = df.sort_values(["route_id", "feed", "ingestion_date"])

    # Fill nulls in optional columns
    for col in ["avg_speed_am_peak_kmh", "avg_speed_pm_peak_kmh",
                "stationary_pct", "slow_pct", "reliability_score"]:
        df[col] = df[col].fillna(df[col].median())

    # Time features
    df["day_of_week"] = df["ingestion_date"].dt.dayofweek   # 0=Monday
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["day_sin"]     = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"]     = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Lag features per route (autoregressive signal)
    grp = df.groupby(["route_id", "feed"])
    df["lag_1_speed"] = grp["avg_speed_kmh"].shift(1)
    df["lag_2_speed"] = grp["avg_speed_kmh"].shift(2)
    df["lag_3_speed"] = grp["avg_speed_kmh"].shift(3)
    df["rolling_7d_speed"] = (
        grp["avg_speed_kmh"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )
    df["speed_momentum"] = df["avg_speed_kmh"] - df["lag_1_speed"]

    # Drop rows where we don't have enough history
    df = df.dropna(subset=["lag_1_speed", "lag_2_speed"])

    # Route encoding (ordinal for LSTM embedding)
    route_map = {r: i for i, r in enumerate(df["route_id"].unique())}
    df["route_encoded"] = df["route_id"].map(route_map)

    feed_map = {"bmtc": 0, "namma_metro": 1}
    df["feed_encoded"] = df["feed"].map(feed_map).fillna(0)

    print(f"\nAfter feature engineering: {len(df):,} rows")
    print(f"Feature columns: {[c for c in df.columns if c not in ['route_id','feed','ingestion_date']]}")
    return df, route_map


def build_sequences(df: pd.DataFrame, seq_len: int = SEQ_LEN):
    """
    Build (X, y) sequences for LSTM training.
    X shape: (n_samples, seq_len, n_features)
    y shape: (n_samples,) — next day's avg_speed_kmh
    """
    feature_cols = [
        "avg_speed_kmh", "avg_speed_am_peak_kmh", "avg_speed_pm_peak_kmh",
        "stationary_pct", "slow_pct", "reliability_score",
        "lag_1_speed", "lag_2_speed", "lag_3_speed",
        "rolling_7d_speed", "speed_momentum",
        "day_sin", "day_cos", "is_weekend",
        "feed_encoded", "route_encoded",
        "unique_vehicles",
    ]

    X_list, y_list, meta_list = [], [], []

    for (route_id, feed), group in df.groupby(["route_id", "feed"]):
        group = group.sort_values("ingestion_date").reset_index(drop=True)
        if len(group) < seq_len + 1:
            continue  # not enough history for this route
        vals = group[feature_cols].values.astype(np.float32)
        targets = group[TARGET_COL].values.astype(np.float32)

        for i in range(seq_len, len(group)):
            X_list.append(vals[i - seq_len : i])
            y_list.append(targets[i])
            meta_list.append({
                "route_id": route_id,
                "feed": feed,
                "date": group["ingestion_date"].iloc[i],
            })

    if not X_list:
        raise ValueError(
            f"No sequences built — need at least {seq_len + 1} days of data per route.\n"
            f"Current data has {df['ingestion_date'].nunique()} unique dates.\n"
            "Keep the pipeline running for more days, or reduce SEQ_LEN."
        )

    X = np.stack(X_list)  # (n, seq_len, n_features)
    y = np.array(y_list)  # (n,)
    print(f"\nSequences built: X={X.shape}, y={y.shape}")
    return X, y, meta_list, feature_cols


def normalize(X: np.ndarray, y: np.ndarray):
    """Min-max normalize. Returns scaled arrays + scaler params for inference."""
    X_min = X.min(axis=(0, 1), keepdims=True)
    X_max = X.max(axis=(0, 1), keepdims=True)
    X_range = np.where((X_max - X_min) == 0, 1, X_max - X_min)
    X_scaled = (X - X_min) / X_range

    y_min, y_max = y.min(), y.max()
    y_range = y_max - y_min if y_max != y_min else 1
    y_scaled = (y - y_min) / y_range

    scaler_params = {
        "X_min": X_min, "X_max": X_max,
        "y_min": float(y_min), "y_max": float(y_max),
    }
    return X_scaled, y_scaled, scaler_params


def save_feature_store(X, y, meta, feature_cols, scaler_params) -> None:
    FEATURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    meta_df = pd.DataFrame(meta)
    meta_df["target_speed_kmh"] = y
    meta_df.to_parquet(FEATURE_OUTPUT, index=False)

    # Save scaler params for inference
    import json
    scaler_path = FEATURE_OUTPUT.parent / "scaler_params.json"
    serializable = {
        "X_min": scaler_params["X_min"].tolist(),
        "X_max": scaler_params["X_max"].tolist(),
        "y_min": scaler_params["y_min"],
        "y_max": scaler_params["y_max"],
        "feature_cols": feature_cols,
        "seq_len": SEQ_LEN,
    }
    with open(scaler_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\n✅ Feature store saved: {FEATURE_OUTPUT}")
    print(f"✅ Scaler params saved: {scaler_path}")
    print(f"   Samples: {len(meta_df)}")
    print(f"   Target range: {y.min():.1f} – {y.max():.1f} km/h")


def main():
    print("=" * 55)
    print("  Feature Engineering Pipeline")
    print("=" * 55)

    df = load_gold_table()
    df, route_map = engineer_features(df)

    # Handle case where we have limited data (early in pipeline)
    available_days = df["ingestion_date"].nunique()
    seq_len = min(SEQ_LEN, max(1, available_days - 1))
    if seq_len < SEQ_LEN:
        print(f"\n⚠️  Only {available_days} days of data — using seq_len={seq_len}")
        print("   More data accumulates as the pipeline keeps running.")

    X, y, meta, feature_cols = build_sequences(df, seq_len=seq_len)
    X_scaled, y_scaled, scaler_params = normalize(X, y)
    save_feature_store(X_scaled, y_scaled, meta, feature_cols, scaler_params)
    print(f"\n  Routes encoded: {len(route_map)}")
    print(f"  Feature dims:  {X.shape[2]}")


if __name__ == "__main__":
    main()