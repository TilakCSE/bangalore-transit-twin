"""
ML Prediction Endpoint
─────────────────────────────────────────────────────────────────────────────
Serves route activity predictions from the trained MLP model.
Called by Unity to color-code routes in the 3D digital twin.

Endpoints:
  GET  /api/v1/predict/route/{route_id}
  POST /api/v1/predict/batch
  GET  /api/v1/predict/network_summary

Add to serving_layer/api/main.py via include_router()
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

MODEL_PATH  = Path("mlops_pipeline/registry/best_eta_model.pt")
SCALER_PATH = Path("mlops_pipeline/features/scaler_params.json")

router = APIRouter(prefix="/api/v1/predict", tags=["predictions"])


# ── Model definition (must match training) ────────────────────────────────────
class TransitMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Model loader (cached — loads once on first request) ───────────────────────
@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run: make retrain"
        )
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    cfg        = checkpoint["config"]
    input_dim  = checkpoint["input_dim"]

    model = TransitMLP(
        input_dim   = input_dim,
        hidden_dims = cfg.get("hidden_dims", [256, 128, 64]),
        dropout     = 0.0,   # disable dropout at inference
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with open(SCALER_PATH) as f:
        scaler = json.load(f)

    return model, scaler, checkpoint


def predict_activity(features: dict[str, float], model, scaler) -> dict:
    """
    Run inference for a single route.
    Returns predicted activity level and Unity color code.
    """
    feature_cols = scaler["feature_cols"]
    x = np.array(
        [features.get(col, 0.0) for col in feature_cols],
        dtype=np.float32
    )

    # Normalize using stored scaler
    X_min   = np.array(scaler["X_min"], dtype=np.float32)
    X_max   = np.array(scaler["X_max"], dtype=np.float32)
    X_range = np.where((X_max - X_min) == 0, 1, X_max - X_min)
    x_scaled = np.clip((x - X_min) / X_range, 0, 1)

    with torch.no_grad():
        tensor = torch.from_numpy(x_scaled).unsqueeze(0)
        pred_norm = model(tensor).item()

    # Denormalize to real units (log_observations)
    y_min   = scaler["y_min"]
    y_max   = scaler["y_max"]
    pred_log = pred_norm * (y_max - y_min) + y_min

    # Convert log_observations back to approximate vehicle count
    pred_vehicles = max(0, np.expm1(pred_log))

    # Activity level 0.0–1.0 for Unity shader
    activity_norm = float(np.clip(pred_norm, 0, 1))

    # Unity color: green (quiet) → yellow (moderate) → red (busy)
    if activity_norm < 0.33:
        color = {"r": 0.2, "g": 0.8, "b": 0.2, "label": "LOW"}
    elif activity_norm < 0.66:
        color = {"r": 1.0, "g": 0.8, "b": 0.0, "label": "MEDIUM"}
    else:
        color = {"r": 0.9, "g": 0.2, "b": 0.1, "label": "HIGH"}

    return {
        "activity_score":     round(activity_norm, 4),
        "predicted_vehicles": round(float(pred_vehicles), 1),
        "color":              color,
    }


# ── Request / Response models ─────────────────────────────────────────────────
class RouteFeatures(BaseModel):
    route_id:           str
    feed:               str = "namma_metro"
    day_of_week:        int = 1        # 0=Monday
    is_weekend:         int = 0
    total_observations: float = 3.0
    unique_vehicles:    float = 3.0
    stationary_pct:     float = 0.0
    slow_pct:           float = 0.0

class BatchPredictRequest(BaseModel):
    routes: list[RouteFeatures]

class PredictionResponse(BaseModel):
    route_id:           str
    feed:               str
    activity_score:     float
    predicted_vehicles: float
    color:              dict
    model_version:      str = "v1"


# ── Endpoints ─────────────────────────────────────────────────────────────────
@router.get("/health")
async def predict_health():
    """Check if model is loaded and ready."""
    try:
        model, scaler, ckpt = load_model()
        return {
            "status":        "ready",
            "model_type":    ckpt.get("model_type", "MLP"),
            "best_val_rmse": round(ckpt.get("best_val_rmse", 0), 5),
            "input_dim":     ckpt.get("input_dim"),
            "feature_cols":  scaler["feature_cols"],
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/route/{route_id}", response_model=PredictionResponse)
async def predict_route(
    route_id:    str,
    feed:        str = "namma_metro",
    day_of_week: int = 1,
    is_weekend:  int = 0,
):
    """
    Predict activity level for a single route.
    Unity calls this to color a specific route GameObject.
    """
    try:
        model, scaler, ckpt = load_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    features = {
        "feed_encoded":        1.0 if feed == "namma_metro" else 0.0,
        "day_sin":             float(np.sin(2 * np.pi * day_of_week / 7)),
        "day_cos":             float(np.cos(2 * np.pi * day_of_week / 7)),
        "is_weekend":          float(is_weekend),
        "total_observations":  3.0,
        "unique_vehicles":     3.0,
        "log_observations":    float(np.log1p(3.0)),
        "log_vehicles":        float(np.log1p(3.0)),
        "stationary_pct":      0.0,
        "slow_pct":            0.0,
        "network_total_routes": 500.0,
        "network_stationary":  0.0,
        "route_encoded":       0.5,
    }

    result = predict_activity(features, model, scaler)
    return PredictionResponse(
        route_id=route_id,
        feed=feed,
        **result,
    )


@router.post("/batch", response_model=list[PredictionResponse])
async def predict_batch(request: BatchPredictRequest):
    """
    Predict activity for multiple routes at once.
    Unity calls this on scene load to color all route GameObjects.
    """
    try:
        model, scaler, ckpt = load_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=503)

    results = []
    for route in request.routes:
        features = {
            "feed_encoded":         1.0 if route.feed == "namma_metro" else 0.0,
            "day_sin":              float(np.sin(2 * np.pi * route.day_of_week / 7)),
            "day_cos":              float(np.cos(2 * np.pi * route.day_of_week / 7)),
            "is_weekend":           float(route.is_weekend),
            "total_observations":   route.total_observations,
            "unique_vehicles":      route.unique_vehicles,
            "log_observations":     float(np.log1p(route.total_observations)),
            "log_vehicles":         float(np.log1p(route.unique_vehicles)),
            "stationary_pct":       route.stationary_pct,
            "slow_pct":             route.slow_pct,
            "network_total_routes": 500.0,
            "network_stationary":   0.0,
            "route_encoded":        0.5,
        }
        result = predict_activity(features, model, scaler)
        results.append(PredictionResponse(
            route_id=route.route_id,
            feed=route.feed,
            **result,
        ))
    return results


@router.get("/network_summary")
async def network_summary():
    """
    Returns network-wide activity summary.
    Unity HUD calls this to display the city health indicator.
    """
    import datetime
    now          = datetime.datetime.now()
    day_of_week  = now.weekday()
    is_weekend   = int(day_of_week >= 5)

    try:
        model, scaler, _ = load_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Sample 10 representative routes
    sample_routes = [f"route_{i:03d}" for i in range(10)]
    scores = []
    for _ in sample_routes:
        features = {
            "feed_encoded":         1.0,
            "day_sin":              float(np.sin(2 * np.pi * day_of_week / 7)),
            "day_cos":              float(np.cos(2 * np.pi * day_of_week / 7)),
            "is_weekend":           float(is_weekend),
            "total_observations":   3.0,
            "unique_vehicles":      3.0,
            "log_observations":     float(np.log1p(3.0)),
            "log_vehicles":         float(np.log1p(3.0)),
            "stationary_pct":       0.0,
            "slow_pct":             0.0,
            "network_total_routes": 500.0,
            "network_stationary":   0.0,
            "route_encoded":        0.5,
        }
        result = predict_activity(features, model, scaler)
        scores.append(result["activity_score"])

    avg_activity = float(np.mean(scores))
    return {
        "timestamp":            now.isoformat(),
        "day_of_week":          day_of_week,
        "is_weekend":           bool(is_weekend),
        "network_activity":     round(avg_activity, 4),
        "network_status":       "BUSY" if avg_activity > 0.66
                                else "MODERATE" if avg_activity > 0.33
                                else "QUIET",
        "color":                "red" if avg_activity > 0.66
                                else "yellow" if avg_activity > 0.33
                                else "green",
    }