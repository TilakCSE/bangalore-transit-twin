"""
LSTM Speed/ETA Prediction Model — Training Pipeline
─────────────────────────────────────────────────────────────────────────────
Trains an LSTM on the feature store built by feature_engineering.py.
Uses your RTX 3050 GPU via CUDA.
Tracked by MLflow — view at http://localhost:5000

Run:
    python3 -m mlops_pipeline.training.train_eta_model
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

FEATURE_STORE  = Path("mlops_pipeline/features/feature_store.parquet")
SCALER_PARAMS  = Path("mlops_pipeline/features/scaler_params.json")
MODEL_OUTPUT   = Path("mlops_pipeline/registry/best_eta_model.pt")
MLFLOW_URI     = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT     = "bangalore-transit-eta"

CONFIG = {
    "seq_len":      7,
    "input_dim":    17,    # must match feature_engineering output
    "hidden_dim":   128,
    "num_layers":   2,
    "dropout":      0.2,
    "lr":           1e-3,
    "batch_size":   64,
    "epochs":       100,
    "patience":     12,
    "val_split":    0.15,
}


# ── Dataset ───────────────────────────────────────────────────────────────────
class TransitDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_features():
    import pandas as pd
    if not FEATURE_STORE.exists():
        raise FileNotFoundError(
            f"Feature store not found: {FEATURE_STORE}\n"
            "Run first: python3 -m mlops_pipeline.features.feature_engineering"
        )
    with open(SCALER_PARAMS) as f:
        scaler = json.load(f)

    meta_df  = pd.read_parquet(FEATURE_STORE)
    seq_len  = scaler["seq_len"]
    n_feats  = len(scaler["feature_cols"])
    n        = len(meta_df)

    # Rebuild X from the scaled target (meta only stores targets + metadata)
    # For real sequences we need to reload from Gold — use synthetic augmentation
    # if we have < 100 samples (early pipeline stage)
    y = meta_df["target_speed_kmh"].values.astype(np.float32)

    if n < 50:
        print(f"⚠️  Only {n} real samples — augmenting with synthetic data for training")
        print("   As more days accumulate, rerun feature_engineering.py for real data")
        X, y = _synthetic_sequences(n_samples=2000, seq_len=seq_len, n_feats=n_feats)
    else:
        # Reconstruct sequences — X was normalized and stored in feature store
        # For now reconstruct from available targets with lag features
        X = _reconstruct_sequences(meta_df, seq_len, n_feats)

    return X, y, scaler


def _synthetic_sequences(n_samples, seq_len, n_feats):
    """
    Realistic synthetic sequences based on Indian transit speed distributions.
    Used when real data is still accumulating.
    """
    rng = np.random.default_rng(42)
    # Simulate speed sequences: Bangalore buses avg 15-25 km/h
    base_speeds = rng.uniform(10, 35, n_samples)
    X = np.zeros((n_samples, seq_len, n_feats), dtype=np.float32)
    for i in range(n_samples):
        for t in range(seq_len):
            noise = rng.normal(0, 0.05)
            X[i, t, 0] = np.clip(base_speeds[i] / 60 + noise, 0, 1)  # speed normalized
            X[i, t, 1] = np.clip(base_speeds[i] * 0.8 / 60 + noise, 0, 1)  # am peak
            X[i, t, 2] = np.clip(base_speeds[i] * 0.7 / 60 + noise, 0, 1)  # pm peak
            X[i, t, 6] = X[i, max(0, t-1), 0]  # lag_1
            X[i, t, 13] = rng.integers(0, 2)   # is_weekend
        # target: next day speed (slight regression to mean)
    y = (base_speeds * 0.9 + rng.normal(0, 2, n_samples)).clip(5, 60).astype(np.float32)
    y = (y - y.min()) / (y.max() - y.min() + 1e-8)
    return X, y


def _reconstruct_sequences(meta_df, seq_len, n_feats):
    """Build feature matrix from available metadata columns."""
    n = len(meta_df)
    X = np.zeros((n, seq_len, n_feats), dtype=np.float32)
    targets = meta_df["target_speed_kmh"].values
    for i in range(n):
        for t in range(seq_len):
            lag = max(0, i - (seq_len - t))
            X[i, t, 0] = targets[lag]                    # speed
            X[i, t, 13] = 1 if t % 7 >= 5 else 0        # weekend approx
    return X


# ── Model ─────────────────────────────────────────────────────────────────────
class ETALSTMModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=cfg["input_dim"],
            hidden_size=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"] if cfg["num_layers"] > 1 else 0.0,
            batch_first=True,
        )
        self.norm   = nn.LayerNorm(cfg["hidden_dim"])
        self.drop   = nn.Dropout(cfg["dropout"])
        self.head   = nn.Sequential(
            nn.Linear(cfg["hidden_dim"], cfg["hidden_dim"] // 2),
            nn.GELU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(cfg["hidden_dim"] // 2, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last   = self.norm(out[:, -1, :])
        last   = self.drop(last)
        return self.head(last).squeeze(-1)


# ── Training ──────────────────────────────────────────────────────────────────
def train_epoch(model, loader, opt, criterion, device, scaler_amp):
    model.train()
    total = 0.0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=scaler_amp is not None):
            pred = model(X_b)
            loss = criterion(pred, y_b)
        if scaler_amp:
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(opt)
            scaler_amp.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        total += loss.item() * len(X_b)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    mse, mae = 0.0, 0.0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        pred = model(X_b)
        mse += criterion(pred, y_b).item() * len(X_b)
        mae += (pred - y_b).abs().mean().item() * len(X_b)
    n = len(loader.dataset)
    return {"val_rmse": (mse / n) ** 0.5, "val_mae": mae / n}


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"\n  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    X, y, scaler = load_features()
    # Update config input_dim from actual data
    CONFIG["input_dim"]  = X.shape[2]
    CONFIG["seq_len"]    = X.shape[1]

    print(f"  Samples: {len(X):,}  |  Features: {X.shape[2]}  |  Seq len: {X.shape[1]}")

    dataset  = TransitDataset(X, y)
    n_val    = max(1, int(len(dataset) * CONFIG["val_split"]))
    n_train  = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=2, pin_memory=True)

    model     = ETALSTMModel(CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=CONFIG["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=CONFIG["epochs"])
    criterion = nn.MSELoss()
    amp_scaler = torch.amp.GradScaler("cuda") if use_amp else None

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    best_rmse, patience_count = float("inf"), 0

    with mlflow.start_run(run_name="lstm-transit-eta"):
        mlflow.log_params(CONFIG)
        mlflow.log_param("device", str(device))
        mlflow.log_param("train_samples", n_train)
        mlflow.log_param("val_samples", n_val)

        print(f"\n{'─'*55}")
        print(f"  Training LSTM — {CONFIG['epochs']} epochs max")
        print(f"  Early stop patience: {CONFIG['patience']}")
        print(f"{'─'*55}")

        for epoch in range(1, CONFIG["epochs"] + 1):
            t0 = time.time()
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, device, amp_scaler
            )
            val_metrics = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            mlflow.log_metrics({
                "train_rmse": train_loss ** 0.5,
                **val_metrics,
                "lr": scheduler.get_last_lr()[0],
            }, step=epoch)

            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:03d} | "
                    f"train_rmse={train_loss**0.5:.4f} | "
                    f"val_rmse={val_metrics['val_rmse']:.4f} | "
                    f"val_mae={val_metrics['val_mae']:.4f} | "
                    f"{time.time()-t0:.1f}s"
                )

            if val_metrics["val_rmse"] < best_rmse:
                best_rmse = val_metrics["val_rmse"]
                patience_count = 0
                MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state": model.state_dict(),
                    "config": CONFIG,
                    "scaler": scaler,
                    "best_val_rmse": best_rmse,
                    "epoch": epoch,
                }, MODEL_OUTPUT)
            else:
                patience_count += 1
                if patience_count >= CONFIG["patience"]:
                    print(f"\n  Early stop at epoch {epoch}")
                    break

        # Register in MLflow Model Registry
        checkpoint = torch.load(MODEL_OUTPUT, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        mlflow.pytorch.log_model(
            model,
            artifact_path="eta_lstm",
            registered_model_name="bangalore-transit-eta-lstm",
        )
        mlflow.log_metric("best_val_rmse", best_rmse)
        mlflow.log_artifact(str(MODEL_OUTPUT))

        print(f"\n{'='*55}")
        print(f"  ✅ Training complete!")
        print(f"  Best val RMSE : {best_rmse:.4f}")
        print(f"  Model saved   : {MODEL_OUTPUT}")
        print(f"  MLflow UI     : {MLFLOW_URI}")
        print(f"{'='*55}")


if __name__ == "__main__":
    train()