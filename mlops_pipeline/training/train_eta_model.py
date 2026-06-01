"""
LSTM/MLP Speed Prediction — Training Pipeline v2
─────────────────────────────────────────────────────────────────────────────
Reads from the feature store built by feature_engineering.py.

Mode detection (automatic):
  - Cross-sectional (< 14 days): MLP on flat feature vectors — fast, accurate
  - Sequence mode  (≥ 14 days): LSTM on time sequences — full temporal model

Both modes log to MLflow and register the best model.

Run:
    python3 -m mlops_pipeline.training.train_eta_model
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

FEATURE_DIR = Path("mlops_pipeline/features")
MODEL_OUTPUT = Path("mlops_pipeline/registry/best_eta_model.pt")
MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT   = "bangalore-transit-eta"

CONFIG = {
    "lr":           1e-3,
    "batch_size":   128,
    "epochs":       150,
    "patience":     15,
    "val_split":    0.2,
    "dropout":      0.3,
    "hidden_dims":  [256, 128, 64],  # MLP layers
    # LSTM config (used when seq mode)
    "lstm_hidden":  128,
    "lstm_layers":  2,
}


# ── Models ────────────────────────────────────────────────────────────────────
class TransitMLP(nn.Module):
    """Fast MLP for cross-sectional features. Works great with 3+ days of data."""
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TransitLSTM(nn.Module):
    """LSTM for sequence mode. Activates after 14+ days of data."""
    def __init__(self, input_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers,
                            dropout=dropout if layers > 1 else 0,
                            batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden // 2, 1)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        return self.head(self.norm(out[:, -1])).squeeze(-1)


# ── Training utilities ────────────────────────────────────────────────────────
def train_epoch(model, loader, opt, criterion, device, amp_scaler):
    model.train()
    total = 0.0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=amp_scaler is not None):
            loss = criterion(model(X_b), y_b)
        if amp_scaler:
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(opt)
            amp_scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        total += loss.item() * len(X_b)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    mse = mae = 0.0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        pred = model(X_b)
        mse += criterion(pred, y_b).item() * len(X_b)
        mae += (pred - y_b).abs().mean().item() * len(X_b)
    n = len(loader.dataset)
    return {"val_rmse": (mse / n) ** 0.5, "val_mae": mae / n}


def load_data():
    X_path = FEATURE_DIR / "X_features.npy"
    y_path = FEATURE_DIR / "y_targets.npy"
    scaler_path = FEATURE_DIR / "scaler_params.json"

    if not X_path.exists():
        raise FileNotFoundError(
            "Feature store not found.\n"
            "Run: python3 -m mlops_pipeline.features.feature_engineering"
        )

    X = np.load(X_path)
    y = np.load(y_path)
    with open(scaler_path) as f:
        scaler = json.load(f)

    print(f"  X shape  : {X.shape}")
    print(f"  y shape  : {y.shape}")
    print(f"  Mode     : {scaler.get('mode', 'cross_sectional')}")
    print(f"  y range  : {y.min():.3f} – {y.max():.3f} (normalized)")
    print(f"  Real km/h: {scaler['y_min']:.1f} – {scaler['y_max']:.1f}")
    return X, y, scaler


def train():
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print("=" * 55)
    print("  Transit ETA Model Training")
    print("=" * 55)
    print(f"\n  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM   : {vram:.1f} GB")

    X, y, scaler = load_data()
    mode = scaler.get("mode", "cross_sectional")

    # Build model based on mode
    input_dim = X.shape[1] if X.ndim == 2 else X.shape[2]
    if mode == "cross_sectional":
        model = TransitMLP(input_dim, CONFIG["hidden_dims"], CONFIG["dropout"])
        model_type = "MLP"
    else:
        model = TransitLSTM(input_dim, CONFIG["lstm_hidden"],
                            CONFIG["lstm_layers"], CONFIG["dropout"])
        model_type = "LSTM"

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model  : {model_type}  ({total_params:,} parameters)")

    # Dataset
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float()
    dataset = TensorDataset(X_t, y_t)
    n_val   = max(10, int(len(dataset) * CONFIG["val_split"]))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=0, pin_memory=True)

    print(f"  Train  : {n_train:,} samples")
    print(f"  Val    : {n_val:,} samples\n")

    optimizer  = torch.optim.AdamW(model.parameters(),
                                   lr=CONFIG["lr"], weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
                     optimizer, T_max=CONFIG["epochs"])
    criterion  = nn.MSELoss()
    amp_scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # MLflow tracking
    try:
        import mlflow
        import mlflow.pytorch
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT)
        mlflow_available = True
    except Exception:
        print("  ⚠️  MLflow not reachable — training without tracking")
        mlflow_available = False

    best_rmse = float("inf")
    patience  = 0

    run_ctx = (mlflow.start_run(run_name=f"transit-{model_type.lower()}")
               if mlflow_available else _null_ctx())

    with run_ctx:
        if mlflow_available:
            mlflow.log_params({**CONFIG, "model_type": model_type,
                               "input_dim": input_dim, "n_train": n_train})

        print(f"{'─'*55}")
        print(f"  Epoch  | train_rmse | val_rmse  | val_mae")
        print(f"{'─'*55}")

        for epoch in range(1, CONFIG["epochs"] + 1):
            t0 = time.time()
            tr_loss = train_epoch(model, train_loader, optimizer,
                                  criterion, device, amp_scaler)
            metrics = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            if mlflow_available:
                mlflow.log_metrics({
                    "train_rmse": tr_loss**0.5, **metrics,
                    "lr": scheduler.get_last_lr()[0],
                }, step=epoch)

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"  {epoch:05d}  | "
                    f"{tr_loss**0.5:.5f}    | "
                    f"{metrics['val_rmse']:.5f}   | "
                    f"{metrics['val_mae']:.5f}  "
                    f"({time.time()-t0:.1f}s)"
                )

            if metrics["val_rmse"] < best_rmse:
                best_rmse = metrics["val_rmse"]
                patience  = 0
                MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state": model.state_dict(),
                    "config":      CONFIG,
                    "scaler":      scaler,
                    "model_type":  model_type,
                    "input_dim":   input_dim,
                    "best_val_rmse": best_rmse,
                    "epoch":       epoch,
                }, MODEL_OUTPUT)
            else:
                patience += 1
                if patience >= CONFIG["patience"]:
                    print(f"\n  Early stop at epoch {epoch}")
                    break

        # Convert RMSE back to km/h for interpretability
        y_range = scaler["y_max"] - scaler["y_min"]
        best_rmse_kmh = best_rmse * y_range

        if mlflow_available:
            ckpt = torch.load(MODEL_OUTPUT, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            mlflow.pytorch.log_model(
                model,
                artifact_path=f"transit_{model_type.lower()}",
                registered_model_name="bangalore-transit-eta",
            )
            mlflow.log_metric("best_val_rmse", best_rmse)
            mlflow.log_metric("best_val_rmse_kmh", best_rmse_kmh)

        print(f"\n{'='*55}")
        print(f"  ✅ Training complete!")
        print(f"  Best val RMSE : {best_rmse:.5f} normalized")
        print(f"                  {best_rmse_kmh:.2f} km/h (real units)")
        print(f"  Model saved   : {MODEL_OUTPUT}")
        if mlflow_available:
            print(f"  MLflow UI     : {MLFLOW_URI}")
        print(f"{'='*55}")


class _null_ctx:
    """No-op context manager when MLflow is unavailable."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


if __name__ == "__main__":
    train()