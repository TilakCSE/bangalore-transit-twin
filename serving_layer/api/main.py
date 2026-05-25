"""
Transit Twin Serving API
─────────────────────────────────────────────────────────────────────────────
REST + WebSocket endpoints consumed by the Unity 3D Digital Twin client.

Endpoints:
  GET  /health                    → liveness probe
  GET  /api/v1/vehicles           → all active vehicle positions (snapshot)
  GET  /api/v1/vehicles/{id}      → single vehicle with ETA prediction
  GET  /api/v1/alerts             → active delay + bunching alerts
  GET  /api/v1/routes             → static route metadata
  WS   /ws/vehicles               → live stream of position updates (Unity)
  WS   /ws/alerts                 → live stream of delay/bunching events
  GET  /api/v1/replay/{date}      → DVR: replay a historical date from Iceberg

Architecture:
  - Vehicle positions read from Redis (written by a Kafka consumer sidecar)
  - ETA predictions served from MLflow Model Registry via KServe
  - WebSocket manager broadcasts updates to all connected Unity clients
  - Replay endpoint streams Iceberg Parquet scans as if they were live
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from typing import AsyncGenerator

import redis.asyncio as aioredis
import structlog
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

log = structlog.get_logger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────
class Settings(BaseSettings):
    redis_url: str = "redis://localhost:6379"
    redis_ttl_vehicles: int = 60
    kafka_bootstrap: str = "localhost:9092"
    mlflow_tracking_uri: str = "http://localhost:5000"
    cors_origins: list[str] = ["http://localhost:3000"]
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()


# ── Models ────────────────────────────────────────────────────────────────────
class VehiclePosition(BaseModel):
    vehicle_id: str
    route_id: str | None = None
    trip_id: str | None = None
    latitude: float
    longitude: float
    bearing: float = 0.0
    speed_mps: float = 0.0
    current_status: str = ""
    feed: str = ""
    timestamp: int
    # ML-enriched fields (added by serving layer)
    eta_seconds: int | None = None
    delay_seconds: int | None = None
    is_bunching: bool = False


class Alert(BaseModel):
    alert_id: str
    alert_type: str  # DELAY | BUNCHING | EARLY | ROUTE_DEVIATION
    route_id: str
    vehicle_id: str | None = None
    vehicle_id_b: str | None = None  # second vehicle for bunching
    delay_sec: int | None = None
    distance_m: float | None = None
    latitude: float | None = None
    longitude: float | None = None
    severity: str = "WARNING"  # INFO | WARNING | CRITICAL
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    message: str = ""


class HealthResponse(BaseModel):
    status: str
    redis: str
    timestamp: datetime


# ── WebSocket connection manager ──────────────────────────────────────────────
class ConnectionManager:
    """
    Manages WebSocket connections for Unity clients.
    Supports two channels: 'vehicles' and 'alerts'.
    Broadcasts are fire-and-forget; disconnected clients are pruned automatically.
    """

    def __init__(self) -> None:
        self._connections: dict[str, set[WebSocket]] = {
            "vehicles": set(),
            "alerts": set(),
        }

    async def connect(self, ws: WebSocket, channel: str) -> None:
        await ws.accept()
        self._connections.setdefault(channel, set()).add(ws)
        log.info("WebSocket connected", channel=channel, total=len(self._connections[channel]))

    def disconnect(self, ws: WebSocket, channel: str) -> None:
        self._connections.get(channel, set()).discard(ws)

    async def broadcast(self, channel: str, message: dict) -> None:
        dead: set[WebSocket] = set()
        payload = json.dumps(message)
        for ws in list(self._connections.get(channel, set())):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self._connections[channel].discard(ws)


manager = ConnectionManager()


# ── App lifespan ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    app.state.redis = await aioredis.from_url(
        settings.redis_url, encoding="utf-8", decode_responses=True
    )
    # Background task: poll Redis for new vehicle positions and broadcast to WS clients
    task = asyncio.create_task(_broadcast_vehicle_positions(app.state.redis))
    log.info("Transit API started", redis=settings.redis_url)
    yield
    task.cancel()
    await app.state.redis.aclose()


async def _broadcast_vehicle_positions(redis_client: aioredis.Redis) -> None:
    """
    Every second, scan Redis for updated vehicle positions and push
    to connected WebSocket clients. This decouples Kafka ingestion from
    WebSocket broadcasting — Unity clients always get the latest state.
    """
    while True:
        try:
            keys = await redis_client.keys("vehicle:*")
            if keys:
                pipe = redis_client.pipeline()
                for k in keys:
                    pipe.get(k)
                values = await pipe.execute()
                positions = []
                for v in values:
                    if v:
                        try:
                            positions.append(json.loads(v))
                        except json.JSONDecodeError:
                            pass
                if positions:
                    await manager.broadcast("vehicles", {"type": "snapshot", "data": positions})
        except Exception as exc:
            log.error("Broadcast error", error=str(exc))
        await asyncio.sleep(1.0)


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Bangalore Transit Digital Twin API",
    description="Real-time serving layer for the 3D smart city digital twin.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST endpoints ────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    # We grab the redis client directly from the app state
    redis = app.state.redis
    try:
        await redis.ping()
        redis_status = "ok"
    except Exception:
        redis_status = "error"
    return HealthResponse(
        status="ok" if redis_status == "ok" else "degraded",
        redis=redis_status,
        timestamp=datetime.now(timezone.utc),
    )


@app.get("/api/v1/vehicles", response_model=list[VehiclePosition], tags=["transit"])
async def get_all_vehicles() -> list[VehiclePosition]:
    """
    Returns a snapshot of all currently active vehicles.
    Used by Unity on initial scene load to place all GameObjects.
    Subsequent updates come via WebSocket /ws/vehicles.
    """
    redis = app.state.redis
    keys = await redis.keys("vehicle:*")
    if not keys:
        return []
    pipe = redis.pipeline()
    for k in keys:
        pipe.get(k)
    values = await pipe.execute()
    positions = []
    for v in values:
        if v:
            try:
                positions.append(VehiclePosition(**json.loads(v)))
            except Exception:
                pass
    return positions


@app.get("/api/v1/vehicles/{vehicle_id}", response_model=VehiclePosition, tags=["transit"])
async def get_vehicle(vehicle_id: str) -> VehiclePosition:
    redis = app.state.redis
    raw = await redis.get(f"vehicle:{vehicle_id}")
    if not raw:
        raise HTTPException(status_code=404, detail=f"Vehicle {vehicle_id} not found")
    return VehiclePosition(**json.loads(raw))


@app.get("/api/v1/alerts", response_model=list[Alert], tags=["transit"])
async def get_alerts() -> list[Alert]:
    """Active delay and bunching alerts from the last 5 minutes."""
    redis = app.state.redis
    keys = await redis.keys("alert:*")
    alerts = []
    if keys:
        pipe = redis.pipeline()
        for k in keys:
            pipe.get(k)
        values = await pipe.execute()
        for v in values:
            if v:
                try:
                    alerts.append(Alert(**json.loads(v)))
                except Exception:
                    pass
    return sorted(alerts, key=lambda a: a.created_at, reverse=True)


@app.get("/api/v1/replay/{replay_date}", tags=["dvr"])
async def get_replay_data(replay_date: date) -> dict:
    """
    DVR replay: returns summarized vehicle trace data for a historical date.
    Full implementation streams from Iceberg Parquet files on GCS.
    Unity time-scrubber calls this endpoint when user drags the replay slider.
    """
    # TODO: implement Iceberg scan via pyiceberg
    # catalog = load_catalog("gcs", **iceberg_config)
    # table = catalog.load_table("silver.vehicle_positions")
    # snapshot = table.scan(row_filter=f"date(event_ts) = '{replay_date}'")
    return {
        "date": replay_date.isoformat(),
        "status": "not_implemented",
        "message": "Iceberg DVR replay — implementation in progress",
    }


# ── WebSocket endpoints ───────────────────────────────────────────────────────
@app.websocket("/ws/vehicles")
async def websocket_vehicles(websocket: WebSocket) -> None:
    """
    Unity connects here on scene load.
    Receives a live stream of VehiclePosition JSON objects at ~1Hz.
    Message format: {"type": "snapshot" | "update", "data": [...]}
    """
    await manager.connect(websocket, "vehicles")
    try:
        while True:
            # Keep-alive: Unity sends a ping every 30s
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, "vehicles")
        log.info("Unity client disconnected from /ws/vehicles")


@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket) -> None:
    """
    Unity HUD subscribes here to receive delay / bunching alert overlays.
    """
    await manager.connect(websocket, "alerts")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, "alerts")