"""
Unit tests — serving layer and GTFS parser
─────────────────────────────────────────────────────────────────────────────
Run: pytest tests/unit/ -v
"""

from __future__ import annotations

import io
import json
import zipfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket

from data_engineering.ingestion.gtfs_static.parser import (
    GTFSFeed,
    parse_gtfs_zip,
    validate_stops_bbox,
)
from serving_layer.api.main import VehiclePosition, Alert, app


# ── Fixtures ──────────────────────────────────────────────────────────────────
def make_gtfs_zip(stops_csv: str, routes_csv: str = "route_id,route_short_name\nR1,Route 1\n") -> bytes:
    """Helper: build a minimal GTFS ZIP in memory."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as zf:
        zf.writestr("stops.txt", stops_csv)
        zf.writestr("routes.txt", routes_csv)
        zf.writestr("trips.txt", "route_id,service_id,trip_id\nR1,S1,T1\n")
        zf.writestr("stop_times.txt",
                    "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
                    "T1,08:00:00,08:00:30,S1,1\n")
        zf.writestr("agency.txt", "agency_id,agency_name,agency_url,agency_timezone\n"
                                  "BMTC,BMTC,https://bmtc.in,Asia/Kolkata\n")
        zf.writestr("calendar.txt",
                    "service_id,monday,tuesday,wednesday,thursday,friday,"
                    "saturday,sunday,start_date,end_date\n"
                    "S1,1,1,1,1,1,0,0,20240101,20241231\n")
    return buf.getvalue()


VALID_STOPS_CSV = (
    "stop_id,stop_name,stop_lat,stop_lon\n"
    "S1,Majestic,12.9767,77.5713\n"
    "S2,Koramangala,12.9279,77.6271\n"
)

INVALID_STOPS_CSV = (
    "stop_id,stop_name,stop_lat,stop_lon\n"
    "S1,Majestic,12.9767,77.5713\n"
    "S99,Out of City,11.0,76.0\n"   # outside Bangalore bbox
)


# ── GTFS Parser tests ─────────────────────────────────────────────────────────
class TestGTFSParser:
    def test_parse_valid_zip(self):
        zip_bytes = make_gtfs_zip(VALID_STOPS_CSV)
        feed = parse_gtfs_zip("bmtc", zip_bytes)
        assert isinstance(feed, GTFSFeed)
        assert feed.feed_name == "bmtc"
        assert "stops" in feed.tables
        assert len(feed.tables["stops"]) == 2

    def test_feed_name_metadata_column(self):
        zip_bytes = make_gtfs_zip(VALID_STOPS_CSV)
        feed = parse_gtfs_zip("bmtc", zip_bytes)
        assert (feed.tables["stops"]["_feed_name"] == "bmtc").all()

    def test_validate_stops_valid_bbox(self):
        zip_bytes = make_gtfs_zip(VALID_STOPS_CSV)
        feed = parse_gtfs_zip("bmtc", zip_bytes)
        violations = validate_stops_bbox(feed)
        assert violations == []

    def test_validate_stops_invalid_bbox(self):
        zip_bytes = make_gtfs_zip(INVALID_STOPS_CSV)
        feed = parse_gtfs_zip("bmtc", zip_bytes)
        violations = validate_stops_bbox(feed)
        assert len(violations) > 0
        assert "S99" in violations[0]

    def test_missing_table_is_warned_not_raised(self):
        # ZIP missing shapes.txt — should not raise, just skip
        zip_bytes = make_gtfs_zip(VALID_STOPS_CSV)
        feed = parse_gtfs_zip("bmtc", zip_bytes, tables_to_load=["stops", "shapes"])
        assert "stops" in feed.tables
        assert "shapes" not in feed.tables  # missing table skipped gracefully

    def test_repr_contains_feed_name(self):
        zip_bytes = make_gtfs_zip(VALID_STOPS_CSV)
        feed = parse_gtfs_zip("namma_metro", zip_bytes)
        assert "namma_metro" in repr(feed)


# ── Pydantic model tests ───────────────────────────────────────────────────────
class TestVehiclePositionModel:
    def test_minimal_valid_vehicle(self):
        v = VehiclePosition(
            vehicle_id="BMTC-1001",
            latitude=12.9716,
            longitude=77.5946,
            timestamp=1700000000,
        )
        assert v.vehicle_id == "BMTC-1001"
        assert v.is_bunching is False
        assert v.eta_seconds is None

    def test_vehicle_with_all_fields(self):
        v = VehiclePosition(
            vehicle_id="METRO-42",
            route_id="Purple Line",
            trip_id="T-9999",
            latitude=12.9279,
            longitude=77.6271,
            bearing=180.0,
            speed_mps=12.5,
            current_status="IN_TRANSIT_TO",
            feed="namma_metro",
            timestamp=1700000100,
            eta_seconds=180,
            delay_seconds=-30,
            is_bunching=False,
        )
        assert v.speed_mps == 12.5
        assert v.eta_seconds == 180


class TestAlertModel:
    def test_alert_defaults(self):
        a = Alert(
            alert_id="abc-123",
            alert_type="DELAY",
            route_id="Route 500C",
            vehicle_id="BMTC-555",
            delay_sec=360,
        )
        assert a.severity == "WARNING"
        assert a.message == ""
        assert a.created_at is not None


# ── API endpoint tests (synchronous test client) ──────────────────────────────
class TestAPIEndpoints:
    @pytest.fixture(autouse=True)
    def patch_redis(self):
        """Replace Redis with a mock so tests don't need a running Redis."""
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        mock_redis.keys.return_value = []
        mock_redis.pipeline.return_value.__aenter__ = AsyncMock(return_value=mock_redis)
        mock_redis.pipeline.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_redis.execute.return_value = []

        with patch("serving_layer.api.main.aioredis.from_url", return_value=mock_redis):
            # Inject mock redis into app state for tests
            app.state.redis = mock_redis
            yield mock_redis

    def test_health_ok(self, patch_redis):
        client = TestClient(app, raise_server_exceptions=False)
        # Health check: redis returns ping=True
        patch_redis.ping = AsyncMock(return_value=True)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ok", "degraded")

    def test_get_vehicles_empty(self, patch_redis):
        patch_redis.keys = AsyncMock(return_value=[])
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/vehicles")
        assert response.status_code == 200
        assert response.json() == []

    def test_get_vehicle_not_found(self, patch_redis):
        patch_redis.get = AsyncMock(return_value=None)
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/vehicles/NONEXISTENT")
        assert response.status_code == 404

    def test_get_alerts_empty(self, patch_redis):
        patch_redis.keys = AsyncMock(return_value=[])
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/alerts")
        assert response.status_code == 200
        assert response.json() == []

    def test_replay_endpoint_returns_date(self, patch_redis):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/replay/2024-01-15")
        assert response.status_code == 200
        assert response.json()["date"] == "2024-01-15"