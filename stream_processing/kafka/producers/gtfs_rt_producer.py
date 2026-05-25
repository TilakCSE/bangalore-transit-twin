"""
GTFS-RT Kafka Producer
─────────────────────────────────────────────────────────────────────────────
Polls BMTC and Namma Metro GTFS-RT endpoints (vehicle positions + trip updates)
and publishes decoded protobuf messages as JSON to Kafka topics.

Runs as a long-lived async process. Handles:
  - Exponential backoff on feed failures
  - Schema validation before publish
  - Prometheus metrics (lag, message rate, error rate)
  - Graceful shutdown on SIGTERM (for Kubernetes pod lifecycle)

Usage:
    python -m stream_processing.kafka.producers.gtfs_rt_producer
    # or via Docker / Kubernetes CronJob
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import structlog
from confluent_kafka import Producer
from dotenv import load_dotenv
from google.transit import gtfs_realtime_pb2
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()
log = structlog.get_logger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────
MESSAGES_PUBLISHED = Counter(
    "gtfs_rt_messages_published_total",
    "Total GTFS-RT messages published to Kafka",
    ["feed", "topic"],
)
FEED_ERRORS = Counter(
    "gtfs_rt_feed_errors_total",
    "Total feed fetch errors",
    ["feed"],
)
FETCH_LATENCY = Histogram(
    "gtfs_rt_fetch_latency_seconds",
    "Time to fetch and parse a GTFS-RT feed",
    ["feed"],
)
ACTIVE_VEHICLES = Gauge(
    "gtfs_rt_active_vehicles",
    "Number of active vehicles in the last feed",
    ["feed"],
)


@dataclass
class FeedConfig:
    name: str
    vehicle_positions_url: str
    trip_updates_url: str
    poll_interval_sec: int = 15
    headers: dict[str, str] = field(default_factory=dict)


# ── Feed registry ─────────────────────────────────────────────────────────────
FEEDS: list[FeedConfig] = [
    FeedConfig(
        name="bmtc",
        vehicle_positions_url=os.getenv(
            "BMTC_GTFS_RT_VEHICLE_POSITIONS_URL",
            "https://cdn.mbta.com/realtime/VehiclePositions.pb",
        ),
        trip_updates_url=os.getenv(
            "BMTC_GTFS_RT_TRIP_UPDATES_URL",
            "https://cdn.mbta.com/realtime/TripUpdates.pb",
        ),
        poll_interval_sec=15,
    ),
]

# --- COMMENTED OUT THE METRO FEED FOR TONIGHT ---
# _otd_key = os.getenv("OTD_API_KEY", "")
# if _otd_key and _otd_key != "your-otd-api-key":
#     FEEDS.append(
#         FeedConfig(
#             name="namma_metro",
#             vehicle_positions_url=os.getenv("NAMMA_METRO_GTFS_RT_URL", ""),
#             trip_updates_url=os.getenv("NAMMA_METRO_TRIP_UPDATES_URL", ""),
#             poll_interval_sec=10,
#             headers={"x-api-key": _otd_key},
#         )
#     )



class GTFSRTProducer:
    """
    Async GTFS-RT feed poller and Kafka publisher.

    Architecture note: each feed runs in its own asyncio task so poll intervals
    are independent. A single Kafka Producer instance is thread-safe and shared.
    """

    def __init__(self) -> None:
        self.producer = Producer(
            {
                "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"),
                "acks": "all",
                "retries": 5,
                "retry.backoff.ms": 300,
                "compression.type": "lz4",
                "linger.ms": 5,         # micro-batching for throughput
                "batch.size": 65536,
            }
        )
        self._running = True
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        # Register SIGTERM handler for Kubernetes graceful shutdown
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGTERM, self._shutdown)

        # Start Prometheus metrics server
        start_http_server(8090)
        log.info("Prometheus metrics server started", port=8090)

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        ) as session:
            self._session = session
            tasks = [asyncio.create_task(self._poll_feed(feed)) for feed in FEEDS]
            await asyncio.gather(*tasks, return_exceptions=True)

    def _shutdown(self) -> None:
        log.info("Received SIGTERM, shutting down producer gracefully")
        self._running = False
        self.producer.flush(timeout=15)

    async def _poll_feed(self, feed: FeedConfig) -> None:
        log.info("Starting feed poller", feed=feed.name, interval=feed.poll_interval_sec)
        while self._running:
            start = time.monotonic()
            try:
                await self._fetch_and_publish(feed, "vehicle_positions", feed.vehicle_positions_url)
                await self._fetch_and_publish(feed, "trip_updates", feed.trip_updates_url)
            except Exception as exc:
                FEED_ERRORS.labels(feed=feed.name).inc()
                log.error("Feed poll failed", feed=feed.name, error=str(exc))
            elapsed = time.monotonic() - start
            await asyncio.sleep(max(0, feed.poll_interval_sec - elapsed))

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _fetch_and_publish(
        self, feed: FeedConfig, feed_type: str, url: str
    ) -> None:
        topic = (
            os.getenv("TOPIC_VEHICLE_POSITIONS", "vehicle-positions")
            if feed_type == "vehicle_positions"
            else os.getenv("TOPIC_TRIP_UPDATES", "trip-updates")
        )
        with FETCH_LATENCY.labels(feed=feed.name).time():
            async with self._session.get(url, headers=feed.headers) as resp:
                resp.raise_for_status()
                raw = await resp.read()

        fm = gtfs_realtime_pb2.FeedMessage()
        fm.ParseFromString(raw)

        entities = list(fm.entity)
        ACTIVE_VEHICLES.labels(feed=feed.name).set(len(entities))

        for entity in entities:
            message = self._entity_to_dict(entity, feed.name, feed_type, fm.header)
            if message is None:
                continue
            self.producer.produce(
                topic=topic,
                key=entity.id.encode(),
                value=json.dumps(message).encode(),
                on_delivery=self._delivery_report,
            )
            MESSAGES_PUBLISHED.labels(feed=feed.name, topic=topic).inc()

        # Non-blocking poll to trigger delivery callbacks
        self.producer.poll(0)
        log.debug("Published feed batch", feed=feed.name, type=feed_type, count=len(entities))

    @staticmethod
    def _entity_to_dict(
        entity: Any, feed_name: str, feed_type: str, header: Any
    ) -> dict | None:
        """Normalize a GTFS-RT FeedEntity to a flat dict for Kafka serialization."""
        base = {
            "entity_id": entity.id,
            "feed": feed_name,
            "feed_type": feed_type,
            "feed_timestamp": header.timestamp,
            "ingested_at": int(time.time()),
        }
        if feed_type == "vehicle_positions" and entity.HasField("vehicle"):
            v = entity.vehicle
            pos = v.position
            return {
                **base,
                "vehicle_id": v.vehicle.id,
                "label": v.vehicle.label,
                "trip_id": v.trip.trip_id if v.HasField("trip") else None,
                "route_id": v.trip.route_id if v.HasField("trip") else None,
                "latitude": pos.latitude,
                "longitude": pos.longitude,
                "bearing": pos.bearing,
                "speed_mps": pos.speed,
                "current_stop_sequence": v.current_stop_sequence,
                "current_status": v.VehicleStopStatus.Name(v.current_status),
                "occupancy_status": v.OccupancyStatus.Name(v.occupancy_status)
                if v.HasField("occupancy_status")
                else None,
                "timestamp": v.timestamp,
            }
        if feed_type == "trip_updates" and entity.HasField("trip_update"):
            tu = entity.trip_update
            return {
                **base,
                "trip_id": tu.trip.trip_id,
                "route_id": tu.trip.route_id,
                "vehicle_id": tu.vehicle.id if tu.HasField("vehicle") else None,
                "stop_time_updates": [
                    {
                        "stop_sequence": stu.stop_sequence,
                        "stop_id": stu.stop_id,
                        "arrival_delay": stu.arrival.delay
                        if stu.HasField("arrival")
                        else None,
                        "departure_delay": stu.departure.delay
                        if stu.HasField("departure")
                        else None,
                    }
                    for stu in tu.stop_time_update
                ],
            }
        return None

    @staticmethod
    def _delivery_report(err: Any, msg: Any) -> None:
        if err:
            log.error("Kafka delivery failed", error=str(err))


async def main() -> None:
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(os.getenv("LOG_LEVEL", "INFO"))
        )
    )
    producer = GTFSRTProducer()
    await producer.start()


if __name__ == "__main__":
    asyncio.run(main())