"""
Vehicle Position Redis Writer
─────────────────────────────────────────────────────────────────────────────
Kafka consumer that reads from 'vehicle-positions' and 'flink-delay-output'
topics and writes the merged state into Redis.

The FastAPI WebSocket server reads from Redis, NOT directly from Kafka.
This decouples the serving layer from Kafka consumer group lag and lets
multiple API replicas read consistent vehicle state without coordination.

Redis key schema:
  vehicle:{vehicle_id}   → JSON blob, TTL = 90s
  alert:{alert_id}       → JSON blob, TTL = 300s
  route_index:{route_id} → SET of vehicle_ids on this route (for bunching viz)

Run:
    python -m serving_layer.cache.kafka_to_redis
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import time
import uuid

import redis.asyncio as aioredis
import structlog
from confluent_kafka import Consumer, KafkaError, KafkaException
from dotenv import load_dotenv

load_dotenv()
log = structlog.get_logger(__name__)

KAFKA_BOOTSTRAP   = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
REDIS_URL         = os.getenv("REDIS_URL", "redis://localhost:6379")
TTL_VEHICLE       = int(os.getenv("REDIS_TTL_VEHICLE_POSITIONS", "90"))
TTL_ALERT         = int(os.getenv("REDIS_TTL_ALERTS", "300"))


class KafkaToRedisWriter:
    def __init__(self) -> None:
        self._running = True
        self._redis: aioredis.Redis | None = None

    async def start(self) -> None:
        self._redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGTERM, self._shutdown)

        # Run position consumer and alert consumer concurrently
        await asyncio.gather(
            self._consume_positions(),
            self._consume_alerts(),
        )

    def _shutdown(self) -> None:
        log.info("Shutting down Kafka→Redis writer")
        self._running = False

    async def _consume_positions(self) -> None:
        consumer = Consumer({
            "bootstrap.servers": KAFKA_BOOTSTRAP,
            "group.id": "redis-position-writer",
            "auto.offset.reset": "latest",
            "enable.auto.commit": True,
        })
        consumer.subscribe([
            os.getenv("TOPIC_VEHICLE_POSITIONS", "vehicle-positions"),
        ])
        log.info("Position consumer started")
        try:
            while self._running:
                msg = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: consumer.poll(timeout=1.0)
                )
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() != KafkaError._PARTITION_EOF:
                        log.error("Kafka error", error=msg.error())
                    continue
                try:
                    data = json.loads(msg.value().decode())
                    vid = data.get("vehicle_id")
                    if not vid:
                        continue
                    pipe = self._redis.pipeline()
                    # Store vehicle state
                    pipe.set(f"vehicle:{vid}", json.dumps(data), ex=TTL_VEHICLE)
                    # Maintain route index for bunching viz
                    route_id = data.get("route_id")
                    if route_id:
                        pipe.sadd(f"route_index:{route_id}", vid)
                        pipe.expire(f"route_index:{route_id}", TTL_VEHICLE)
                    await pipe.execute()
                except Exception as exc:
                    log.error("Position write failed", error=str(exc))
        finally:
            consumer.close()

    async def _consume_alerts(self) -> None:
        consumer = Consumer({
            "bootstrap.servers": KAFKA_BOOTSTRAP,
            "group.id": "redis-alert-writer",
            "auto.offset.reset": "latest",
        })
        consumer.subscribe([
            os.getenv("TOPIC_FLINK_DELAYS", "flink-delay-output"),
            os.getenv("TOPIC_FLINK_BUNCHING", "flink-bunching-alerts"),
        ])
        log.info("Alert consumer started")
        try:
            while self._running:
                msg = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: consumer.poll(timeout=1.0)
                )
                if msg is None:
                    continue
                if msg.error():
                    continue
                try:
                    data = json.loads(msg.value().decode())
                    alert_id = str(uuid.uuid4())
                    data["alert_id"] = alert_id
                    data["created_at"] = time.time()
                    # Classify severity
                    delay = abs(data.get("delay_sec", 0) or 0)
                    data["severity"] = "CRITICAL" if delay > 600 else "WARNING" if delay > 120 else "INFO"
                    await self._redis.set(
                        f"alert:{alert_id}", json.dumps(data), ex=TTL_ALERT
                    )
                except Exception as exc:
                    log.error("Alert write failed", error=str(exc))
        finally:
            consumer.close()


async def main() -> None:
    writer = KafkaToRedisWriter()
    await writer.start()


if __name__ == "__main__":
    asyncio.run(main())