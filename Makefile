# ─────────────────────────────────────────────────────────────────────────────
#  Bangalore Transit Digital Twin — Makefile
#  Usage: make <target>
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: help up down logs ps test lint train-local kafka-topics status

PYTHON := python3
PIP    := pip

# ── Dev environment ───────────────────────────────────────────────────────────
help:                  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

install:               ## Install Python deps in editable mode
	$(PIP) install -e ".[dev]"

up:                    ## Start all local services (Kafka, Flink, Airflow, MLflow, Redis, MinIO)
	docker compose up -d
	@echo ""
	@echo "  Services starting up:"
	@echo "  ✅ Kafka UI    → http://localhost:8080"
	@echo "  ✅ Flink UI    → http://localhost:8082"
	@echo "  ✅ Airflow     → http://localhost:8083  (admin/admin)"
	@echo "  ✅ MLflow      → http://localhost:5000"
	@echo "  ✅ MinIO       → http://localhost:9001  (minioadmin/minioadmin)"
	@echo "  ✅ Transit API → http://localhost:8000"
	@echo ""

down:                  ## Stop all local services
	docker compose down

reset:                 ## Stop all services AND delete all volumes (fresh start)
	docker compose down -v

logs:                  ## Tail logs from all services
	docker compose logs -f

ps:                    ## Show status of all containers
	docker compose ps

status:                ## Quick health check of all services
	@echo "── Kafka ──────────────────────────────────"
	@curl -sf http://localhost:8081/subjects | python3 -m json.tool || echo "Schema Registry: DOWN"
	@echo "── MLflow ─────────────────────────────────"
	@curl -sf http://localhost:5000/health | python3 -m json.tool || echo "MLflow: DOWN"
	@echo "── Transit API ────────────────────────────"
	@curl -sf http://localhost:8000/health | python3 -m json.tool || echo "Transit API: DOWN"
	@echo "── MinIO ──────────────────────────────────"
	@curl -sf http://localhost:9000/minio/health/live && echo "MinIO: UP" || echo "MinIO: DOWN"

# ── Kafka ─────────────────────────────────────────────────────────────────────
kafka-topics:          ## Create all required Kafka topics
	docker compose exec kafka kafka-topics \
	  --bootstrap-server localhost:9092 --create --if-not-exists \
	  --topic vehicle-positions --partitions 4 --replication-factor 1
	docker compose exec kafka kafka-topics \
	  --bootstrap-server localhost:9092 --create --if-not-exists \
	  --topic trip-updates --partitions 4 --replication-factor 1
	docker compose exec kafka kafka-topics \
	  --bootstrap-server localhost:9092 --create --if-not-exists \
	  --topic flink-delay-output --partitions 2 --replication-factor 1
	docker compose exec kafka kafka-topics \
	  --bootstrap-server localhost:9092 --create --if-not-exists \
	  --topic flink-bunching-alerts --partitions 2 --replication-factor 1
	@echo "✅ Kafka topics ready"

kafka-list:            ## List all Kafka topics
	docker compose exec kafka kafka-topics \
	  --bootstrap-server localhost:9092 --list

# ── Data pipeline ─────────────────────────────────────────────────────────────
ingest-gtfs:           ## Manually trigger GTFS static ingestion (local)
	$(PYTHON) -c "\
from data_engineering.ingestion.gtfs_static.parser import download_gtfs_zip, parse_gtfs_zip; \
import os; \
url=os.getenv('BMTC_GTFS_STATIC_URL','https://bmtcwebportal.pascos.in/gtfs/bmtc_gtfs.zip'); \
print(f'Downloading from {url}...'); \
raw=download_gtfs_zip(url); \
feed=parse_gtfs_zip('bmtc', raw); \
print(feed)"

producer-start:        ## Start the GTFS-RT Kafka producer
	$(PYTHON) -m stream_processing.kafka.producers.gtfs_rt_producer

redis-writer-start:    ## Start the Kafka→Redis vehicle position writer
	$(PYTHON) -m serving_layer.cache.kafka_to_redis

api-start:             ## Start the FastAPI server (dev mode, hot reload)
	uvicorn serving_layer.api.main:app --reload --host 0.0.0.0 --port 8000

# ── ML pipeline ───────────────────────────────────────────────────────────────
train-local:           ## Train ETA model locally on synthetic data
	$(PYTHON) -m mlops_pipeline.training.train_eta_model --local --synthetic

train-cloud:           ## Train ETA model on Ray cluster (requires GCP + KubeRay)
	$(PYTHON) -m mlops_pipeline.training.train_eta_model --ray-address auto

# ── Testing ───────────────────────────────────────────────────────────────────
test:                  ## Run full test suite
	pytest tests/ -v --tb=short

test-unit:             ## Run unit tests only (fast, no external services)
	pytest tests/unit/ -v --tb=short

test-cov:              ## Run tests with coverage report
	pytest tests/ --cov=. --cov-report=html --cov-report=term-missing
	@echo "Coverage report: open htmlcov/index.html"

# ── Code quality ──────────────────────────────────────────────────────────────
lint:                  ## Run ruff linter
	ruff check .

lint-fix:              ## Auto-fix ruff issues
	ruff check . --fix

typecheck:             ## Run mypy type checker
	mypy data_engineering stream_processing serving_layer --ignore-missing-imports

fmt:                   ## Format code with ruff
	ruff format .

# ── Git helpers ───────────────────────────────────────────────────────────────
pre-commit-install:    ## Install pre-commit hooks
	pre-commit install

pre-commit-run:        ## Run all pre-commit hooks on staged files
	pre-commit run --all-files