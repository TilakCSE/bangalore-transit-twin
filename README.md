# Bangalore Transit Digital Twin
### Real-time 3D Smart City Platform — Applied AI & Data Architecture Portfolio

[![CI/CD](https://github.com/yourusername/bangalore-transit-twin/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/bangalore-transit-twin/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Unity 6](https://img.shields.io/badge/Unity-6_HDRP-black.svg)](https://unity.com/)

A production-grade, end-to-end Applied AI platform that ingests live BMTC bus and Namma Metro GTFS-RT feeds, processes them through a distributed streaming engine, trains ML models for ETA prediction, and renders everything in a photorealistic 3D city digital twin built in Unity 6 HDRP with Blender-authored assets.

---

## Architecture Overview

```
GTFS-RT Feeds (BMTC + Namma Metro)
        │
        ▼
  Apache Kafka  ──────────────────────────────────────────┐
        │                                                  │
        ▼                                                  │
  Apache Flink (stream processing)                        │
  - Delay detection & bunching alerts                     │
  - Spatial joins via MobilityDB                          │
  - Real-time feature generation                          │
        │                                                  │
        ├──────────────────────┐                           │
        ▼                      ▼                           │
  Iceberg Lakehouse      FastAPI + WebSocket               │
  (Bronze/Silver/Gold)   serving layer                     │
        │                      │                           │
        ▼                      ▼                           │
  MLOps Pipeline         Unity 6 HDRP                     │
  (Ray + MLflow)         3D Digital Twin  ◄────────────────┘
  LSTM ETA model         Blender assets
```

## Key Features

- **Live data ingestion** from BMTC GTFS-RT and Namma Metro APIs
- **Sub-50ms latency** stream processing with Apache Flink
- **Medallion Lakehouse** on GCS using Apache Iceberg + dbt
- **LSTM/TCN ETA prediction** trained on 3+ years of historical GTFS data
- **Photorealistic 3D twin** in Unity 6 HDRP with Blender-authored city assets
- **DVR replay** — scrub through any historical date and watch traffic unfold
- **Congestion heatmaps** rendered as dynamic shader overlays on road meshes
- **LLM RAG agent** — ask natural language questions about the transit network
- **Full CI/CD** via GitHub Actions with automated model retraining triggers

## Stack

| Layer | Technology |
|---|---|
| Ingestion | Python, GTFS-RT protobuf, Confluent Kafka |
| Stream Processing | Apache Flink 1.19, Flink SQL, PyFlink |
| Storage | Apache Iceberg, GCS, dbt |
| Orchestration | Apache Airflow 2.9, Docker, Kubernetes |
| ML Training | PyTorch, Ray Train, MLflow, DVC |
| Serving | FastAPI, WebSocket, Redis, KServe |
| 3D Rendering | Unity 6 HDRP, Blender 4.x, C# |
| LLMOps | vLLM, Haystack, Milvus |
| Infrastructure | GKE, Terraform, Helm, KubeRay |
| CI/CD | GitHub Actions, Pytest, Great Expectations |

## Repository Structure

```
bangalore-transit-twin/
├── infrastructure/          # Terraform (GCP), Kubernetes Helm charts, Docker
├── data_engineering/        # Airflow DAGs, dbt models, ingestion scripts
├── stream_processing/       # Flink jobs, Kafka producers/consumers, Avro schemas
├── mlops_pipeline/          # Training, evaluation, feature store, model registry
├── serving_layer/           # FastAPI, WebSocket server, Redis cache
├── unity_client/            # C# scripts for data binding and 3D rendering
├── agentic_llm/             # RAG pipeline, LLM agents, Haystack
├── notebooks/               # EDA, model prototyping, architecture experiments
├── tests/                   # Unit, integration, e2e test suites
├── docs/                    # Architecture diagrams, ADRs, API specs
└── .github/workflows/       # CI/CD pipelines
```

## Quickstart (Local Development)

### Prerequisites
- Docker Desktop 4.x
- Python 3.11+
- Java 11+ (for Flink)
- GCP account with billing enabled

```bash
# 1. Clone and configure
git clone https://github.com/yourusername/bangalore-transit-twin.git
cd bangalore-transit-twin
cp .env.example .env  # fill in your GCP project ID, API keys

# 2. Spin up local infrastructure (Kafka, Flink, Airflow, MLflow)
docker-compose up -d

# 3. Install Python dependencies
pip install -e ".[dev]"

# 4. Run the GTFS-RT producer (starts streaming live BMTC data)
python stream_processing/kafka/producers/gtfs_rt_producer.py

# 5. Submit the Flink delay-detection job
python stream_processing/flink/jobs/submit_jobs.py

# 6. Launch the serving API
uvicorn serving_layer.api.main:app --reload --port 8000
```

## Data Sources

| Source | Format | Update Frequency | License |
|---|---|---|---|
| BMTC GTFS Static | GTFS ZIP | Weekly | Open (ODbL) |
| BMTC GTFS-RT | Protobuf | 15s | Open |
| Namma Metro GTFS-RT | Protobuf | 10s | Open |
| OpenWeatherMap | REST JSON | 10min | Free tier |
| OpenStreetMap (Bangalore) | PBF | Daily snapshot | ODbL |

## Roadmap

- [x] Project scaffold and infrastructure provisioning
- [x] GTFS-RT Kafka producer (BMTC + Metro)
- [x] Flink delay detection job
- [x] Iceberg lakehouse with dbt Medallion models
- [ ] LSTM ETA prediction model (Ray Train + MLflow)
- [ ] FastAPI + WebSocket serving layer
- [ ] Blender city asset pipeline
- [ ] Unity 6 HDRP scene + C# data bridge
- [ ] Congestion heatmap shader
- [ ] DVR time-scrubber feature
- [ ] LLM RAG agent
- [ ] Full CI/CD + GitHub Actions

## Contributing

This is an open portfolio project. Issues and PRs are welcome.

## License

MIT — see [LICENSE](LICENSE)
