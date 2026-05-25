# GCP Onboarding Guide — Bangalore Transit Digital Twin

This guide takes you from zero GCP account to a fully running cloud infrastructure.
Everything is free-tier safe for the first 90 days ($300 credit). Estimated monthly
cost after free tier: ~$80–120/month for a minimal production cluster.

---

## Step 1 — Create your GCP Account

1. Go to https://cloud.google.com → click **Get started for free**
2. Sign in with a Google account (create one if needed)
3. Enter billing info — **you will NOT be charged** during the $300 free trial
4. Select **India** as your country → timezone: **Asia/Kolkata**
5. After account creation, you land in the GCP Console

> Your $300 credit lasts 90 days. This project will consume roughly $30–50 of that
> during development, leaving plenty of headroom.

---

## Step 2 — Create a Project

In the GCP Console top bar, click the project dropdown → **New Project**

```
Project name:  bangalore-transit-twin
Project ID:    bangalore-transit-twin   (or add a random suffix if taken)
```

Note your **Project ID** — you'll use it everywhere. Set it in your shell:

```bash
export GCP_PROJECT_ID=bangalore-transit-twin
gcloud config set project $GCP_PROJECT_ID
```

---

## Step 3 — Install Google Cloud SDK in WSL

Run this inside your WSL terminal (not PowerShell):

```bash
# Install the SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL   # restart shell to pick up PATH

# Authenticate
gcloud auth login                         # opens browser
gcloud auth application-default login    # for SDK/Python access

# Set your project
gcloud config set project bangalore-transit-twin
gcloud config set compute/region asia-south1    # Mumbai — closest to Bangalore
gcloud config set compute/zone asia-south1-a
```

Verify:
```bash
gcloud config list
# Should show your project, region, zone
```

---

## Step 4 — Enable Required APIs

Run this block once — it enables all GCP services this project needs:

```bash
gcloud services enable \
  container.googleapis.com \
  storage.googleapis.com \
  sqladmin.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com \
  monitoring.googleapis.com \
  logging.googleapis.com \
  pubsub.googleapis.com \
  dataflow.googleapis.com
```

This takes ~2 minutes. You'll see `Operation finished successfully` for each.

---

## Step 5 — Create a Service Account

Your app (Airflow, PyIceberg, MLflow) needs a service account to authenticate:

```bash
# Create the service account
gcloud iam service-accounts create transit-twin-sa \
  --display-name="Transit Twin Service Account" \
  --description="Used by all Transit Twin services"

# Grant required roles
SA_EMAIL="transit-twin-sa@${GCP_PROJECT_ID}.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/container.developer"

gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/artifactregistry.writer"

# Download the key JSON — store it securely, NEVER commit to git
gcloud iam service-accounts keys create \
  ~/.config/gcp/transit-twin-sa-key.json \
  --iam-account=${SA_EMAIL}

echo "Key saved to ~/.config/gcp/transit-twin-sa-key.json"
```

Now update your `.env`:
```bash
# In your project root:
cp .env.example .env

# Edit .env and set:
GCP_PROJECT_ID=bangalore-transit-twin
GOOGLE_APPLICATION_CREDENTIALS=/home/YOUR_WSL_USERNAME/.config/gcp/transit-twin-sa-key.json
```

---

## Step 6 — Create GCS Buckets (your Iceberg Lakehouse storage)

```bash
# Main lakehouse bucket (multi-regional for durability)
gsutil mb -p $GCP_PROJECT_ID \
  -c STANDARD \
  -l ASIA \
  gs://bangalore-transit-twin-prod/

# Create the Medallion layer prefixes
gsutil -m cp /dev/null gs://bangalore-transit-twin-prod/lakehouse/bronze/.keep
gsutil -m cp /dev/null gs://bangalore-transit-twin-prod/lakehouse/silver/.keep
gsutil -m cp /dev/null gs://bangalore-transit-twin-prod/lakehouse/gold/.keep
gsutil -m cp /dev/null gs://bangalore-transit-twin-prod/mlflow-artifacts/.keep
gsutil -m cp /dev/null gs://bangalore-transit-twin-prod/flink-checkpoints/.keep

# Enable versioning on the lakehouse bucket (important for Iceberg time travel)
gsutil versioning set on gs://bangalore-transit-twin-prod/

echo "✅ GCS buckets ready"
```

Update `.env`:
```
GCS_BUCKET=bangalore-transit-twin-prod
```

---

## Step 7 — Create GKE Cluster (Autopilot — cheapest for dev)

GKE Autopilot automatically manages nodes and only charges for actual pod resource
usage. Perfect for this project — no idle node costs.

```bash
gcloud container clusters create-auto bangalore-transit-cluster \
  --region=asia-south1 \
  --project=$GCP_PROJECT_ID \
  --release-channel=stable

# Get kubeconfig credentials
gcloud container clusters get-credentials bangalore-transit-cluster \
  --region=asia-south1

# Verify
kubectl get nodes
# Should show Autopilot nodes managed by GKE
```

---

## Step 8 — Create Artifact Registry (Docker image store)

```bash
gcloud artifacts repositories create transit-twin-repo \
  --repository-format=docker \
  --location=asia-south1 \
  --description="Transit Twin Docker images"

# Configure Docker to use it
gcloud auth configure-docker asia-south1-docker.pkg.dev

echo "✅ Artifact Registry ready"
echo "Image prefix: asia-south1-docker.pkg.dev/${GCP_PROJECT_ID}/transit-twin-repo"
```

---

## Step 9 — Store Secrets in Secret Manager

**Never** put real API keys in `.env` for production. Use Secret Manager:

```bash
# Create secrets
echo -n "your-openweather-key" | \
  gcloud secrets create openweather-api-key --data-file=-

echo -n "your-otd-api-key" | \
  gcloud secrets create otd-api-key --data-file=-

# Grant your service account access
gcloud secrets add-iam-policy-binding openweather-api-key \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding otd-api-key \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/secretmanager.secretAccessor"
```

---

## Step 10 — Set GitHub Actions Secrets

Go to your GitHub repo → **Settings → Secrets and Variables → Actions** → **New secret**

Add these:

| Secret Name | Value |
|---|---|
| `GCP_PROJECT_ID` | `bangalore-transit-twin` |
| `GCP_SA_KEY` | Contents of `~/.config/gcp/transit-twin-sa-key.json` |
| `GCS_BUCKET` | `bangalore-transit-twin-prod` |
| `MLFLOW_TRACKING_URI` | Your MLflow URL (after deploying) |
| `RAY_CLUSTER_ADDRESS` | Your Ray cluster address (after deploying) |

---

## Step 11 — Verify End-to-End (local → GCS test)

```bash
# From your WSL terminal, in the project root:
pip install google-cloud-storage

python - << 'EOF'
from google.cloud import storage
client = storage.Client()
bucket = client.bucket("bangalore-transit-twin-prod")
blob = bucket.blob("test/hello.txt")
blob.upload_from_string("Transit Twin GCS connection works!")
print("✅ GCS write successful")
print(blob.public_url)
EOF
```

---

## Cost Breakdown (Approximate)

| Service | Spec | Monthly Cost |
|---|---|---|
| GKE Autopilot | ~4 vCPU, 8GB RAM avg | ~$60 |
| GCS (Lakehouse) | ~50GB storage + ops | ~$5 |
| Artifact Registry | ~5GB images | ~$1 |
| Cloud Logging | Free tier | $0 |
| **Total** | | **~$66/month** |

> During development, actual usage is much lower. Suspend the GKE cluster when not
> working: `gcloud container clusters resize bangalore-transit-cluster --num-nodes=0`

---

## WSL-Specific Tips

**Docker Desktop integration with WSL:**
- Open Docker Desktop → Settings → Resources → WSL Integration
- Enable for your WSL distro (Ubuntu)
- Verify: `docker ps` works in WSL terminal

**Slow file I/O in WSL?** Always work in the Linux filesystem:
```bash
# GOOD: /home/yourname/projects/bangalore-transit-twin
# BAD: /mnt/c/Users/yourname/projects/...
```

**Port forwarding:** Docker Desktop auto-forwards ports from WSL to Windows,
so `localhost:8000` in your browser reaches the FastAPI container.

**VS Code:** Install the "WSL" extension → open your project with `code .`
from inside WSL. Everything (terminal, extensions, git) runs in Linux context.

---

## What's Next After GCP Setup

Once your GCP account is live and these steps are done, the next phase is:

1. **Run the local stack** — `docker-compose up -d` and verify all 8 services start
2. **First data ingestion** — trigger the Airflow DAG manually to pull BMTC GTFS
3. **Verify Kafka** — open Kafka UI at `localhost:8080`, confirm topics exist
4. **Train the first ETA model** — `python -m mlops_pipeline.training.train_eta_model --local --synthetic`
5. **Open MLflow** at `localhost:5000` and see the first experiment run