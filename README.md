# ML Platform — Model Lifecycle and Serving Infrastructure

A self-hosted platform for training, registering, promoting, serving, and monitoring ML models on Kubernetes.

Built to answer a specific question: **what does it actually take to put a model into production when you have to be able to prove, afterwards, how it got there?** The modelling problem is deliberately trivial. The lifecycle around it is the subject.

Demonstrated on a customer churn dataset.

---

## What this demonstrates

- **Model registry with a promotion path** — champion/challenger staging in MLflow, so a new model replaces a live one through a defined transition rather than a redeploy.
- **Zero-downtime promotion** — serving reads the champion alias from the registry; promotion does not restart the serving pods.
- **Scheduled retraining as an isolated workload** — retraining runs in its own container on its own schedule, and cannot take the serving path down with it.
- **Autoscaled inference** — FastAPI behind an nginx ingress with an HPA scaling 1→5 replicas on load.
- **Operational observability** — p95 latency, request rate, and prediction distribution instrumented in Prometheus and surfaced in Grafana.
- **Reproducible from zero** — one command locally, one manifest set on Kubernetes.

---

## Architecture

```
                          ┌─────────────────────────────┐
                          │  namespace: mlops-data      │
   ┌──────────┐  train    │  ┌────────┐   ┌───────────┐ │
   │ Prefect  │──────────▶│  │ MLflow │──▶│ PostgreSQL│ │  metadata
   │ retrain  │  register │  │ server │   └───────────┘ │
   │ (cron)   │           │  │        │──▶┌───────────┐ │  artifacts
   └──────────┘           │  └────────┘   │   MinIO   │ │
        ▲                 │       ▲       └───────────┘ │
        │                 └───────┼─────────────────────┘
        │                         │ load @champion
        │                 ┌───────┴─────────────────────┐
        │                 │  namespace: mlops-app       │
        │                 │  ┌─────────┐   ┌──────────┐ │
   ┌────┴─────┐           │  │ FastAPI │◀──│ Ingress  │◀─── requests
   │ schedule │           │  │  (HPA   │   │  nginx   │ │
   │ Mon 02:00│           │  │  1–5)   │   └──────────┘ │
   └──────────┘           │  └────┬────┘                │
                          │       │ /metrics            │
                          │  ┌────▼───────┐  ┌────────┐ │
                          │  │ Prometheus │─▶│ Grafana│ │
                          │  └────────────┘  └────────┘ │
                          └─────────────────────────────┘
```

Namespaces separate stateful platform services (`mlops-data`) from the request path (`mlops-app`), so serving can be scaled, restarted, or rolled back without touching the registry or its backing stores.

> **[ADD SCREENSHOT: `kubectl get pods -A` showing both namespaces healthy]**

---

## Design decisions
--- To be filled

## Model lifecycle

```
train ──▶ log run ──▶ register version ──▶ @challenger ──▶ [evaluation] ──▶ @champion
                                                                              │
                                                        serving reads ────────┘
```

Retraining runs weekly (Mondays, 02:00) via a Prefect cron schedule. Each run logs a new registered version. Promotion to `@champion` is the controlled step — serving resolves the champion alias at load, so promotion takes effect without a redeploy and rollback is a single alias change.

Manual trigger:

```bash
kubectl exec -n mlops-app deployment/prefect -- \
  prefect deployment run 'churn-retraining/churn-retraining-deployment'
```

> **[ADD SCREENSHOT: MLflow registry showing two versions with @champion and @challenger aliases]**

---

## Observability

| Metric | Why it's instrumented |
|---|---|
| p95 inference latency | Tail latency is what breaks SLAs; the mean hides it |
| Request rate | Drives HPA scaling behaviour and shows whether autoscaling responds to real load |
| Prediction distribution | A shift in output distribution is the earliest cheap signal that inputs have changed — a precursor to proper drift detection |

Measured p95 is ~30ms warm, on a single-node minikube cluster. This is a local-cluster figure, not a production benchmark.

> **[ADD SCREENSHOT: Grafana dashboard under load, showing latency and request rate]**
> **[ADD SCREENSHOT: `kubectl get hpa` mid-scale, showing replica count above 1]**

---

## Known limitations

Stated deliberately. This is a demonstration platform, not a production deployment.

- **Single-node minikube.** No multi-node scheduling, no pod anti-affinity, no real failure-domain separation. HPA scaling is demonstrated, not stress-tested.
- **No authentication on the inference endpoint.** `/predict` is open. A real deployment needs authn/authz at the ingress and per-caller rate limiting.
- **Local credentials are defaults.** `minioadmin/minioadmin` and `admin/admin` are fine for a laptop and unacceptable anywhere else. Kubernetes secrets are applied from plaintext manifests; a real deployment needs sealed secrets or an external secret store.
- **No automated evaluation gate.** Promotion to `@champion` is currently a manual decision. Nothing programmatically blocks a worse model from being promoted. *(Next on the roadmap — see below.)*
- **No drift detection.** Prediction distribution is observed but nothing acts on it; there is no input-drift monitoring and no drift-triggered retraining.
- **Infrastructure is not declarative end-to-end.** Manifests are applied by hand. No Terraform, no GitOps reconciliation.
- **Retraining uses a fixed dataset.** There is no upstream data pipeline, so "retraining" re-fits on the same data rather than on new arrivals.
- **Model quality is not the point.** ROC-AUC 0.84 from a logistic regression on a public churn dataset is unremarkable and intended to be. Its role here is as a *threshold* — a number the evaluation gate can enforce against — not as a result.

---

## Roadmap

| Status | Item |
|---|---|
| In progress | **CI evaluation gate** — score the challenger against a versioned held-out set and block promotion on metric regression beyond threshold |
| Planned | Input-drift detection, and retraining triggered by drift rather than by cron |
| Planned | Terraform for cluster and platform provisioning |
| Planned | Load-test harness with published latency-under-concurrency curves |

---

## Running it

<details>
<summary><b>Option 1 — Docker Compose (local)</b></summary>

**Prerequisites:** Docker + Docker Compose, Python 3.12+

```bash
git clone https://github.com/edmtiong/ml-lifecycle-platform.git
cd ml-lifecycle-platform
docker compose up --build
```

Starts Postgres, MinIO, MLflow, FastAPI, Prometheus, and Grafana. Allow ~60s for MLflow to become healthy.

Train and register the initial champion (MLflow starts with an empty registry):

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
export MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin

pip install -r requirements.txt
python src/pipelines/retrain_flow.py
```

Or: `bash start.sh`

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000 |
| MLflow | http://localhost:5001 |
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |
| MinIO console | http://localhost:9001 |

Default local credentials are `admin/admin` (Grafana) and `minioadmin/minioadmin` (MinIO). Local development only.

</details>

<details>
<summary><b>Option 2 — Kubernetes (minikube)</b></summary>

**Prerequisites:** minikube + kubectl, Python 3.12+

```bash
minikube start --driver=docker --kubernetes-version=v1.32.0
minikube tunnel   # keep running in a separate terminal
```

```bash
kubectl apply -f k8s/namespaces/
kubectl apply -f k8s/secrets/
kubectl apply -f k8s/configmaps/
kubectl apply -f k8s/data/
kubectl apply -f k8s/app/
kubectl apply -f k8s/monitoring/
kubectl apply -f k8s/ingress/
```

Verify:

```bash
kubectl get pods -n mlops-app
kubectl get pods -n mlops-data
kubectl get hpa -n mlops-app
```

Train and register the initial champion:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1/mlflow
export MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin

pip install -r requirements.txt
python src/pipelines/retrain_flow.py
```

| Service | URL |
|---|---|
| FastAPI | http://127.0.0.1/fastapi/health |
| MLflow | http://127.0.0.1/mlflow |
| Grafana | http://127.0.0.1:3000 |
| Prometheus | http://127.0.0.1/prometheus |

</details>

<details>
<summary><b>Run a prediction</b></summary>

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 600,
    "Age": 40,
    "Tenure": 5,
    "Balance": 50000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 80000,
    "Geography": "France",
    "Gender": "Male"
  }'
```

```json
{"churn_probability": 0.23, "prediction": 0}
```

On Kubernetes, substitute `http://127.0.0.1/fastapi/predict`.

</details>

---

## Stack

| Layer | Tools |
|---|---|
| Serving | FastAPI, Kubernetes, nginx Ingress, HPA |
| Registry & tracking | MLflow, PostgreSQL, MinIO |
| Orchestration | Prefect |
| Observability | Prometheus, Grafana |
| CI/CD | GitHub Actions, Docker |
