# MLOps Churn Prediction Pipeline

A full ML platform deployed on Kubernetes, built to mirror production MLOps patterns.

## Architecture

![Architecture Diagram](docs/architecture.png)

## Stack

| Layer | Tools |
|---|---|
| Serving | FastAPI, Kubernetes, Nginx Ingress, HPA |
| Experiment tracking | MLflow, PostgreSQL, MinIO |
| Orchestration | Prefect (weekly retraining, cron) |
| Observability | Prometheus, Grafana |
| CI/CD | GitHub Actions, Docker |

## What it does

- Serves a churn prediction model via REST API (`POST /predict`)
- Champion model auto-loaded from MLflow registry at startup
- Weekly retraining triggered by Prefect every Monday 2am
- Autoscales FastAPI pods on CPU > 70% (1–5 replicas)
- Grafana dashboards: p95 latency, request rate, churn distribution

## Model performance

| Metric | Value |
|---|---|
| ROC-AUC | 0.84 |
| p95 latency | ~30ms (warm) |
| Algorithm | Logistic Regression pipeline |

## Prerequisites

- Docker + Docker Compose
- minikube + kubectl (for Kubernetes deployment)

## Option 1 — run locally with Docker Compose
```bash
# Clone the repo
git clone https://github.com/edmtiong/mlops-churn-pipeline.git
cd mlops-churn-pipeline

# Start all services (Postgres, MinIO, MLflow, FastAPI, Prometheus, Grafana)
docker compose up --build
```

Services will be available at:

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000 |
| MLflow | http://localhost:5001 |
| Grafana | http://localhost:3000 (admin/admin) |
| Prometheus | http://localhost:9090 |
| MinIO console | http://localhost:9001 (minioadmin/minioadmin) |

## Option 2 — deploy on Kubernetes (minikube)
```bash
# Start minikube
minikube start --driver=docker --kubernetes-version=v1.32.0
minikube tunnel  # keep this running in a separate terminal

# Apply manifests in order
kubectl apply -f k8s/namespaces/
kubectl apply -f k8s/secrets/
kubectl apply -f k8s/configmaps/
kubectl apply -f k8s/data/
kubectl apply -f k8s/app/
kubectl apply -f k8s/monitoring/
kubectl apply -f k8s/ingress/

# Verify everything is running
kubectl get pods -n mlops-app
kubectl get pods -n mlops-data
```

Services will be available at:

| Service | URL |
|---|---|
| FastAPI | http://127.0.0.1/fastapi/health |
| MLflow | http://127.0.0.1/mlflow |
| Grafana | http://127.0.0.1:3000 (admin/admin) |
| Prometheus | http://127.0.0.1/prometheus |

## Run a prediction
```bash
curl -X POST http://127.0.0.1/fastapi/predict \
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

Expected response:
```json
{"churn_probability": 0.23, "prediction": 0}
```

## Trigger retraining manually
```bash
kubectl exec -n mlops-app deployment/prefect -- \
  prefect deployment run 'churn-retraining/churn-retraining-deployment'
```

Retraining also runs automatically every Monday at 2am via Prefect cron schedule.