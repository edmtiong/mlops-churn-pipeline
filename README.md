# MLOps Churn Prediction Pipeline

A full ML platform deployed on Kubernetes, built to mirror production MLOps patterns.

## Flow

```
[ Data ] → [ Train ] → [ Register ] → [ Promote @champion ]
                                                ↓
[ Response ] ← [ Request ] ← [ Monitor ] ← [ Serve ]
      ↓
[ Retrain (Prefect, Mon 2am) ] - - - - - - - ↑
```

## Stack

| Layer | Tools |
|---|---|
| Serving | FastAPI, Kubernetes, Nginx Ingress, HPA |
| Experiment tracking | MLflow, PostgreSQL, MinIO |
| Orchestration | Prefect (weekly retraining, cron) |
| Observability | Prometheus, Grafana |
| CI/CD | GitHub Actions, Docker |

## Model performance

| Metric | Value |
|---|---|
| ROC-AUC | 0.84 |
| p95 latency | ~30ms (warm) |
| Algorithm | Logistic Regression pipeline |

---

## Prerequisites

- Docker + Docker Compose
- minikube + kubectl (for Kubernetes deployment)
- Python 3.12+

---

## Option 1 — Run locally with Docker Compose

### 1. Clone the repo

```bash
git clone https://github.com/edmtiong/mlops-churn-pipeline.git
cd mlops-churn-pipeline
```

### 2. Start all services

```bash
docker compose up --build
```

This starts Postgres, MinIO, MLflow, FastAPI, Prometheus, and Grafana.
Wait until all services are healthy before proceeding (~60s for MLflow).

### 3. Train and register the initial model

The first time you run the stack, MLflow has no model registered yet.
Run the training pipeline to train and register the champion model:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
export MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin

pip install -r requirements.txt
python src/pipelines/retrain_flow.py
```

Or use the helper script:

```bash
bash start.sh
```

### 4. Services

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000 |
| MLflow | http://localhost:5001 |
| Grafana | http://localhost:3000 (admin/admin) |
| Prometheus | http://localhost:9090 |
| MinIO console | http://localhost:9001 (minioadmin/minioadmin) |

---

## Option 2 — Deploy on Kubernetes (minikube)

### 1. Clone the repo

```bash
git clone https://github.com/edmtiong/mlops-churn-pipeline.git
cd mlops-churn-pipeline
```

### 2. Start minikube

```bash
minikube start --driver=docker --kubernetes-version=v1.32.0
minikube tunnel  # keep running in a separate terminal
```

### 3. Apply manifests

```bash
kubectl apply -f k8s/namespaces/
kubectl apply -f k8s/secrets/
kubectl apply -f k8s/configmaps/
kubectl apply -f k8s/data/
kubectl apply -f k8s/app/
kubectl apply -f k8s/monitoring/
kubectl apply -f k8s/ingress/
```

### 4. Verify cluster

```bash
kubectl get pods -n mlops-app
kubectl get pods -n mlops-data
kubectl get hpa -n mlops-app
```

### 5. Train and register the initial model

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1/mlflow
export MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin

pip install -r requirements.txt
python src/pipelines/retrain_flow.py
```

### 6. Services

| Service | URL |
|---|---|
| FastAPI | http://127.0.0.1/fastapi/health |
| MLflow | http://127.0.0.1/mlflow |
| Grafana | http://127.0.0.1:3000 (admin/admin) |
| Prometheus | http://127.0.0.1/prometheus |

---

## Run a prediction

### Docker Compose

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

### Kubernetes

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

---

## Trigger retraining manually

Retraining runs automatically every Monday at 2am via Prefect cron schedule.
To trigger it manually:

```bash
kubectl exec -n mlops-app deployment/prefect -- \
  prefect deployment run 'churn-retraining/churn-retraining-deployment'
```
