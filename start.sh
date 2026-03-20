#!/bin/bash
export PREFECT_API_URL=http://127.0.0.1:4200/api
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
export MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
python src/pipelines/retrain_flow.py