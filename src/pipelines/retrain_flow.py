import os
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from prefect import flow, task, get_run_logger

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ── Task 1: Load data ─────────────────────────────────────────────────────────

@task(name="load-data", retries=2, retry_delay_seconds=5, log_prints=True)
def load_data(path: str = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv") -> pd.DataFrame:
    logger = get_run_logger()
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    return df


# ── Task 2: Preprocess ────────────────────────────────────────────────────────

@task(name="preprocess", log_prints=True)
def preprocess(df: pd.DataFrame):
    logger = get_run_logger()

    # Drop customerID, fix TotalCharges
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    X = df.drop(columns=["Churn"])
    y = (df["Churn"] == "Yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    return X_train, X_test, y_train, y_test


# ── Task 3: Train and log to MLflow ──────────────────────────────────────────

@task(name="train-and-log", log_prints=True)
def train_and_log(X_train, X_test, y_train, y_test):
    logger = get_run_logger()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
    mlflow.set_experiment("churn-prefect")

    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    num_cols = X_train.select_dtypes(include="number").columns.tolist()

    with mlflow.start_run() as run:
        pipeline = Pipeline([
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ])
        pipeline.fit(X_train, y_train)

        acc = accuracy_score(y_test, pipeline.predict(X_test))
        mlflow.log_metric("accuracy", acc)
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.sklearn.log_model(pipeline, "model", registered_model_name="churn-model")

        logger.info(f"Accuracy: {acc:.4f} | Run ID: {run.info.run_id}")
        return run.info.run_id, acc


# ── Task 4: Promote if better than current champion ──────────────────────────

@task(name="promote-if-better", log_prints=True)
def promote_if_better(run_id: str, new_accuracy: float):
    logger = get_run_logger()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
    client = MlflowClient()

    # Find current champion accuracy
    try:
        champion_versions = client.get_model_version_by_alias("churn-model", "champion")
        champ_run = mlflow.get_run(champion_versions.run_id)
        champ_acc = float(champ_run.data.metrics.get("accuracy", 0))
        logger.info(f"Champion accuracy: {champ_acc:.4f}")
    except Exception:
        champ_acc = 0.0
        logger.info("No champion found — will promote automatically")

    if new_accuracy > champ_acc:
        # Find the new model version by run_id
        versions = client.search_model_versions(f"run_id='{run_id}'")
        if versions:
            new_version = versions[0].version
            client.set_registered_model_alias("churn-model", "champion", new_version)
            logger.info(f"Promoted v{new_version} to champion ({new_accuracy:.4f} > {champ_acc:.4f})")
    else:
        logger.info(f"No promotion — new model ({new_accuracy:.4f}) did not beat champion ({champ_acc:.4f})")


# ── Flow ──────────────────────────────────────────────────────────────────────

@flow(name="churn-retraining")
def retrain_flow():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)
    run_id, accuracy = train_and_log(X_train, X_test, y_train, y_test)
    promote_if_better(run_id, accuracy)


if __name__ == "__main__":
    retrain_flow.serve(
        name="churn-retraining-deployment",
        cron="0 2 * * 1",
    )