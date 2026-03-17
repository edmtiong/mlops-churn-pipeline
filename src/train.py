import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
from pathlib import Path

# MLflow experiment name — groups all runs together

def load_raw():
    path = Path(__file__).resolve().parents[1] / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df = df.drop(columns=["customerID"])
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df.drop(columns=["Churn"]), df["Churn"]

def train_model(model, model_name: str):
    X, y = load_raw()

    cat_cols = list(X.select_dtypes(include="object").columns)
    num_cols = list(X.select_dtypes(exclude="object").columns)

    preprocessor = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ("num", StandardScaler(), num_cols)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run(run_name=model_name):
        pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = pipeline.predict(X_test)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        # Log params and metrics
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("roc_auc", roc_auc)

        report          = classification_report(y_test, y_pred, output_dict=True)
        churn_recall    = report["1"]["recall"]
        churn_precision = report["1"]["precision"]

        mlflow.log_metric("churn_recall",    churn_recall)
        mlflow.log_metric("churn_precision", churn_precision)
        mlflow.sklearn.log_model(pipeline, name="model")

        print(f"\n{'='*50}")
        print(f"Model: {model_name}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5001")
    mlflow.set_experiment("churn-pipeline")

    train_model(
        LogisticRegression(random_state=42, max_iter=1000),
        "LR-pipeline"
    )
    train_model(
        RandomForestClassifier(random_state=42, n_estimators=100),
        "RF-pipeline"
    )