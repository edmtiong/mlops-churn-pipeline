import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from src.data.preprocess import load_and_preprocess

# MLflow experiment name — groups all runs together
EXPERIMENT_NAME = "churn-baseline"


def train_model(model, model_name: str, params: dict):
    X, y = load_and_preprocess()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=model_name):
        # Model
        model.fit(X_train, y_train)

        # Evaluate
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

       # Log params and metrics
        mlflow.log_param("model_type", model_name)
        for key, value in params.items():
            mlflow.log_param(key, value)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.sklearn.log_model(model, name="model")

        print(f"\n{'='*50}")
        print(f"Model: {model_name}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")


if __name__ == "__main__":
    # Run 1 — Logistic Regression
    train_model(
        model=LogisticRegression(max_iter=1000, random_state=42),
        model_name="LogisticRegression",
        params={"max_iter": 1000, "test_size": 0.2}
    )

    # Run 2 — Random Forest
    train_model(
        model=RandomForestClassifier(n_estimators=100, random_state=42),
        model_name="RandomForest",
        params={"n_estimators": 100, "test_size": 0.2}
    )