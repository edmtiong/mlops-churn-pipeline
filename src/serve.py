import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter
import os

app = FastAPI(title="Churn Prediction API")
model = None

# --- Prometheus metrics ---
churn_counter = Counter(
    "churn_predictions_total",
    "Total predictions by outcome",
    ["prediction"]
)

Instrumentator().instrument(app).expose(app)

class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.on_event("startup")
async def load_model():
    global model
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    model = mlflow.sklearn.load_model("models:/churn-model@champion")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(customer: CustomerFeatures):
    df = pd.DataFrame([customer.model_dump()])
    prob = model.predict_proba(df)[0][1]
    outcome = "churn" if prob >= 0.5 else "no_churn"
    churn_counter.labels(prediction=outcome).inc()
    return {
        "churn_probability": round(float(prob), 4),
        "prediction": outcome
    }