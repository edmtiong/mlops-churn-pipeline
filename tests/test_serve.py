from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np

mock_model = MagicMock()
mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])

with patch("src.serve.mlflow.set_tracking_uri"), \
     patch("src.serve.mlflow.sklearn.load_model", return_value=mock_model):
    from src.serve import app
    import asyncio
    asyncio.run(app.router.startup())

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_returns_probability():
    payload = {
        "tenure": 1,
        "MonthlyCharges": 80.0,
        "TotalCharges": 80.0,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "PaymentMethod": "Electronic check",
        "PaperlessBilling": "Yes",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "churn_probability" in response.json()
    assert "prediction" in response.json()

def test_predict_high_risk_label():
    payload = {
        "tenure": 1, "MonthlyCharges": 80.0, "TotalCharges": 80.0,
        "Contract": "Month-to-month", "InternetService": "Fiber optic",
        "PaymentMethod": "Electronic check", "PaperlessBilling": "Yes",
        "gender": "Male", "SeniorCitizen": 0, "Partner": "No",
        "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "No",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No"
    }
    response = client.post("/predict", json=payload)
    assert response.json()["prediction"] == "churn"