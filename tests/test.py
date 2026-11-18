

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import importlib
import sys

# Create a dummy model for testing
dummy_model = MagicMock()

dummy_model.feature_names_in_ = [
    "tenure",
    "MonthlyCharges"
]

dummy_model.classes_ = [0, 1]

dummy_model.predict.return_value = [1]

dummy_model.predict_proba.return_value = [
    [0.20, 0.80]
]

# Mock MLflow model loading before importing FastAPI
with patch(
    "mlflow.sklearn.load_model",
    return_value=dummy_model
):
    sys.modules.pop("fastapi_app", None)
    import fastapi_app

client = TestClient(fastapi_app.app)


def test_home_endpoint():

    response = client.get("/")

    assert response.status_code == 200


def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json()["status"] == "healthy"


def test_prediction_endpoint():

    customer = {
        "tenure": 24,
        "MonthlyCharges": 70.5
    }

    response = client.post(
        "/predict",
        json=customer
    )

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] == "1"

    assert result["churn_prob"] == 0.8


def test_missing_features():

    customer = {
        "tenure": 24
    }

    response = client.post(
        "/predict",
        json=customer
    )

    assert response.status_code == 422
