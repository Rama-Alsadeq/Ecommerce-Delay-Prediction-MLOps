import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


VALID_ORDER = {
    "customer_lat": -19.46970149792253,
    "customer_lng": -42.562507159587966,
    "item_count": 1.0,
    "unique_products": 1.0,
    "seller_count": 1.0,
    "total_price": 46.0,
    "min_freight": 18.42,
    "unique_seller_states": 1.0,
    "sellers_same_customer_state": 0.0,
    "unique_category_count": 1.0,
    "min_weight_g": 200.0,
    "min_volume_cm3": 4096.0,
    "min_seller_distance_km": 635.2426647769917,
    "total_payment": 64.42,
    "max_installments": 1.0,
    "purchase_year": 2018.0,
    "purchase_month": 6.0,
    "purchase_weekday": 3.0,
    "purchase_hour": 8.0,
    "purchase_quarter": 2.0,
    "approval_year": 2018.0,
    "approval_quarter": 2.0,
    "approval_month": 6.0,
    "approval_weekday": 3.0,
    "approval_day": 21.0,
    "approval_hour": 8.0,
    "customer_zip_code_prefix_mean_delay": -12.309242317498658,
    "customer_city_mean_delay": -12.685134089882563,
    "customer_state_mean_delay": -12.269081183109853,
    "seller_city_mean_delay": -10.214625499395048,
}


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_name"] == "Brazilian-E-Commerce-Classifier"
    assert data["model_version"] == "1"


def test_model_info(client):
    response = client.get("/model")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == "Brazilian-E-Commerce-Classifier"
    assert data["model_version"] == "1"
    assert data["model_uri"] == "models:/Brazilian-E-Commerce-Classifier/1"
    assert data["model_type"] == "LogisticRegression"
    assert data["feature_count"] == 30


def test_predict(client):
    response = client.post("/predict", json=VALID_ORDER)

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["probability"] <= 1.0
    assert data["model_version"] == "1"


def test_predict_batch(client):
    response = client.post(
        "/predict/batch",
        json={
            "orders": [
                VALID_ORDER,
                VALID_ORDER,
            ]
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data["predictions"]) == 2
    assert data["model_version"] == "1"

    assert data["predictions"][0] == data["predictions"][1]


def test_missing_feature_rejected(client):
    order = VALID_ORDER.copy()
    order.pop("total_price")

    response = client.post("/predict", json=order)

    assert response.status_code == 422


def test_extra_feature_rejected(client):
    order = VALID_ORDER.copy()
    order["test_feature"] = 123

    response = client.post("/predict", json=order)

    assert response.status_code == 422


def test_negative_value_rejected(client):
    order = VALID_ORDER.copy()
    order["total_price"] = -46.0

    response = client.post("/predict", json=order)

    assert response.status_code == 422
