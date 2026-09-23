import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.features import prepare_features
from src.logging_config import get_logger
from src.monitoring import MonitoringMetrics, load_reference_distribution
from src.predictor import predict, predict_batch
from src.preprocessing import preprocess_features
from src.validation import validate_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    CONFIG = yaml.safe_load(file)

API_CONFIG = CONFIG["api"]
MLFLOW_CONFIG = CONFIG["mlflow"]
MONITORING_CONFIG = CONFIG["monitoring"]

MLFLOW_TRACKING_URI = MLFLOW_CONFIG["tracking_uri"]
MODEL_NAME = MLFLOW_CONFIG["model_name"]
MODEL_VERSION = str(MLFLOW_CONFIG["model_version"])
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

ARTIFACT_DIR = PROJECT_ROOT / CONFIG["paths"]["feature_engineering"]
MONITORING_LOG_PATH = PROJECT_ROOT / MONITORING_CONFIG["prediction_log"]
REFERENCE_DISTRIBUTION_PATH = PROJECT_ROOT / MONITORING_CONFIG["reference_distribution"]
MONITORING_THRESHOLDS = MONITORING_CONFIG["thresholds"]

monitoring = MonitoringMetrics(MONITORING_LOG_PATH)
reference_distribution = load_reference_distribution(REFERENCE_DISTRIBUTION_PATH)

logger = get_logger(__name__)

app = FastAPI(
    title="Brazilian E-Commerce Prediction API",
    description="API for delayed-order classification.",
    version="1.0.0",
)


class OrderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_lat: float
    customer_lng: float
    item_count: float
    unique_products: float
    seller_count: float
    total_price: float
    min_freight: float
    unique_seller_states: float
    sellers_same_customer_state: float
    unique_category_count: float
    min_weight_g: float
    min_volume_cm3: float
    min_seller_distance_km: float
    total_payment: float
    max_installments: float
    purchase_year: float
    purchase_month: float
    purchase_weekday: float
    purchase_hour: float
    purchase_quarter: float
    approval_year: float
    approval_quarter: float
    approval_month: float
    approval_weekday: float
    approval_day: float
    approval_hour: float
    customer_zip_code_prefix_mean_delay: float
    customer_city_mean_delay: float
    customer_state_mean_delay: float
    seller_city_mean_delay: float


class BatchOrderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    orders: list[OrderRequest] = Field(min_length=1)


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    model_version: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_uri: str
    model_type: str
    feature_count: int


class MetricsResponse(BaseModel):
    request_count: int
    error_count: int
    error_rate: float
    avg_latency_ms: float
    prediction_distribution: dict[str, float]
    prediction_drift: dict[str, float | str | None]
    alerts: dict[str, bool]


model = None


def load_registered_model():
    global model

    logger.info("Loading model from MLflow Registry: %s", MODEL_URI)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(MODEL_URI)

    logger.info(
        "Model loaded successfully: name=%s version=%s type=%s features=%s",
        MODEL_NAME,
        MODEL_VERSION,
        type(model).__name__,
        getattr(model, "n_features_in_", "unknown"),
    )


@app.on_event("startup")
def startup_event():
    load_registered_model()


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
    )


@app.get("/model", response_model=ModelInfoResponse)
def model_info():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    return ModelInfoResponse(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        model_uri=MODEL_URI,
        model_type=type(model).__name__,
        feature_count=getattr(model, "n_features_in_", 0),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_single(order: OrderRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    start_time = time.perf_counter()

    try:
        df = validate_features(__import__("pandas").DataFrame([order.model_dump()]))
        df = prepare_features(df, ARTIFACT_DIR)
        X = preprocess_features(df, ARTIFACT_DIR)

        result = predict(model, X)

        latency_ms = (time.perf_counter() - start_time) * 1000

        monitoring.record_request(latency_ms)
        monitoring.record_prediction(
            result["prediction"],
            result["probability"],
            MODEL_VERSION,
            latency_ms,
        )

        logger.info(
            "Prediction request completed: prediction=%s probability=%s "
            "model_version=%s latency_ms=%.2f",
            result["prediction"],
            result["probability"],
            MODEL_VERSION,
            latency_ms,
        )

        return PredictionResponse(
            prediction=result["prediction"],
            probability=result["probability"],
            model_version=MODEL_VERSION,
        )

    except (ValueError, TypeError) as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000
        monitoring.record_request(latency_ms, error=True)

        logger.warning(
            "Prediction request rejected: error=%s latency_ms=%.2f",
            str(exc),
            latency_ms,
        )

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000
        monitoring.record_request(latency_ms, error=True)

        logger.exception(
            "Prediction request failed: latency_ms=%.2f",
            latency_ms,
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch_orders(request: BatchOrderRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    start_time = time.perf_counter()

    try:
        import pandas as pd

        df = pd.DataFrame([order.model_dump() for order in request.orders])

        df = validate_features(df)
        df = prepare_features(df, ARTIFACT_DIR)
        X = preprocess_features(df, ARTIFACT_DIR)

        results = predict_batch(model, X)

        latency_ms = (time.perf_counter() - start_time) * 1000

        monitoring.record_request(latency_ms)

        for result in results:
            monitoring.record_prediction(
                result["prediction"],
                result["probability"],
                MODEL_VERSION,
                latency_ms,
            )

        logger.info(
            "Batch prediction completed: count=%d model_version=%s " "latency_ms=%.2f",
            len(results),
            MODEL_VERSION,
            latency_ms,
        )

        return BatchPredictionResponse(
            predictions=[
                PredictionResponse(
                    prediction=result["prediction"],
                    probability=result["probability"],
                    model_version=MODEL_VERSION,
                )
                for result in results
            ],
            model_version=MODEL_VERSION,
        )

    except (ValueError, TypeError) as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000
        monitoring.record_request(latency_ms, error=True)

        logger.warning(
            "Batch prediction request rejected: error=%s latency_ms=%.2f",
            str(exc),
            latency_ms,
        )

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000
        monitoring.record_request(latency_ms, error=True)

        logger.exception(
            "Batch prediction failed: latency_ms=%.2f",
            latency_ms,
        )

        raise HTTPException(
            status_code=500,
            detail="Batch prediction failed.",
        ) from exc


@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    return monitoring.get_metrics(
        reference_distribution,
        MONITORING_THRESHOLDS,
    )
