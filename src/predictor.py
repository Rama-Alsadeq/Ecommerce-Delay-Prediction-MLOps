from pathlib import Path

import joblib

from .logging_config import get_logger

logger = get_logger(__name__)


def load_model(model_path):
    model_path = Path(model_path)

    logger.info("Loading model from: %s", model_path)

    if not model_path.exists():
        logger.error("Model file not found: %s", model_path)
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)

    logger.info("Model loaded successfully: type=%s", type(model).__name__)

    return model


def predict(model, X):
    logger.info("Starting single prediction: input_shape=%s", X.shape)

    prediction = model.predict(X)[0]

    probability = None

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(X)[0, 1]

    result = {
        "prediction": int(prediction),
        "probability": float(probability) if probability is not None else None,
    }

    logger.info(
        "Single prediction completed: prediction=%s, probability=%s",
        result["prediction"],
        result["probability"],
    )

    return result


def predict_batch(model, X):
    logger.info("Starting batch prediction: input_shape=%s", X.shape)

    predictions = model.predict(X)

    probabilities = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]

    results = []

    for i, prediction in enumerate(predictions):
        results.append(
            {
                "prediction": int(prediction),
                "probability": (
                    float(probabilities[i]) if probabilities is not None else None
                ),
            }
        )

    logger.info("Batch prediction completed: count=%d", len(results))

    return results
