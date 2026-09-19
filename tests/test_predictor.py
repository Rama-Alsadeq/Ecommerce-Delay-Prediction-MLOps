from pathlib import Path

import pandas as pd

from src.features import prepare_features
from src.predictor import load_model, predict, predict_batch
from src.preprocessing import preprocess_features

PROJECT_ROOT = Path("/app")
ARTIFACT_DIR = PROJECT_ROOT / "artifacts/feature_engineering"
MODEL_PATH = (
    PROJECT_ROOT / "artifacts/models/classification/logistic_regression_smote.joblib"
)
TEST_FEATURES_PATH = ARTIFACT_DIR / "test_features_raw.csv"


def get_test_data():
    df = pd.read_csv(TEST_FEATURES_PATH)
    df = prepare_features(df, ARTIFACT_DIR)
    X = preprocess_features(df, ARTIFACT_DIR)
    return X


def test_load_model():
    model = load_model(MODEL_PATH)

    assert type(model).__name__ == "LogisticRegression"
    assert model.n_features_in_ == 30


def test_single_prediction():
    X = get_test_data().iloc[[0]]
    model = load_model(MODEL_PATH)

    result = predict(model, X)

    assert isinstance(result, dict)
    assert "prediction" in result
    assert "probability" in result
    assert result["prediction"] in [0, 1]
    assert 0.0 <= result["probability"] <= 1.0


def test_batch_prediction():
    X = get_test_data().iloc[:5]
    model = load_model(MODEL_PATH)

    results = predict_batch(model, X)

    assert isinstance(results, list)
    assert len(results) == 5

    for result in results:
        assert "prediction" in result
        assert "probability" in result
        assert result["prediction"] in [0, 1]
        assert 0.0 <= result["probability"] <= 1.0


def test_single_and_batch_prediction_are_consistent():
    X = get_test_data().iloc[:1]
    model = load_model(MODEL_PATH)

    single_result = predict(model, X)
    batch_results = predict_batch(model, X)

    assert single_result == batch_results[0]
