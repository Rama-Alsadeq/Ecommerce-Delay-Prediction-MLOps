from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.features import prepare_features
from src.preprocessing import load_scaler, preprocess_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "tests/fixtures/feature_engineering"
TEST_FEATURES_PATH = ARTIFACT_DIR / "test_features_raw.csv"


def test_load_scaler():
    scaler = load_scaler(ARTIFACT_DIR)

    assert isinstance(scaler, MinMaxScaler)
    assert scaler.n_features_in_ == 30


def test_preprocess_features_shape():
    df = pd.read_csv(TEST_FEATURES_PATH)
    prepared = prepare_features(df, ARTIFACT_DIR)

    result = preprocess_features(prepared, ARTIFACT_DIR)

    assert result.shape == prepared.shape
    assert result.shape[1] == 30
    assert result.columns.tolist() == prepared.columns.tolist()


def test_preprocess_features_uses_fitted_scaler():
    df = pd.read_csv(TEST_FEATURES_PATH)
    prepared = prepare_features(df, ARTIFACT_DIR)

    scaler = load_scaler(ARTIFACT_DIR)
    original_min = scaler.data_min_.copy()
    original_max = scaler.data_max_.copy()

    preprocess_features(prepared, ARTIFACT_DIR)

    assert (scaler.data_min_ == original_min).all()
    assert (scaler.data_max_ == original_max).all()
