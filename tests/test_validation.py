from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import FINAL_FEATURES
from src.validation import validate_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "tests/fixtures/feature_engineering"
TEST_FEATURES_PATH = ARTIFACT_DIR / "test_features_raw.csv"


@pytest.fixture
def valid_dataframe():
    df = pd.read_csv(TEST_FEATURES_PATH)
    return df.iloc[[0]].copy()


def test_valid_dataframe_passes(valid_dataframe):
    result = validate_features(valid_dataframe)
    assert result.shape == (1, 30)
    assert result.columns.tolist() == FINAL_FEATURES


def test_missing_feature_rejected(valid_dataframe):
    valid_dataframe = valid_dataframe.drop(columns=["total_price"])
    with pytest.raises(ValueError, match="Missing required features"):
        validate_features(valid_dataframe)


def test_unexpected_feature_rejected(valid_dataframe):
    valid_dataframe["unexpected_feature"] = 1
    with pytest.raises(ValueError, match="Unexpected features"):
        validate_features(valid_dataframe)


def test_empty_dataframe_rejected():
    empty_df = pd.DataFrame(columns=FINAL_FEATURES)
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_features(empty_df)


def test_non_numeric_value_rejected(valid_dataframe):
    valid_dataframe["total_price"] = valid_dataframe["total_price"].astype(object)
    valid_dataframe.loc[0, "total_price"] = "invalid"
    with pytest.raises(ValueError, match="Non-numeric values"):
        validate_features(valid_dataframe)


def test_negative_value_rejected(valid_dataframe):
    valid_dataframe.loc[0, "total_price"] = -1
    with pytest.raises(ValueError, match="Negative values"):
        validate_features(valid_dataframe)


def test_infinite_value_rejected(valid_dataframe):
    valid_dataframe.loc[0, "total_price"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        validate_features(valid_dataframe)


def test_missing_value_is_allowed_for_preprocessing(valid_dataframe):
    valid_dataframe.loc[0, "total_payment"] = np.nan
    result = validate_features(valid_dataframe)
    assert result.loc[0, "total_payment"] != result.loc[0, "total_payment"]
