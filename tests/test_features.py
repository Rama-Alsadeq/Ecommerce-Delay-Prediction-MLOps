from pathlib import Path

import pandas as pd

from src.features import (
    apply_feature_rules,
    load_feature_artifacts,
    prepare_features,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "tests/fixtures/feature_engineering"
TEST_FEATURES_PATH = ARTIFACT_DIR / "test_features_raw.csv"


def test_load_feature_artifacts():
    feature_names, imputation_medians, replacement_values = load_feature_artifacts(
        ARTIFACT_DIR
    )

    assert len(feature_names) == 30
    assert len(imputation_medians) == 13
    assert set(replacement_values) == {
        "item_count",
        "unique_category_count",
        "unique_seller_states",
    }


def test_apply_feature_rules_replacements_and_clipping():
    _, imputation_medians, replacement_values = load_feature_artifacts(ARTIFACT_DIR)

    df = pd.DataFrame(
        {
            "item_count": [21, 10],
            "unique_category_count": [6, 2],
            "unique_seller_states": [3, 1],
            "seller_count": [5, 2],
            "unique_products": [7, 3],
        }
    )

    result = apply_feature_rules(
        df,
        imputation_medians,
        replacement_values,
    )

    assert result.loc[0, "item_count"] == 6
    assert result.loc[1, "item_count"] == 6
    assert result.loc[0, "unique_category_count"] == 5
    assert result.loc[0, "unique_seller_states"] == 2
    assert result.loc[0, "seller_count"] == 3
    assert result.loc[0, "unique_products"] == 4


def test_apply_feature_rules_imputation():
    _, imputation_medians, replacement_values = load_feature_artifacts(ARTIFACT_DIR)

    column = next(iter(imputation_medians))
    median = imputation_medians[column]

    df = pd.DataFrame({column: [None]})

    result = apply_feature_rules(
        df,
        imputation_medians,
        replacement_values,
    )

    assert result.loc[0, column] == median
    assert result[column].isna().sum() == 0


def test_prepare_features_shape_and_order():
    df = pd.read_csv(TEST_FEATURES_PATH)

    result = prepare_features(df, ARTIFACT_DIR)

    feature_names, _, _ = load_feature_artifacts(ARTIFACT_DIR)

    assert result.shape[1] == 30
    assert result.columns.tolist() == feature_names
    assert result.shape[0] == df.shape[0]
    assert result.isna().sum().sum() == 0
