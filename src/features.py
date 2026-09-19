import json
from pathlib import Path

import joblib
import pandas as pd

from .logging_config import get_logger

logger = get_logger(__name__)


def load_feature_artifacts(artifact_dir):
    artifact_dir = Path(artifact_dir)

    logger.info("Loading feature artifacts from: %s", artifact_dir)

    with open(artifact_dir / "feature_names.json", "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    imputation_medians = joblib.load(artifact_dir / "imputation_medians.joblib")

    replacement_values = joblib.load(artifact_dir / "replacement_values.joblib")

    logger.info(
        "Feature artifacts loaded successfully: features=%d, medians=%d, replacements=%d",
        len(feature_names),
        len(imputation_medians),
        len(replacement_values),
    )

    return feature_names, imputation_medians, replacement_values


def apply_feature_rules(df, imputation_medians, replacement_values):
    df = df.copy()

    replacement_count = 0
    imputation_count = 0

    # Replace values that were removed during training
    for column, replacement in replacement_values.items():
        if column not in df.columns:
            continue

        if column == "item_count":
            mask = df[column] == 21
            replacement_count += int(mask.sum())
            df[column] = df[column].replace(21, replacement)

        elif column == "unique_category_count":
            mask = df[column] == 6
            replacement_count += int(mask.sum())
            df[column] = df[column].replace(6, replacement)

        elif column == "unique_seller_states":
            mask = df[column] == 3
            replacement_count += int(mask.sum())
            df[column] = df[column].replace(3, replacement)

    # Apply upper limits used during training
    if "seller_count" in df.columns:
        df["seller_count"] = df["seller_count"].clip(upper=3)

    if "item_count" in df.columns:
        df["item_count"] = df["item_count"].clip(upper=6)

    if "unique_products" in df.columns:
        df["unique_products"] = df["unique_products"].clip(upper=4)

    # Fill missing values using training medians
    for column, median in imputation_medians.items():
        if column in df.columns:
            missing_count = int(df[column].isna().sum())

            if missing_count > 0:
                imputation_count += missing_count
                df[column] = df[column].fillna(median)

    if replacement_count > 0:
        logger.warning(
            "Feature values replaced using training rules: count=%d", replacement_count
        )

    if imputation_count > 0:
        logger.warning(
            "Missing values imputed using training medians: count=%d", imputation_count
        )

    if replacement_count == 0 and imputation_count == 0:
        logger.info("No value replacements or imputations were required")

    return df


def prepare_features(df, artifact_dir):
    logger.info("Preparing features: input_shape=%s", df.shape)

    feature_names, imputation_medians, replacement_values = load_feature_artifacts(
        artifact_dir
    )

    missing_features = [
        feature for feature in feature_names if feature not in df.columns
    ]

    if missing_features:
        logger.error(
            "Missing required features during preparation: %s", missing_features
        )
        raise ValueError(f"Missing required features: {missing_features}")

    df = apply_feature_rules(df, imputation_medians, replacement_values)

    df = df[feature_names].copy()
    df = df.apply(pd.to_numeric, errors="coerce")

    logger.info("Features prepared successfully: output_shape=%s", df.shape)

    return df
