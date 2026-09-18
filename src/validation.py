import numpy as np
import pandas as pd

from .data import FINAL_FEATURES
from .logging_config import get_logger

logger = get_logger(__name__)


def validate_features(df):
    logger.info(
        "Starting feature validation: input_shape=%s",
        getattr(df, "shape", None)
    )

    if not isinstance(df, pd.DataFrame):
        logger.error(
            "Feature validation failed: input is not a pandas DataFrame"
        )
        raise TypeError("Input must be a pandas DataFrame.")

    missing_features = [
        feature
        for feature in FINAL_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        logger.error(
            "Feature validation failed: missing required features: %s",
            missing_features
        )
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    extra_features = [
        feature
        for feature in df.columns
        if feature not in FINAL_FEATURES
    ]

    if extra_features:
        logger.error(
            "Feature validation failed: unexpected features: %s",
            extra_features
        )
        raise ValueError(
            f"Unexpected features: {extra_features}"
        )

    if df.empty:
        logger.error(
            "Feature validation failed: input DataFrame is empty"
        )
        raise ValueError("Input DataFrame cannot be empty.")

    numeric_df = df[FINAL_FEATURES].apply(
        pd.to_numeric,
        errors="coerce"
    )

    non_numeric_columns = []

    for column in FINAL_FEATURES:
        original_non_null = df[column].notna()
        converted_non_null = numeric_df[column].notna()

        if (original_non_null & ~converted_non_null).any():
            non_numeric_columns.append(column)

    if non_numeric_columns:
        logger.error(
            "Feature validation failed: non-numeric values found in: %s",
            non_numeric_columns
        )
        raise ValueError(
            f"Non-numeric values found in: {non_numeric_columns}"
        )

    missing_value_columns = numeric_df.columns[
        numeric_df.isna().any()
    ].tolist()

    if missing_value_columns:
        logger.warning(
            "Missing values detected and will be handled by feature preprocessing: %s",
            missing_value_columns
        )

    finite_values = numeric_df.dropna().to_numpy()

    if not np.isfinite(finite_values).all():
        logger.error(
            "Feature validation failed: input contains infinite values"
        )
        raise ValueError("Input contains infinite values.")

    non_negative_features = [
        "item_count",
        "unique_products",
        "seller_count",
        "total_price",
        "min_freight",
        "unique_seller_states",
        "sellers_same_customer_state",
        "unique_category_count",
        "min_weight_g",
        "min_volume_cm3",
        "min_seller_distance_km",
        "total_payment",
        "max_installments",
    ]

    negative_features = [
        feature
        for feature in non_negative_features
        if (numeric_df[feature] < 0).any()
    ]

    if negative_features:
        logger.error(
            "Feature validation failed: negative values found in: %s",
            negative_features
        )
        raise ValueError(
            f"Negative values found in: {negative_features}"
        )

    logger.info(
        "Feature validation completed successfully: shape=%s",
        numeric_df.shape
    )

    return numeric_df[FINAL_FEATURES].copy()
