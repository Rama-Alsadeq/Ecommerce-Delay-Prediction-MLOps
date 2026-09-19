import pandas as pd

from .logging_config import get_logger

logger = get_logger(__name__)

FINAL_FEATURES = [
    "customer_lat",
    "customer_lng",
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
    "purchase_year",
    "purchase_month",
    "purchase_weekday",
    "purchase_hour",
    "purchase_quarter",
    "approval_year",
    "approval_quarter",
    "approval_month",
    "approval_weekday",
    "approval_day",
    "approval_hour",
    "customer_zip_code_prefix_mean_delay",
    "customer_city_mean_delay",
    "customer_state_mean_delay",
    "seller_city_mean_delay",
]


def order_to_dataframe(order: dict) -> pd.DataFrame:
    """Convert one API order into a DataFrame with the final model features."""
    logger.info("Converting single order to DataFrame")

    missing = [feature for feature in FINAL_FEATURES if feature not in order]

    if missing:
        logger.warning("Single order is missing required features: %s", missing)
        raise ValueError(f"Missing required features: {missing}")

    df = pd.DataFrame(
        [[order[feature] for feature in FINAL_FEATURES]], columns=FINAL_FEATURES
    )

    logger.info("Single order converted successfully: shape=%s", df.shape)

    return df


def orders_to_dataframe(orders: list[dict]) -> pd.DataFrame:
    """Convert multiple API orders into a DataFrame with the final model features."""
    logger.info("Converting batch of orders to DataFrame: count=%d", len(orders))

    if not orders:
        logger.warning("Received an empty orders list")
        raise ValueError("The orders list cannot be empty.")

    missing = [
        feature
        for feature in FINAL_FEATURES
        if any(feature not in order for order in orders)
    ]

    if missing:
        logger.warning(
            "Batch contains orders with missing required features: %s", missing
        )
        raise ValueError(f"Missing required features: {missing}")

    df = pd.DataFrame(orders, columns=FINAL_FEATURES)

    logger.info("Batch converted successfully: shape=%s", df.shape)

    return df
