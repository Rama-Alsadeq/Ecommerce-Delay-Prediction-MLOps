import great_expectations as gx
from src.data import FINAL_FEATURES

NON_NEGATIVE_FEATURES = [
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

INTEGER_FEATURES = [
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
]

RANGE_FEATURES = {
    "purchase_month": (1, 12),
    "purchase_weekday": (0, 6),
    "purchase_hour": (0, 23),
    "purchase_quarter": (1, 4),
    "approval_quarter": (1, 4),
    "approval_month": (1, 12),
    "approval_weekday": (0, 6),
    "approval_day": (1, 31),
    "approval_hour": (0, 23),
}

context = gx.get_context(mode="file")

try:
    context.suites.delete("test_features_quality")
except Exception:
    pass

suite = gx.ExpectationSuite(name="test_features_quality")

for column in FINAL_FEATURES:
    suite.add_expectation(
        gx.expectations.ExpectColumnToExist(
            column=column
        )
    )

for column in FINAL_FEATURES:
    expected_type = "int64" if column in INTEGER_FEATURES else "float64"
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeOfType(
            column=column,
            type_=expected_type
        )
    )

for column in NON_NEGATIVE_FEATURES:
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column=column,
            min_value=0,
            strict_min=False
        )
    )

for column, (min_value, max_value) in RANGE_FEATURES.items():
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column=column,
            min_value=min_value,
            max_value=max_value,
            strict_min=False,
            strict_max=False
        )
    )

for column in FINAL_FEATURES:
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column=column
        )
    )

context.suites.add(suite)

print(f"Expectations added: {len(suite.expectations)}")