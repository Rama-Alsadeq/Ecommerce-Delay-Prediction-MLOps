import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parent
FEATURE_DIR = ROOT / "feature_engineering"

FEATURE_NAMES = [
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

FEATURE_DIR.mkdir(parents=True, exist_ok=True)

with open(FEATURE_DIR / "feature_names.json", "w", encoding="utf-8") as f:
    json.dump(FEATURE_NAMES, f, indent=2)

medians = {
    "total_price": 100.0,
    "min_freight": 20.0,
    "min_weight_g": 500.0,
    "min_volume_cm3": 5000.0,
    "min_seller_distance_km": 100.0,
    "total_payment": 120.0,
    "max_installments": 1.0,
    "purchase_year": 2018.0,
    "purchase_month": 6.0,
    "purchase_weekday": 3.0,
    "purchase_hour": 12.0,
    "approval_day": 15.0,
    "approval_hour": 12.0,
}
joblib.dump(medians, FEATURE_DIR / "imputation_medians.joblib")

replacement_values = {
    "item_count": 6,
    "unique_category_count": 5,
    "unique_seller_states": 2,
}
joblib.dump(replacement_values, FEATURE_DIR / "replacement_values.joblib")

rng = np.random.default_rng(42)
X_train = pd.DataFrame(
    rng.uniform(0, 1, size=(20, len(FEATURE_NAMES))),
    columns=FEATURE_NAMES,
)
y_train = np.array([0, 1] * 10)

scaler = MinMaxScaler()
scaler.fit(X_train)
joblib.dump(scaler, FEATURE_DIR / "scaler.joblib")

base_row = [
    -19.4697,
    -42.5625,
    1.0,
    1.0,
    1.0,
    46.0,
    18.42,
    1.0,
    0.0,
    1.0,
    200.0,
    4096.0,
    635.24,
    64.42,
    1.0,
    2018.0,
    6.0,
    3.0,
    8.0,
    2.0,
    2018.0,
    2.0,
    6.0,
    3.0,
    21.0,
    8.0,
    -12.3,
    -12.7,
    -12.2,
    -10.2,
]

test_data = pd.DataFrame(
    [base_row] * 5,
    columns=FEATURE_NAMES,
)
test_data.to_csv(FEATURE_DIR / "test_features_raw.csv", index=False)

X_model = scaler.transform(X_train)
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_model, y_train)
joblib.dump(model, ROOT / "logistic_regression.joblib")

print("Test fixtures created successfully.")
