from pathlib import Path

import pandas as pd

from src.data import order_to_dataframe
from src.features import prepare_features
from src.predictor import load_model, predict
from src.preprocessing import preprocess_features
from src.validation import validate_features

ARTIFACT_DIR = Path("artifacts/feature_engineering")
MODEL_PATH = Path("artifacts/models/classification/logistic_regression_smote.joblib")
TEST_FEATURES_PATH = ARTIFACT_DIR / "test_features_raw.csv"


print("=" * 70)
print("INFERENCE PIPELINE LOGGING TEST")
print("=" * 70)

test_features = pd.read_csv(TEST_FEATURES_PATH)

order = test_features.iloc[0].to_dict()

print("\n1. DATA")
df = order_to_dataframe(order)
print("Input shape:", df.shape)

print("\n2. VALIDATION")
df = validate_features(df)
print("Validated shape:", df.shape)

print("\n3. FEATURE PREPARATION")
df = prepare_features(df, ARTIFACT_DIR)
print("Prepared shape:", df.shape)

print("\n4. PREPROCESSING")
X = preprocess_features(df, ARTIFACT_DIR)
print("Scaled shape:", X.shape)

print("\n5. MODEL")
model = load_model(MODEL_PATH)

print("\n6. PREDICTION")
result = predict(model, X)

print("Prediction:", result["prediction"])
print("Probability:", result["probability"])

print("\n" + "=" * 70)
print("PIPELINE TEST COMPLETED")
print("=" * 70)
