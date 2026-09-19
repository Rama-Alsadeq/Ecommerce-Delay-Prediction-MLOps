from pathlib import Path

from src.features import load_feature_artifacts
from src.predictor import load_model
from src.preprocessing import load_scaler

ARTIFACT_DIR = Path("artifacts/feature_engineering")
MODEL_PATH = Path("artifacts/models/classification/logistic_regression_smote.joblib")


print("=" * 70)
print("1. LOADING FEATURE ARTIFACTS")
print("=" * 70)

feature_names, imputation_medians, replacement_values = load_feature_artifacts(
    ARTIFACT_DIR
)

print("Feature count:", len(feature_names))
print("Medians loaded:", len(imputation_medians))
print("Replacement rules:", replacement_values)


print("\n" + "=" * 70)
print("2. LOADING SCALER")
print("=" * 70)

scaler = load_scaler(ARTIFACT_DIR)

print("Scaler:", type(scaler).__name__)
print("Scaler feature count:", scaler.n_features_in_)


print("\n" + "=" * 70)
print("3. LOADING MODEL")
print("=" * 70)

model = load_model(MODEL_PATH)

print("Model:", type(model).__name__)
print("Model feature count:", model.n_features_in_)


print("\n" + "=" * 70)
print("4. CONSISTENCY CHECK")
print("=" * 70)

assert len(feature_names) == 30
assert scaler.n_features_in_ == 30
assert model.n_features_in_ == 30

print("All components expect 30 features.")
print("Initial artifact check PASSED.")
