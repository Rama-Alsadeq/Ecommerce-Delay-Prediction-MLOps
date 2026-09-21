from pathlib import Path

import joblib

from app import main

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
FEATURE_DIR = FIXTURE_DIR / "feature_engineering"
MODEL_PATH = FIXTURE_DIR / "logistic_regression.joblib"

main.ARTIFACT_DIR = FEATURE_DIR


def load_test_model():
    main.model = joblib.load(MODEL_PATH)


main.load_registered_model = load_test_model
