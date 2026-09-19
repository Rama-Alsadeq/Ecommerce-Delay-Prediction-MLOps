from pathlib import Path

import joblib
import pandas as pd

from .logging_config import get_logger

logger = get_logger(__name__)


def load_scaler(artifact_dir):
    artifact_dir = Path(artifact_dir)

    scaler_path = artifact_dir / "scaler.joblib"

    logger.info("Loading scaler from: %s", scaler_path)

    if not scaler_path.exists():
        logger.error("Scaler artifact not found: %s", scaler_path)
        raise FileNotFoundError(f"Scaler artifact not found: {scaler_path}")

    scaler = joblib.load(scaler_path)

    logger.info("Scaler loaded successfully: type=%s", type(scaler).__name__)

    return scaler


def preprocess_features(df, artifact_dir):
    logger.info("Starting preprocessing: input_shape=%s", df.shape)

    scaler = load_scaler(artifact_dir)

    X = df.copy()

    X_scaled = scaler.transform(X)

    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    logger.info("Preprocessing completed successfully: output_shape=%s", X_scaled.shape)

    return X_scaled
