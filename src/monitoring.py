import json
import math
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .logging_config import get_logger

logger = get_logger(__name__)


class MonitoringMetrics:
    def __init__(self, log_path):
        self.prediction_log = Path(log_path)
        self.prediction_log.parent.mkdir(parents=True, exist_ok=True)
        self.request_count = 0
        self.error_count = 0
        self.total_latency_ms = 0.0
        self.prediction_counts = Counter()
        self.lock = threading.Lock()

    def record_request(self, latency_ms, error=False):
        with self.lock:
            self.request_count += 1
            self.total_latency_ms += float(latency_ms)
            if error:
                self.error_count += 1

    def record_prediction(self, prediction, probability, model_version, latency_ms):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prediction": int(prediction),
            "probability": float(probability) if probability is not None else None,
            "model_version": str(model_version),
            "latency_ms": round(float(latency_ms), 2),
        }
        with self.lock:
            self.prediction_counts[str(prediction)] += 1
            with self.prediction_log.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record) + "\n")

    def get_metrics(self, reference_distribution=None, thresholds=None):
        thresholds = thresholds or {}
        with self.lock:
            request_count = self.request_count
            error_count = self.error_count
            total_latency_ms = self.total_latency_ms
            distribution = self._prediction_distribution()
            prediction_count = sum(self.prediction_counts.values())

        error_rate = error_count / request_count if request_count else 0.0
        avg_latency_ms = total_latency_ms / request_count if request_count else 0.0
        min_samples = int(thresholds.get("min_samples_for_drift", 30))
        psi = None
        if prediction_count >= min_samples:
            psi = calculate_psi(reference_distribution, distribution)

        return {
            "request_count": request_count,
            "error_count": error_count,
            "error_rate": round(error_rate, 4),
            "avg_latency_ms": round(avg_latency_ms, 2),
            "prediction_distribution": distribution,
            "prediction_drift": {
                "psi": round(psi, 4) if psi is not None else None,
                "status": get_drift_status(psi, thresholds),
            },
            "alerts": {
                "error_rate": error_rate > float(thresholds.get("error_rate", 0.05)),
                "latency": avg_latency_ms > float(thresholds.get("latency_ms", 500)),
                "prediction_drift": psi is not None
                and psi > float(thresholds.get("psi_alert", 0.20)),
            },
        }

    def _prediction_distribution(self):
        total = sum(self.prediction_counts.values())
        if total == 0:
            return {"0": 0.0, "1": 0.0}
        return {
            "0": round(self.prediction_counts.get("0", 0) / total, 4),
            "1": round(self.prediction_counts.get("1", 0) / total, 4),
        }


def calculate_psi(reference, current):
    if not reference or not current:
        return None
    if sum(float(value) for value in current.values()) == 0:
        return None
    psi = 0.0
    for key in ("0", "1"):
        expected = max(float(reference.get(key, 0.0)), 1e-6)
        actual = max(float(current.get(key, 0.0)), 1e-6)
        psi += (actual - expected) * math.log(actual / expected)
    return psi


def get_drift_status(psi, thresholds):
    if psi is None:
        return "no_data"
    if psi > float(thresholds.get("psi_alert", 0.20)):
        return "alert"
    if psi >= float(thresholds.get("psi_warning", 0.10)):
        return "warning"
    return "normal"


def load_reference_distribution(path):
    path = Path(path)
    if not path.exists():
        logger.warning("Reference prediction distribution not found: %s", path)
        return None
    with path.open("r", encoding="utf-8") as file:
        distribution = json.load(file)
    logger.info("Reference prediction distribution loaded from: %s", path)
    return distribution
