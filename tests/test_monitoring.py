import json

from src.monitoring import MonitoringMetrics, calculate_psi, get_drift_status


def test_record_request_and_metrics(tmp_path):
    monitor = MonitoringMetrics(tmp_path / "predictions.jsonl")
    monitor.record_request(100)
    monitor.record_request(200, error=True)
    metrics = monitor.get_metrics({"0": 0.5, "1": 0.5})
    assert metrics["request_count"] == 2
    assert metrics["error_count"] == 1
    assert metrics["error_rate"] == 0.5
    assert metrics["avg_latency_ms"] == 150.0


def test_record_prediction_creates_jsonl(tmp_path):
    log_path = tmp_path / "predictions.jsonl"
    monitor = MonitoringMetrics(log_path)
    monitor.record_prediction(1, 0.72, "1", 25.5)
    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["prediction"] == 1
    assert record["probability"] == 0.72
    assert record["model_version"] == "1"
    assert record["latency_ms"] == 25.5


def test_prediction_distribution():
    monitor = MonitoringMetrics("predictions.jsonl")
    monitor.record_prediction(0, 0.2, "1", 10)
    monitor.record_prediction(1, 0.8, "1", 10)
    monitor.record_prediction(1, 0.9, "1", 10)
    metrics = monitor.get_metrics()
    assert metrics["prediction_distribution"] == {"0": 0.3333, "1": 0.6667}


def test_psi_no_drift():
    psi = calculate_psi({"0": 0.5, "1": 0.5}, {"0": 0.5, "1": 0.5})
    assert psi == 0.0
    assert get_drift_status(psi, {"psi_warning": 0.10, "psi_alert": 0.20}) == "normal"


def test_psi_alert():
    psi = calculate_psi({"0": 0.5, "1": 0.5}, {"0": 0.01, "1": 0.99})
    assert psi > 0.20
    assert get_drift_status(psi, {"psi_warning": 0.10, "psi_alert": 0.20}) == "alert"
