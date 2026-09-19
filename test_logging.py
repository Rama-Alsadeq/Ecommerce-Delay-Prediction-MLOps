from src.logging_config import get_logger

logger = get_logger(__name__)

logger.info("Logging system test - INFO")
logger.warning("Logging system test - WARNING")
logger.error("Logging system test - ERROR")

print("\nLogging test completed.")
