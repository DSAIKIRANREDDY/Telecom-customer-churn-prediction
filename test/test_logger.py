
from telecom_churn.utils.logger import get_logger

logger = get_logger("telecom_churn_test")

logger.debug("This is a debug message")
logger.info("Logger initialized successfully")
logger.warning("This is a warning message")
logger.error("This is an error message")

try:
    result = 10 / 0

except Exception:
    logger.exception("An exception occurred during testing")
