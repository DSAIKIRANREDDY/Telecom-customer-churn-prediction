
import logging

from pathlib import Path

from telecom_churn.constants import PROJECT_ROOT, CONFIG_FILE_PATH
from telecom_churn.utils.common import read_yaml


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a configured logger.
    """

    config = read_yaml(CONFIG_FILE_PATH)
    logging_config = config.logging

    log_file = PROJECT_ROOT / logging_config.log_file

    # Create log directory
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(
        getattr(logging, logging_config.level.upper())
    )

    logger.propagate = False

    formatter = logging.Formatter(
        fmt=logging_config.log_format,
        datefmt=logging_config.date_format
    )

    # File handler
    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8"
    )

    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()

    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
