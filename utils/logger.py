# utils/logger.py

import logging
import os
from datetime import datetime

def get_logger(name: str = "intellistream", log_dir: str = "logs") -> logging.Logger:
    """
    Returns a configured logger instance.
    Logs are written to both file and console.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers
    if not logger.handlers:

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
