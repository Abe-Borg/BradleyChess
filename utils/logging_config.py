# utils/logging_config.py
import logging
from pathlib import Path
import game_settings

def setup_logger(name, log_file, level=logging.ERROR):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger
