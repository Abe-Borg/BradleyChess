
import logging
import multiprocessing

def setup_logger(name: str, log_file: str, level=logging.ERROR) -> logging.Logger:
    logger = logging.getLogger(f"{name}_{multiprocessing.current_process().pid}")
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        process_id = multiprocessing.current_process().pid
        log_file = f"{log_file}_{process_id}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
