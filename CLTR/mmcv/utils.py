"""Minimal mmcv.utils shim."""
import logging
import os


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    """Get a named logger, optionally with a file handler."""
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger
    logger.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, mode=file_mode)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger
