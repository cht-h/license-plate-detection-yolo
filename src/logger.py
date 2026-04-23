import logging
import sys
from pathlib import Path

_LOG_FILE = Path("./data/log_file.log")

try:
    from loguru import logger as _loguru_logger
except ImportError:
    _loguru_logger = None


def _setup_std_logging() -> logging.Logger:
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("license_plate_detection")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
        )
    )

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%H:%M:%S")
    )

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def setup_logging():
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    if _loguru_logger is None:
        return _setup_std_logging()

    _loguru_logger.remove()
    _loguru_logger.add(
        _LOG_FILE,
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        encoding="utf-8",
    )
    _loguru_logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    )
    return _loguru_logger


log = setup_logging()