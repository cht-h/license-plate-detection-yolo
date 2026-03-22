"""
Logging configuration module.
Configures logging to both console and file as required by the assignment.
"""
import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logger(
    name: str = "license_plate_detector",
    log_file: Optional[Path] = None,
    level: int = logging.DEBUG,
    console_level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logger with both file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (default: data/logs/log_file.log)
        level: Logging level for file handler
        console_level: Logging level for console handler
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        # Default log file location
        log_file = Path(__file__).parent.parent.parent / "data" / "logs" / "log_file.log"
    else:
        log_file = Path(log_file)
    
    # Create directory if it doesn't exist
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logger initialized. Log file: {log_file}")
    
    return logger


# Create a default logger instance
default_logger = setup_logger()