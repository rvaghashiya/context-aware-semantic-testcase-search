"""
Logging utilities for the semantic clustering project.
"""
import logging
import os
import sys
from datetime import datetime
from typing import Optional

def get_project_logger(name: str, 
                       level: int = logging.INFO,
                       log_file: Optional[str] = None,
                       format_string: Optional[str] = None) -> logging.Logger:
    """Get a configured logger for the project.
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level
        log_file: Optional log file path
        format_string: Custom format string
        
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def setup_logger(name: str, 
                 log_file: Optional[str] = None, 
                 level: int = logging.INFO) -> logging.Logger:
    """Legacy function for backward compatibility."""
    return get_project_logger(name, level, log_file)

class ProgressLogger:
    """Progress logger for long-running operations."""
    
    def __init__(self, logger: logging.Logger, total_items: int, log_interval: int = 100):
        """Initialize progress logger.
        
        Args:
            logger: Logger instance
            total_items: Total number of items to process
            log_interval: Log progress every N items
        """
        self.logger = logger
        self.total_items = total_items
        self.log_interval = log_interval
        self.processed = 0
        self.start_time = datetime.now()
    
    def update(self, count: int = 1) -> None:
        """Update progress counter.
        
        Args:
            count: Number of items processed
        """
        self.processed += count
        
        if self.processed % self.log_interval == 0 or self.processed == self.total_items:
            elapsed = datetime.now() - self.start_time
            progress_pct = (self.processed / self.total_items) * 100
            
            self.logger.info(
                f"Progress: {self.processed}/{self.total_items} "
                f"({progress_pct:.1f}%) - Elapsed: {elapsed}"
            )
    
    def finish(self) -> None:
        """Log completion."""
        elapsed = datetime.now() - self.start_time
        self.logger.info(f"Completed {self.processed} items in {elapsed}")

def log_function_call(func):
    """Decorator to log function calls."""
    def wrapper(*args, **kwargs):
        logger = get_project_logger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}")
            raise
    
    return wrapper

def create_log_directory(log_dir: str = "logs") -> str:
    """Create log directory if it doesn't exist.
    
    Args:
        log_dir: Directory name for logs
        
    Returns:
        Path to log directory
    """
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def get_timestamp() -> str:
    """Get current timestamp for log files."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")