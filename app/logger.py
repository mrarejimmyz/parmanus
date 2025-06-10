import os
import sys
from datetime import datetime

from loguru import logger as _logger

from app.config import PROJECT_ROOT

_print_level = "INFO"
_logfile_level = "DEBUG"
_max_log_size = "10 MB"
_log_retention = "7 days"


def define_log_level(
    print_level="INFO",
    logfile_level="DEBUG",
    name: str = None,
    max_size: str = "10 MB",
    retention: str = "7 days",
    enable_file_logging: bool = True,
):
    """
    Configure logging with optimized settings for production use.

    Args:
        print_level: Console log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        logfile_level: File log level
        name: Optional prefix for log file names
        max_size: Maximum size per log file before rotation
        retention: How long to keep old log files
        enable_file_logging: Whether to enable file logging
    """
    global _print_level, _logfile_level, _max_log_size, _log_retention
    _print_level = print_level
    _logfile_level = logfile_level
    _max_log_size = max_size
    _log_retention = retention

    # Remove all existing handlers
    _logger.remove()

    # Console handler with optimized format
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    _logger.add(
        sys.stderr,
        level=print_level,
        format=console_format,
        colorize=True,
        backtrace=False,  # Reduce verbosity
        diagnose=False,  # Reduce verbosity in production
    )

    if enable_file_logging:
        # Ensure logs directory exists
        logs_dir = PROJECT_ROOT / "logs"
        logs_dir.mkdir(exist_ok=True)

        # File handler with rotation and compression
        current_date = datetime.now()
        formatted_date = current_date.strftime("%Y%m%d")
        log_name = f"{name}_{formatted_date}" if name else formatted_date

        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} - "
            "{message}"
        )

        _logger.add(
            logs_dir / f"{log_name}.log",
            level=logfile_level,
            format=file_format,
            rotation=max_size,
            retention=retention,
            compression="gz",  # Compress old logs
            backtrace=True,  # Keep full traces in files
            diagnose=True,  # Keep diagnostics in files
            enqueue=True,  # Thread-safe logging
        )

    return _logger


def get_optimized_logger(name: str = None, level: str = "INFO"):
    """
    Get a logger instance optimized for the specific component.

    Args:
        name: Component name for the logger
        level: Log level for this specific logger

    Returns:
        Configured logger instance
    """
    # For production, reduce verbosity of frequent operations
    if name and any(keyword in name.lower() for keyword in ["llm", "model", "tool"]):
        # Reduce verbosity for LLM and tool operations
        return define_log_level(
            print_level="WARNING",  # Only show warnings and errors on console
            logfile_level="INFO",  # Log info and above to file
            name=name,
            enable_file_logging=True,
        )

    return define_log_level(
        print_level=level, logfile_level="DEBUG", name=name, enable_file_logging=True
    )


def create_component_logger(component_name: str, verbose: bool = False):
    """
    Create a logger for a specific component with appropriate verbosity.

    Args:
        component_name: Name of the component (e.g., 'browser', 'llm', 'agent')
        verbose: Whether to enable verbose logging for this component

    Returns:
        Configured logger instance
    """
    level = "DEBUG" if verbose else "INFO"

    # Special handling for high-frequency components
    if component_name.lower() in ["llm", "model", "tokenizer"]:
        level = "WARNING" if not verbose else "INFO"

    return get_optimized_logger(name=component_name, level=level)


# Create default logger with optimized settings
logger = define_log_level(
    print_level=os.getenv("LOG_LEVEL", "INFO"),
    logfile_level=os.getenv("FILE_LOG_LEVEL", "DEBUG"),
    max_size=os.getenv("LOG_MAX_SIZE", "10 MB"),
    retention=os.getenv("LOG_RETENTION", "7 days"),
    enable_file_logging=os.getenv("ENABLE_FILE_LOGGING", "true").lower() == "true",
)


# Utility functions for conditional logging
def log_if_verbose(level: str, message: str, verbose: bool = False):
    """Log message only if verbose mode is enabled."""
    if verbose:
        getattr(logger, level.lower())(message)


def log_performance(func_name: str, duration: float, threshold: float = 1.0):
    """Log performance metrics only if duration exceeds threshold."""
    if duration > threshold:
        logger.warning(
            f"Performance: {func_name} took {duration:.2f}s (threshold: {threshold}s)"
        )
    else:
        logger.debug(f"Performance: {func_name} took {duration:.2f}s")


if __name__ == "__main__":
    # Test the optimized logging system
    logger.info("Testing optimized logging system")
    logger.debug("Debug message - should appear in file only")
    logger.warning("Warning message - should appear in both console and file")
    logger.error("Error message - should appear in both console and file")

    # Test component-specific loggers
    llm_logger = create_component_logger("llm", verbose=False)
    llm_logger.info("LLM operation completed")  # Should be filtered
    llm_logger.warning("LLM warning")  # Should appear

    browser_logger = create_component_logger("browser", verbose=True)
    browser_logger.info("Browser operation completed")  # Should appear
