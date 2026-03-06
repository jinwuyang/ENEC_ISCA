import logging
import os
from datetime import datetime

class LoggerGenerator:
    """
    General logger generator.
    Pass in the log directory path to return a configured logger instance.
    """
    # Cache for created loggers to avoid duplicate configuration
    _logger_cache = {}

    @staticmethod
    def get_logger(log_dir, name=None, console_output=True):
        """
        Get a configured logger.

        :param log_dir: Log folder path.
        :param name: Logger name (defaults to calling module's name).
        :param console_output: Whether to output to console simultaneously.
        :return: logging.Logger instance.
        """
        # If name is not specified, use the caller's module name
        if name is None:
            import inspect
            caller_frame = inspect.currentframe().f_back
            name = caller_frame.f_globals.get("__name__", "unknown")

        # Check cache to avoid adding handlers repeatedly
        if name in LoggerGenerator._logger_cache:
            return LoggerGenerator._logger_cache[name]

        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)

        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)  # Capture all levels
        logger.propagate = False  # Prevent upward propagation to avoid duplicate logs

        # Avoid adding handlers repeatedly
        if logger.hasHandlers():
            logger.handlers.clear()

        # Log filename: split by day + exact differentiation
        log_filename = os.path.join(
            log_dir,
            f"app_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        )

        # File handler - record all levels of logs to a file
        file_handler = logging.FileHandler(log_filename, encoding='utf-8', mode='a')
        file_handler.setLevel(logging.DEBUG)

        # Console handler (optional)
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)  # Console only shows INFO and above

        # Formatter
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(name)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        if console_output:
            console_handler.setFormatter(formatter)

        # Add handlers
        logger.addHandler(file_handler)
        if console_output:
            logger.addHandler(console_handler)

        # Cache logger
        LoggerGenerator._logger_cache[name] = logger

        return logger


# Usage example
if __name__ == "__main__":
    # Specify log directory
    log_directory = "./logs"

    # Get logger
    logger = LoggerGenerator.get_logger(log_directory, name="model", console_output=True)

    # Write logs (all levels will be written to the file)
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
