import logging
import re
import sys


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record):
        # Add color to the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
            )
        # Format the message
        message = super().format(record)
        # Add color to specific parts of the message
        if levelname == "INFO":
            # Highlight numbers and percentages
            message = re.sub(
                r"(\d+\.?\d*\s*(?:GB|MB|%|docs))",
                rf"{self.BOLD}\1{self.RESET}",
                message,
            )
            message = re.sub(
                r"(Shard \d+)",
                rf"{self.COLORS['INFO']}{self.BOLD}\1{self.RESET}",
                message,
            )
        return message


def get_logger(name="my_logger", level=logging.INFO, log_file=None):
    """Creates and returns a logger.

    Args:
        name (str): Logger name.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file (str, optional): Path to log file. If None, only prints to console.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Prevent duplicate logs if root logger is used

    # Clear previous handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(
        ColoredFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(ch)

    return logger
