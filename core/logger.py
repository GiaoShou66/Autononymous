"""Logging configuration for the Autononymous agent.

Provides centralized logging setup with file and console handlers.
"""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(
    name: str,
    log_path: Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.

    Args:
        name: Logger name (usually __name__)
        log_path: Optional path to log file. If None, no file handler is added.
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture all levels, handlers filter

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # File handler
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch_formatter = logging.Formatter("%(levelname)s: %(message)s")
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger
