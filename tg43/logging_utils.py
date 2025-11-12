"""Helper utilities for configuring application-wide logging."""

from __future__ import annotations

import logging
from typing import Optional

DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO, *, log_format: Optional[str] = None) -> logging.Logger:
    """Configure the root logger with a sensible default format.

    Parameters
    ----------
    level : int, optional
        Logging level applied to the root logger. Defaults to ``logging.INFO``.
    log_format : str, optional
        Custom format string. When omitted ``DEFAULT_FORMAT`` is used.

    Returns
    -------
    logging.Logger
        Root logger after configuration.
    """

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=level, format=log_format or DEFAULT_FORMAT)
    else:
        root_logger.setLevel(level)
        if log_format:
            formatter = logging.Formatter(log_format)
            for handler in root_logger.handlers:
                handler.setFormatter(formatter)
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger."""

    return logging.getLogger(name)
