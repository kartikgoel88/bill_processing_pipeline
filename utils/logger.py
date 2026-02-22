"""Structured logging; no global state beyond logging tree."""

from __future__ import annotations

import logging
import sys
from typing import Any


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module/component. No side effects."""
    return logging.getLogger(name)


def setup_logging(
    level: str = "INFO",
    format_string: str | None = None,
    stream: Any = None,
) -> None:
    """
    Configure root logger once. Safe to call from main or tests.
    """
    if format_string is None:
        format_string = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    stream = stream or sys.stdout
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=format_string,
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=stream,
        force=True,
    )


def log_structured(logger: logging.Logger, level: int, msg: str, **kwargs: Any) -> None:
    """Emit a log record with extra keys for structured aggregation."""
    logger.log(level, msg, extra=kwargs)
