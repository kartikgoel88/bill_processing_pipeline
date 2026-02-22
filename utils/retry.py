"""Retry with exponential backoff. No global state."""

from __future__ import annotations

import time
import logging
from typing import Callable, TypeVar

from core.exceptions import BillProcessingError

logger = logging.getLogger(__name__)
T = TypeVar("T")


def with_retry(
    fn: Callable[[], T],
    max_attempts: int = 3,
    delay_sec: float = 2.0,
    backoff: bool = True,
    retry_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """
    Execute fn; on retry_exceptions retry with exponential backoff.
    Raises last exception after max_attempts.
    """
    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except retry_exceptions as e:
            last_err = e
            if attempt >= max_attempts - 1:
                raise
            wait = delay_sec * (2**attempt) if backoff else delay_sec
            logger.warning(
                "Retry attempt %s/%s after %.2fs: %s",
                attempt + 1,
                max_attempts,
                wait,
                e,
            )
            time.sleep(wait)
    if last_err:
        raise last_err
    raise RuntimeError("retry exhausted")
