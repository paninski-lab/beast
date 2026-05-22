"""Logging utilities for the BEAST package."""

import logging
from typing import Any


def log_step(
    msg: str,
    level: str | None = None,
    flush: bool = True,
    logger: Any = None,
) -> None:
    """Log a message at the given level.

    Parameters
    ----------
    msg: message to log
    level: 'info', 'debug', or 'error'; defaults to 'info'
    flush: unused; retained for backward compatibility
    logger: if provided, use this logger instead of the default beast logger

    """
    _logger = logger if logger is not None else logging.getLogger('beast')
    if level == 'debug':
        _logger.debug(msg)
    elif level == 'error':
        _logger.error(msg)
    else:
        _logger.info(msg)
