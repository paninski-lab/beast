"""Unified logging utility for the BEAST package."""

import time
from typing import Any


def log_step(
    msg: str,
    level: str | None = None,
    flush: bool = True,
    logger: Any = None,
) -> None:
    """Unified logging function with optional level.

    Parameters
    ----------
    msg: message to log
    level: None (plain timestamp + msg), 'info', 'debug', or 'error'
    flush: whether to flush stdout after printing
    logger: if provided, delegate to logger.info/debug/error instead of printing

    """
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    if level == 'info':
        if logger is not None:
            logger.info(msg)
        else:
            print(f'[{timestamp}] INFO: {msg}', flush=flush)
    elif level == 'debug':
        if logger is not None:
            logger.debug(msg)
        else:
            print(f'[{timestamp}] DEBUG: {msg}', flush=flush)
    elif level == 'error':
        if logger is not None:
            logger.error(msg)
        else:
            print(f'[{timestamp}] ERROR: {msg}', flush=flush)
    else:
        print(f'[{timestamp}] {msg}', flush=flush)
