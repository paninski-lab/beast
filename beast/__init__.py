# Hacky way to get version from pypackage.toml.
# Adapted from: https://github.com/python-poetry/poetry/issues/273#issuecomment-1877789967
import importlib.metadata
import time
from pathlib import Path
from typing import Any, Optional

__package_version = "unknown"


def log_step(
    msg: str,
    level: Optional[str] = None,
    flush: bool = True,
    logger: Any = None,
) -> None:
    """Unified logging function with optional level.

    Parameters
    ----------
    msg: message to log
    level: None (plain timestamp + msg), 'info', 'debug', or 'error'
    flush: whether to flush stdout after printing
    logger: if provided and level is 'info', also call logger.info(msg);
        if provided and level is 'error', also call logger.error(msg)
    """
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    if level == 'info':
        if logger is not None:
            logger.info(msg)
        else:
            print(f"[{timestamp}] INFO: {msg}", flush=flush)
    elif level == 'debug':
        if logger is not None:
            logger.debug(msg)
        else:
            print(f"[{timestamp}] DEBUG: {msg}", flush=flush)
    elif level == 'error':
        if logger is not None:
            logger.error(msg)
        else:
            print(f"[{timestamp}] ERROR: {msg}", flush=flush)
    else:
        print(f"[{timestamp}] {msg}", flush=flush)


def __get_package_version() -> str:
    """Find the version of this package."""

    global __package_version

    if __package_version != 'unknown':
        # We already set it at some point in the past,
        # so return that previous value without any
        # extra work.
        return __package_version

    try:
        # Try to get the version of the current package if
        # it is running from a distribution.
        __package_version = importlib.metadata.version('beast')
    except importlib.metadata.PackageNotFoundError:
        # Fall back on getting it from a local pyproject.toml.
        # This works in a development environment where the
        # package has not been installed from a distribution.
        import warnings

        import toml

        warnings.warn('beast not pip-installed, getting version from pyproject.toml.')

        pyproject_toml_file = Path(__file__).parent.parent / 'pyproject.toml'
        __package_version = toml.load(pyproject_toml_file)['project']['version']

    return __package_version


def __getattr__(name: str) -> Any:
    """Get package attributes."""
    if name in ('version', '__version__'):
        return __get_package_version()
    else:
        raise AttributeError(f'No attribute {name} in module {__name__}.')
