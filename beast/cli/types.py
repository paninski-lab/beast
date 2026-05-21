"""Type validators for CLI arguments."""

from pathlib import Path


def valid_file(path_str: str | Path) -> Path:
    """Validate that a path exists and is a file."""
    path = Path(path_str)
    if not path.exists():
        raise OSError(f'File does not exist: {path}')
    if not path.is_file():
        raise OSError(f'Not a file: {path}')
    return path


def valid_dir(path_str: str | Path) -> Path:
    """Validate that a path exists and is a directory."""
    path = Path(path_str)
    if not path.exists():
        raise OSError(f'Directory does not exist: {path}')
    if not path.is_dir():
        raise OSError(f'Not a directory: {path}')
    return path


def config_file(path_str: str | Path) -> Path:
    """Validate a config file path."""
    path = valid_file(path_str)
    ext = path.suffix.lower()
    if ext not in ('.yaml', '.yml'):
        raise ValueError(f'Config file must be YAML: {path}')
    return path


def output_dir(path_str: str | Path) -> Path:
    """Create output directory if it does not exist."""
    path = Path(path_str)
    path.mkdir(parents=True, exist_ok=True)
    return path
