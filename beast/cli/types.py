"""Type validators for CLI arguments."""

from pathlib import Path


def valid_file(path_str):
    """Validate that a path exists and is a file."""
    path = Path(path_str)
    if not path.exists():
        raise ValueError(f"File does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    return path


def valid_dir(path_str):
    """Validate that a path exists and is a directory."""
    path = Path(path_str)
    if not path.exists():
        raise ValueError(f"Directory does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Not a directory: {path}")
    return path


def config_file(path_str):
    """Validate a config file path."""
    path = valid_file(path_str)
    ext = path.suffix.lower()
    if ext not in (".yaml", ".yml", ".json"):
        raise ValueError(f"Config file must be YAML or JSON: {path}")
    return path


def output_dir(path_str):
    """Create output directory if it doesn't exist."""
    path = Path(path_str)
    path.mkdir(parents=True, exist_ok=True)
    return path
