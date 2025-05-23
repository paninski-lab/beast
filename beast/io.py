from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load yaml configuration file to a nested dictionary structure.

    Parameters
    ----------
    path: absolute path to config yaml file

    Returns
    -------
    nested configuration dictionary

    """

    path = Path(path)
    assert path.is_file(), f'{path} does not exist'

    with open(path, 'r') as file:
        config = yaml.safe_load(file)

    return config


def apply_config_overrides(config: dict, overrides: dict | list) -> dict:
    """Apply configuration overrides to a nested dictionary structure.

    Parameters
    ----------
    config: base configuration dictionary to modify
    overrides:
        dictionary with dot-notation keys and values to override
        list with KEY=VALUE entries, keys use dot-notation

    Returns
    -------
    The modified configuration dictionary

    """

    if isinstance(overrides, list):
        overrides = {item.split('=')[0]: item.split('=')[1] for item in overrides}

    for field, value in overrides.items():
        keys = field.split('.')
        current = config

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the final value
        current[keys[-1]] = value

    return config
