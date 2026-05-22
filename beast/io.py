"""Configuration file loading and override utilities."""

from pathlib import Path

import yaml

from beast.config import BeastConfig


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
    if not path.is_file():
        raise FileNotFoundError(f'{path} does not exist')

    # load raw yaml into an untyped dict
    with open(path) as file:
        raw = yaml.safe_load(file)

    # validate against the schema; raises ValidationError on missing required
    # fields, wrong types, or invalid Literal values
    validated = BeastConfig.model_validate(raw)

    # convert back to a plain nested dict so callers don't depend on pydantic types;
    # this also fills in any fields that have defaults but were absent from the yaml
    return validated.model_dump()


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
