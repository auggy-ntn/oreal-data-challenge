"""Utilities for loading model inputs."""

import yaml

from constants.paths import PARAMS_FILE


def load_params() -> dict:
    """Load parameters from /params.yaml.

    Returns:
        dict: Parameters dictionary.
    """
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)
