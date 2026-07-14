"""Load and manage experiment configuration."""

import yaml
from typing import Any


def load_config(path: str = "configs/default.yaml") -> dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config
