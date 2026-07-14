"""Load and manage experiment configuration."""

import numpy as np
import yaml
from typing import Any


def load_config(path: str = "configs/default.yaml") -> dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_quantiles(config: dict[str, Any]) -> np.ndarray:
    """Build the array of quantile levels to forecast from config["evaluation"]."""
    ev = config["evaluation"]
    start, end, step = ev["quantiles_start"], ev["quantiles_end"], ev["quantiles_step"]
    return np.round(np.arange(start, end + step / 2, step), 2)
