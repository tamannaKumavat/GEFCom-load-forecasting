"""Tests that guard against future-data leakage in the backtest."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data import get_task_splits
from src.features import build_features

HAS_DATA = Path("data/Load").exists()


def test_load_lags_are_leakage_safe():
    """Every configured load lag must be >= the longest possible test month (~744h),
    otherwise it could resolve to a real value from later in the same test period."""
    config = load_config("configs/default.yaml")
    max_test_month_hours = 31 * 24
    for lag in config["features"]["load_lags_hours"]:
        assert lag >= max_test_month_hours


@pytest.mark.skipif(not HAS_DATA, reason="requires GEFCom2014 data under data/")
def test_rolling_origin_folds_have_no_train_test_overlap():
    """train_end must be strictly before test_start for every fold."""
    splits = get_task_splits("data")
    assert len(splits) > 0
    for split in splits:
        assert split["train_end"] < split["test_start"]
        assert split["test_start"] <= split["test_end"]


def test_load_lag_feature_uses_past_values_only():
    """A load lag feature must come from `lag` hours earlier, never from later in the series."""
    idx = pd.date_range("2020-01-01", periods=100, freq="h")
    df = pd.DataFrame({"load": np.arange(100, dtype=float)}, index=idx)
    config = {"features": {"load_lags_hours": [5], "rolling_windows_hours": [], "use_temperature": False}}

    out = build_features(df, config)

    assert out["load_lag_5h"].iloc[10] == df["load"].iloc[5]
    assert pd.isna(out["load_lag_5h"].iloc[3])  # not enough history yet, not a peek into the future
