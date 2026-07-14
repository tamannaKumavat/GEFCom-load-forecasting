"""Feature engineering for load forecasting.

Load lags are capped to >=744h (longest possible forecast month) so they
can't leak into the test period -- see README for details. This leaves
only the 8760h (1yr) lag; shorter ones were dropped for that reason.
"""

import pandas as pd
import numpy as np
from typing import Any


def build_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    feat = config.get("features", {})
    out = df.copy()

    if feat.get("use_hour", True):
        out["hour"] = out.index.hour
    if feat.get("use_dayofweek", True):
        out["dayofweek"] = out.index.dayofweek
    if feat.get("use_month", True):
        out["month"] = out.index.month
    if feat.get("use_is_weekend", True):
        out["is_weekend"] = (out.index.dayofweek >= 5).astype(int)

    # cyclical encoding so e.g. hour 23 and hour 0 are close together
    out["hour_sin"] = np.sin(2 * np.pi * out.index.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out.index.hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out.index.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out.index.dayofweek / 7)
    out["month_sin"] = np.sin(2 * np.pi * (out.index.month - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (out.index.month - 1) / 12)

    if feat.get("use_temperature", True):
        temp_col = "temp_mean"
        if temp_col in out.columns:
            # heating/cooling degree hours capture the U-shaped load/temp relationship
            out["hdh"] = np.maximum(18.0 - out[temp_col], 0)
            out["cdh"] = np.maximum(out[temp_col] - 18.0, 0)

        for lag in feat.get("temp_lags_hours", []):
            if temp_col in out.columns:
                out[f"temp_lag_{lag}h"] = out[temp_col].shift(lag)

    for lag in feat.get("load_lags_hours", []):
        out[f"load_lag_{lag}h"] = out["load"].shift(lag)

    for window in feat.get("rolling_windows_hours", []):
        min_lag = min(feat.get("load_lags_hours", [8760]))
        shifted = out["load"].shift(min_lag)
        out[f"load_rolling_mean_{window}h"] = shifted.rolling(window, min_periods=1).mean()
        out[f"load_rolling_std_{window}h"] = shifted.rolling(window, min_periods=1).std()

    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """All columns except load and the raw per-station weather columns (w1..w25)."""
    exclude = {"load"}
    exclude.update(c for c in df.columns if c.startswith("w") and c[1:].isdigit())
    exclude.update(c for c in df.columns if c.startswith("temp_") and c != "temp_mean")
    return [c for c in df.columns if c not in exclude]
