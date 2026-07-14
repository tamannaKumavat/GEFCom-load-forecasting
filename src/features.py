"""Feature engineering for load forecasting.

All features are constructed respecting temporal ordering:
- Calendar features (hour, day-of-week, month, etc.) are deterministic and safe.
- Temperature features use the actual observed temperature. This is an explicit
  assumption documented in the config(in production settings, forecasted temperatures
  would be used instead)
- Lagged load features use only past values. All lags are capped to be >= the
  longest possible forecast horizon (~744h, a 31-day month), so a lag can
  never reach into the test period itself. This currently leaves only the
  8760h (1-year) load lag; shorter lags (168h/336h/etc.) were dropped because
  they could otherwise pull in real values from later in the same test month.
"""

import pandas as pd
import numpy as np
from typing import Any


def build_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Build all features from the merged load+temperature DataFrame.

    Parameters
    ----------
    df : DataFrame with datetime index, columns include 'load' and 'temp_mean'.
    config : Experiment configuration dict.

    Returns
    -------
    DataFrame with original columns plus engineered features.
    """
    feat = config.get("features", {})
    out = df.copy()

    #Calendar features
    if feat.get("use_hour", True):
        out["hour"] = out.index.hour
    if feat.get("use_dayofweek", True):
        out["dayofweek"] = out.index.dayofweek
    if feat.get("use_month", True):
        out["month"] = out.index.month
    if feat.get("use_is_weekend", True):
        out["is_weekend"] = (out.index.dayofweek >= 5).astype(int)

    # Cyclical encoding for hour and day-of-week
    out["hour_sin"] = np.sin(2 * np.pi * out.index.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out.index.hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out.index.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out.index.dayofweek / 7)
    out["month_sin"] = np.sin(2 * np.pi * (out.index.month - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (out.index.month - 1) / 12)

    # --- Temperature features ---
    if feat.get("use_temperature", True):
        # Actual temperature (see assumption in config)
        # Already present as temp_mean
        # Heating/cooling degree hours capture the U-shaped load/temperature
        # relationship (see EDA notebook); temp_sq would encode the same
        # nonlinearity redundantly, so it's intentionally left out.
        temp_col = "temp_mean"
        if temp_col in out.columns:
            # Heating degree hours (below 65°F / ~18°C threshold)
            out["hdh"] = np.maximum(18.0 - out[temp_col], 0)
            # Cooling degree hours (above 65°F / ~18°C threshold)
            out["cdh"] = np.maximum(out[temp_col] - 18.0, 0)

        # Lagged temperature: same leakage constraint as load lags applies,
        # and there's no well-motivated long-horizon temp lag, so this is
        # left empty by default (see temp_lags_hours in config).
        for lag in feat.get("temp_lags_hours", []):
            if temp_col in out.columns:
                out[f"temp_lag_{lag}h"] = out[temp_col].shift(lag)

    # --- Lagged load features ---
    # All lags in config are capped to be leakage-safe (see module docstring).
    for lag in feat.get("load_lags_hours", []):
        out[f"load_lag_{lag}h"] = out["load"].shift(lag)

    # --- Rolling statistics on load ---
    # Anchored to the same (leakage-safe) minimum lag as above, so these
    # inherit the same safety guarantee automatically.
    for window in feat.get("rolling_windows_hours", []):
        min_lag = min(feat.get("load_lags_hours", [8760]))
        shifted = out["load"].shift(min_lag)
        out[f"load_rolling_mean_{window}h"] = shifted.rolling(window, min_periods=1).mean()
        out[f"load_rolling_std_{window}h"] = shifted.rolling(window, min_periods=1).std()

    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return list of feature column names (everything except load and raw temp stations)."""
    exclude = {"load"}
    # Exclude the 25 raw per-station weather columns (w1..w25) and any
    # leftover temp_* helper columns other than temp_mean -- keep only the
    # aggregated/derived temperature features, not the raw stations
    # (justified in the EDA notebook: stations are highly correlated,
    # r >= 0.87, so temp_mean captures nearly all of the signal).
    exclude.update(c for c in df.columns if c.startswith("w") and c[1:].isdigit())
    exclude.update(c for c in df.columns if c.startswith("temp_") and c != "temp_mean")
    return [c for c in df.columns if c not in exclude]
