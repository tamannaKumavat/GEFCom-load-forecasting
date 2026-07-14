"""Two simple baselines: seasonal-naive and climatology."""

import numpy as np
import pandas as pd
from typing import Optional


class SeasonalNaiveBaseline:
    """Point forecast = same hour/weekday N weeks ago, quantiles from historical residuals."""

    def __init__(self, lag_weeks: int = 1):
        self.lag_hours = lag_weeks * 168
        self.residual_quantiles = None

    def fit(self, train_df: pd.DataFrame, quantiles: np.ndarray):
        self.quantiles = quantiles
        load = train_df["load"].values
        naive_pred = train_df["load"].shift(self.lag_hours).values

        mask = ~(np.isnan(load) | np.isnan(naive_pred))
        residuals = load[mask] - naive_pred[mask]

        hours = train_df.index.hour.values[mask]
        self.residual_quantiles_by_hour = {}
        for h in range(24):
            h_mask = hours == h
            if h_mask.sum() > 0:
                self.residual_quantiles_by_hour[h] = np.quantile(
                    residuals[h_mask], quantiles
                )
            else:
                self.residual_quantiles_by_hour[h] = np.quantile(residuals, quantiles)

    def predict(self, train_df: pd.DataFrame, test_index: pd.DatetimeIndex) -> np.ndarray:
        full_load = train_df["load"]
        n_test = len(test_index)
        n_q = len(self.quantiles)
        preds = np.full((n_test, n_q), np.nan)

        for i, dt in enumerate(test_index):
            lag_dt = dt - pd.Timedelta(hours=self.lag_hours)
            if lag_dt in full_load.index and not np.isnan(full_load.loc[lag_dt]):
                point = full_load.loc[lag_dt]
            else:
                h = dt.hour
                h_mask = full_load.index.hour == h
                point = full_load[h_mask].mean()

            h = dt.hour
            residual_q = self.residual_quantiles_by_hour.get(
                h, np.zeros(n_q)
            )
            preds[i, :] = point + residual_q

        return preds


class ClimatologyBaseline:
    """Empirical load quantiles per (hour, month, is_weekend) group from training data."""

    def __init__(self, lookback_years: Optional[int] = None):
        self.lookback_years = lookback_years
        self.quantile_table = None

    def fit(self, train_df: pd.DataFrame, quantiles: np.ndarray):
        self.quantiles = quantiles
        df = train_df[["load"]].dropna().copy()

        if self.lookback_years and len(df) > 0:
            cutoff = df.index.max() - pd.DateOffset(years=self.lookback_years)
            df = df[df.index >= cutoff]

        df["hour"] = df.index.hour
        df["month"] = df.index.month
        df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

        self.quantile_table = {}
        for (h, m, w), group in df.groupby(["hour", "month", "is_weekend"]):
            self.quantile_table[(h, m, w)] = np.quantile(
                group["load"].values, quantiles
            )

        self.global_quantiles = np.quantile(df["load"].values, quantiles)

    def predict(self, train_df: pd.DataFrame, test_index: pd.DatetimeIndex) -> np.ndarray:
        # train_df unused -- already fitted, kept for API symmetry with SeasonalNaiveBaseline
        n_test = len(test_index)
        n_q = len(self.quantiles)
        preds = np.full((n_test, n_q), np.nan)

        for i, dt in enumerate(test_index):
            key = (dt.hour, dt.month, int(dt.dayofweek >= 5))
            preds[i, :] = self.quantile_table.get(key, self.global_quantiles)

        return preds
