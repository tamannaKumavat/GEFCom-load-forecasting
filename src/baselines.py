"""Baseline models for probabilistic load forecasting.

Two baselines:
1. Seasonal-Naïve: Uses the same hour from the same weekday last week as
   the point forecast, then derives quantiles from historical residuals.
2. Climatology (Empirical Quantile): For each (hour, month, dayofweek),
   compute empirical quantiles from the training set.
"""

import numpy as np
import pandas as pd
from typing import Optional


class SeasonalNaiveBaseline:
    """Seasonal-naïve baseline with empirical residual quantiles.

    Point forecast: load from the same hour, same weekday, one week ago.
    Quantiles: derived by adding empirical quantiles of past residuals
    (actual - naive_prediction) to the point forecast.
    """

    def __init__(self, lag_weeks: int = 1):
        self.lag_hours = lag_weeks * 168
        self.residual_quantiles = None

    def fit(self, train_df: pd.DataFrame, quantiles: np.ndarray):
        """Compute residual distribution from training data.

        Parameters
        ----------
        train_df : DataFrame with datetime index and 'load' column.
        quantiles : Array of quantile levels to predict.
        """
        self.quantiles = quantiles
        load = train_df["load"].values
        naive_pred = train_df["load"].shift(self.lag_hours).values

        # Residuals where both actual and prediction are available
        mask = ~(np.isnan(load) | np.isnan(naive_pred))
        residuals = load[mask] - naive_pred[mask]

        # Compute quantiles of residual distribution, grouped by hour
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
        """Produce quantile forecasts for the test period.

        Parameters
        ----------
        train_df : Available history (includes data up to forecast origin).
        test_index : DatetimeIndex of the forecast period.

        Returns
        -------
        Array of shape (len(test_index), n_quantiles)
        """
        full_load = train_df["load"]
        n_test = len(test_index)
        n_q = len(self.quantiles)
        preds = np.full((n_test, n_q), np.nan)

        for i, dt in enumerate(test_index):
            lag_dt = dt - pd.Timedelta(hours=self.lag_hours)
            if lag_dt in full_load.index and not np.isnan(full_load.loc[lag_dt]):
                point = full_load.loc[lag_dt]
            else:
                # Fallback: use mean load at this hour from training
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
    """Empirical-quantile climatology baseline.

    For each (hour_of_day, month, is_weekend) combination, compute the
    empirical quantiles of load from the training set.
    """

    def __init__(self, lookback_years: Optional[int] = None):
        self.lookback_years = lookback_years
        self.quantile_table = None

    def fit(self, train_df: pd.DataFrame, quantiles: np.ndarray):
        """Compute empirical quantiles per (hour, month, is_weekend).

        Parameters
        ----------
        train_df : DataFrame with datetime index and 'load' column.
        quantiles : Array of quantile levels.
        """
        self.quantiles = quantiles
        df = train_df[["load"]].dropna().copy()

        # Optionally restrict to recent history
        if self.lookback_years and len(df) > 0:
            cutoff = df.index.max() - pd.DateOffset(years=self.lookback_years)
            df = df[df.index >= cutoff]

        df["hour"] = df.index.hour
        df["month"] = df.index.month
        df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

        # Compute quantiles per group
        self.quantile_table = {}
        for (h, m, w), group in df.groupby(["hour", "month", "is_weekend"]):
            self.quantile_table[(h, m, w)] = np.quantile(
                group["load"].values, quantiles
            )

        # Global fallback
        self.global_quantiles = np.quantile(df["load"].values, quantiles)

    def predict(self, train_df: pd.DataFrame, test_index: pd.DatetimeIndex) -> np.ndarray:
        """Produce quantile forecasts for the test period.

        Parameters
        ----------
        train_df : Not used directly (already fitted), kept for API consistency.
        test_index : DatetimeIndex of the forecast period.

        Returns
        -------
        Array of shape (len(test_index), n_quantiles)
        """
        n_test = len(test_index)
        n_q = len(self.quantiles)
        preds = np.full((n_test, n_q), np.nan)

        for i, dt in enumerate(test_index):
            key = (dt.hour, dt.month, int(dt.dayofweek >= 5))
            preds[i, :] = self.quantile_table.get(key, self.global_quantiles)

        return preds
