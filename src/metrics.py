"""Metrics for probabilistic load forecasting."""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_pinball_loss as _sklearn_pinball_loss
from typing import Optional


def _pinball_loss_elementwise(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> np.ndarray:
    # per-observation loss, needed for the DM test below (sklearn only gives an averaged value)
    diff = y_true - y_pred
    return np.where(diff >= 0, tau * diff, (tau - 1) * diff)


def mean_pinball_loss(
    y_true: np.ndarray,
    quantile_preds: np.ndarray,
    quantiles: np.ndarray,
) -> float:
    """Mean pinball loss across all quantile levels."""
    y_true = np.asarray(y_true, dtype=float)
    quantile_preds = np.asarray(quantile_preds, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)

    total = 0.0
    for j, tau in enumerate(quantiles):
        total += _sklearn_pinball_loss(y_true, quantile_preds[:, j], alpha=tau)
    return total / len(quantiles)


def calibration_coverage(
    y_true: np.ndarray,
    quantile_preds: np.ndarray,
    quantiles: np.ndarray,
) -> pd.DataFrame:
    """For each quantile level, what fraction of actuals fell below it (should ~= the level)."""
    y_true = np.asarray(y_true, dtype=float)
    results = []
    for j, tau in enumerate(quantiles):
        observed = np.mean(y_true <= quantile_preds[:, j])
        results.append({"quantile": tau, "nominal": tau, "observed": observed})
    return pd.DataFrame(results)


def interval_coverage(
    y_true: np.ndarray,
    quantile_preds: np.ndarray,
    quantiles: np.ndarray,
    nominal_level: float = 0.90,
) -> dict:
    """Observed coverage of the central interval at the given nominal level (e.g. 0.90)."""
    lower_q = (1 - nominal_level) / 2
    upper_q = 1 - lower_q

    lower_idx = np.argmin(np.abs(quantiles - lower_q))
    upper_idx = np.argmin(np.abs(quantiles - upper_q))

    lower_vals = quantile_preds[:, lower_idx]
    upper_vals = quantile_preds[:, upper_idx]

    in_interval = (y_true >= lower_vals) & (y_true <= upper_vals)
    observed = float(np.mean(in_interval))

    return {
        "nominal_level": nominal_level,
        "lower_quantile": quantiles[lower_idx],
        "upper_quantile": quantiles[upper_idx],
        "observed_coverage": observed,
    }


def diebold_mariano_test(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    quantiles: np.ndarray,
    alternative: str = "two-sided",
) -> dict:
    """Diebold-Mariano test (1995) on pinball loss, HAC/Newey-West variance. Negative stat = A better."""
    y_true = np.asarray(y_true, dtype=float)
    loss_a = np.zeros(len(y_true))
    loss_b = np.zeros(len(y_true))
    for j, tau in enumerate(quantiles):
        loss_a += _pinball_loss_elementwise(y_true, preds_a[:, j], tau)
        loss_b += _pinball_loss_elementwise(y_true, preds_b[:, j], tau)
    loss_a /= len(quantiles)
    loss_b /= len(quantiles)

    d = loss_a - loss_b

    n = len(d)
    d_mean = np.mean(d)
    gamma_0 = np.var(d, ddof=1)
    h = max(1, int(n ** (1.0 / 3.0)))
    hac_var = gamma_0
    for k in range(1, h + 1):
        gamma_k = np.mean((d[k:] - d_mean) * (d[:-k] - d_mean))
        hac_var += 2 * (1 - k / (h + 1)) * gamma_k

    dm_stat = d_mean / np.sqrt(hac_var / n) if hac_var > 0 else 0.0

    if alternative == "two-sided":
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    elif alternative == "less":
        p_value = stats.norm.cdf(dm_stat)
    else:
        p_value = 1 - stats.norm.cdf(dm_stat)

    return {
        "test_statistic": float(dm_stat),
        "p_value": float(p_value),
        "mean_loss_diff": float(d_mean),
        "alternative": alternative,
    }
