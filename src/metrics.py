"""Evaluation metrics for probabilistic load forecasting. """

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_pinball_loss as _sklearn_pinball_loss
from typing import Optional


def _pinball_loss_elementwise(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> np.ndarray:
    """Per-observation pinball loss, without averaging.

    Needed for the Diebold-Mariano test below, which requires the raw
    per-timestep loss sequence rather than a single aggregate value (which is
    all sklearn's mean_pinball_loss exposes).
    """
    diff = y_true - y_pred
    return np.where(diff >= 0, tau * diff, (tau - 1) * diff)


def mean_pinball_loss(
    y_true: np.ndarray,
    quantile_preds: np.ndarray,
    quantiles: np.ndarray,
) -> float:
    """Compute mean pinball loss averaged over all quantile levels.

    Parameters
    ----------
    y_true : Actual values, shape (n,)
    quantile_preds : Predicted quantiles, shape (n, n_quantiles)
    quantiles : Quantile levels, shape (n_quantiles,)

    Returns
    -------
    Mean pinball loss across all quantiles (lower is better).
    """
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
    """Assess calibration by checking observed coverage at each quantile level.

    For a well-calibrated model, the fraction of observations below the
    tau-th quantile prediction should be approximately tau.

    Parameters
    ----------
    y_true : Actual values, shape (n,)
    quantile_preds : Predicted quantiles, shape (n, n_quantiles)
    quantiles : Quantile levels, shape (n_quantiles,)

    Returns
    -------
    DataFrame with columns: quantile, nominal, observed
    """
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
    """Check if a nominal X% prediction interval has ~X% coverage.

    Uses the symmetric interval: [quantile at (1-level)/2, quantile at (1+level)/2].

    Parameters
    ----------
    y_true : Actual values
    quantile_preds : shape (n, n_quantiles)
    quantiles : Quantile levels
    nominal_level : e.g. 0.90 for 90% interval

    Returns
    -------
    Dict with nominal_level, lower_q, upper_q, observed_coverage
    """
    lower_q = (1 - nominal_level) / 2
    upper_q = 1 - lower_q

    # Find nearest quantile indices
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
    """Diebold-Mariano test comparing two sets of quantile forecasts.

    Tests H0: E[L_A - L_B] = 0, where L is the pinball loss.
    A negative test statistic means model A has lower loss (is better).

    Parameters
    ----------
    y_true : Actual values, shape (n,)
    preds_a : Quantile predictions from model A, shape (n, n_quantiles)
    preds_b : Quantile predictions from model B, shape (n, n_quantiles)
    quantiles : Quantile levels
    alternative : 'two-sided', 'less' (A < B), or 'greater' (A > B)

    Returns
    -------
    Dict with test_statistic, p_value, mean_diff (negative = A better)
    """
    # Compute per-observation average pinball loss
    y_true = np.asarray(y_true, dtype=float)
    loss_a = np.zeros(len(y_true))
    loss_b = np.zeros(len(y_true))
    for j, tau in enumerate(quantiles):
        loss_a += _pinball_loss_elementwise(y_true, preds_a[:, j], tau)
        loss_b += _pinball_loss_elementwise(y_true, preds_b[:, j], tau)
    loss_a /= len(quantiles)
    loss_b /= len(quantiles)

    d = loss_a - loss_b  # loss differences

    # Newey-West style variance (simple version with lag-1 autocorrelation)
    n = len(d)
    d_mean = np.mean(d)
    # Use HAC variance estimator
    gamma_0 = np.var(d, ddof=1)
    # Truncation at h = int(n^(1/3))
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
