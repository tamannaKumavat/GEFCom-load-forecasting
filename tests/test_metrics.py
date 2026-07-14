"""Tests verifying the metric implementations against hand-computed examples."""

import numpy as np
import pytest

from src.metrics import (
    mean_pinball_loss,
    calibration_coverage,
    interval_coverage,
    diebold_mariano_test,
)


def test_mean_pinball_loss_matches_hand_computation():
    # actual=108, predicted median=107.5 -> error=0.5 (>=0) -> loss = tau * error = 0.5 * 0.5
    y_true = np.array([108.0])
    y_pred = np.array([[107.5]])
    loss = mean_pinball_loss(y_true, y_pred, np.array([0.5]))
    assert loss == pytest.approx(0.25)


def test_pinball_loss_penalizes_undershoot_more_at_high_quantile():
    # at tau=0.95, undershooting the true value should cost more than overshooting by the same amount
    y_true = np.array([100.0])
    tau = np.array([0.95])
    over = mean_pinball_loss(y_true, np.array([[110.0]]), tau)
    under = mean_pinball_loss(y_true, np.array([[90.0]]), tau)
    assert over < under


def test_calibration_coverage_matches_hand_computation():
    y_true = np.array([1, 2, 3, 4, 5], dtype=float)
    quantile_preds = np.full((5, 1), 3.0)  # predict 3 for every row
    result = calibration_coverage(y_true, quantile_preds, np.array([0.5]))
    # 3/5 of actuals are <= 3
    assert result["observed"].iloc[0] == pytest.approx(3 / 5)


def test_interval_coverage_full_containment():
    y_true = np.array([100.0, 102.0, 98.0])
    quantiles = np.array([0.05, 0.95])
    preds = np.tile(np.array([[90.0, 110.0]]), (3, 1))
    result = interval_coverage(y_true, preds, quantiles, nominal_level=0.90)
    assert result["observed_coverage"] == 1.0


def test_diebold_mariano_favors_the_more_accurate_model():
    rng = np.random.default_rng(0)
    y_true = rng.normal(100, 10, 200)
    quantiles = np.array([0.5])
    preds_a = (y_true + rng.normal(0, 1, 200)).reshape(-1, 1)   # small errors
    preds_b = (y_true + rng.normal(0, 5, 200)).reshape(-1, 1)   # larger errors

    result = diebold_mariano_test(y_true, preds_a, preds_b, quantiles, alternative="less")

    assert result["test_statistic"] < 0
    assert result["p_value"] < 0.05
