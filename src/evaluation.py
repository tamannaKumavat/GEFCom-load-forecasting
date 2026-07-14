"""Rolling-origin (expanding window) evaluation framework.

Implements a strict temporal backtesting scheme:
- The training set grows with each fold (expanding window).
- The test set is always one month ahead of the training cutoff.
- No information from the test period or later leaks into training,
  preprocessing, or feature construction.

This mirrors the GEFCom2014 competition structure, where each round
reveals one more month of data.
"""

import numpy as np
import pandas as pd
from typing import Any
from src.config import get_quantiles
from src.data import get_task_splits, load_benchmark
from src.features import build_features, get_feature_columns
from src.baselines import SeasonalNaiveBaseline, ClimatologyBaseline
from src.model import LightGBMQuantileModel
from src.metrics import mean_pinball_loss, diebold_mariano_test


def create_task_based_splits(config: dict[str, Any]) -> list[dict]:
    """Create train/test splits following the GEFCom2014 competition rounds.

    Uses the actual task boundaries (data.get_task_splits): fold k trains on
    everything through round k and tests on round k+1's month. This gives
    15 folds total instead of scanning every month, which is much cheaper
    to run.
    """
    data_dir = config["data"]["raw_dir"]
    task_splits = get_task_splits(data_dir)

    max_rounds = config.get("evaluation", {}).get("max_rounds")
    if max_rounds and max_rounds < len(task_splits):
        task_splits = task_splits[-max_rounds:]

    splits = [
        {
            "fold": s["task"],
            "train_end": s["train_end"],
            "test_start": s["test_start"],
            "test_end": s["test_end"],
        }
        for s in task_splits
    ]

    print(f"Created {len(splits)} task-based folds")
    if splits:
        print(f"  First fold: train until {splits[0]['train_end']}, "
              f"test {splits[0]['test_start']} to {splits[0]['test_end']}")
        print(f"  Last fold:  train until {splits[-1]['train_end']}, "
              f"test {splits[-1]['test_start']} to {splits[-1]['test_end']}")
    return splits


def run_evaluation(
    df: pd.DataFrame,
    config: dict[str, Any],
    quantile_step: int = 5,
) -> dict:
    """Run rolling-origin evaluation of the baselines and LightGBM.

    Parameters
    ----------
    df : Prepared dataset (load + temperature, datetime index).
    config : Experiment configuration.
    quantile_step : Fit every N-th quantile level directly for LightGBM
                    (the rest are interpolated); use 1 to fit all 99.

    Returns
    -------
    Dict with per-fold and aggregate results.
    """
    data_dir = config["data"]["raw_dir"]
    quantiles = get_quantiles(config)
    splits = create_task_based_splits(config)

    if not splits:
        raise ValueError("No valid evaluation splits found. Check data range.")

    # Initialize baselines
    seasonal_naive = SeasonalNaiveBaseline(
        lag_weeks=config.get("baselines", {}).get("seasonal_naive", {}).get("lag_weeks", 1)
    )
    climatology = ClimatologyBaseline(
        lookback_years=config.get("baselines", {}).get("climatology", {}).get("lookback_years")
    )

    # Features are built once on the whole series: every lag is capped to be
    # >= the longest possible test month, so this can never leak test-period
    # values into a fold's training features (see src/features.py).
    full_featured = build_features(df, config)
    feature_cols = get_feature_columns(full_featured)

    fold_results = []
    last_fold_predictions = None

    for split in splits:
        fold = split["fold"]
        print(f"\n{'='*60}")
        print(f"Fold {fold}: train until {split['train_end']}, "
              f"test {split['test_start']} to {split['test_end']}")
        print(f"{'='*60}")

        # --- Strict temporal split ---
        train_mask = df.index <= split["train_end"]
        test_mask = (df.index >= split["test_start"]) & (df.index <= split["test_end"])

        train_raw = df[train_mask].copy()
        test_raw = df[test_mask].copy()

        # Drop rows with missing load in test
        test_raw = test_raw.dropna(subset=["load"])
        if len(test_raw) == 0:
            print(f"  Skipping fold {fold}: no test data.")
            continue

        y_test = test_raw["load"].values
        test_index = test_raw.index

        # --- Baseline 1: Seasonal Naïve ---
        print("  Fitting seasonal-naive baseline...")
        seasonal_naive.fit(train_raw, quantiles)
        preds_naive = seasonal_naive.predict(train_raw, test_index)
        loss_naive = mean_pinball_loss(y_test, preds_naive, quantiles)
        print(f"  Seasonal-naive pinball loss: {loss_naive:.2f}")

        # --- Baseline 2: Climatology ---
        print("  Fitting climatology baseline...")
        climatology.fit(train_raw, quantiles)
        preds_clim = climatology.predict(train_raw, test_index)
        loss_clim = mean_pinball_loss(y_test, preds_clim, quantiles)
        print(f"  Climatology pinball loss: {loss_clim:.2f}")

        # --- Baseline 3: Official GEFCom2014 benchmark ---
        print("  Loading official benchmark forecast...")
        benchmark_series = load_benchmark(data_dir, fold, prev_date=split["train_end"]).reindex(test_index)
        preds_benchmark = np.tile(benchmark_series.values.reshape(-1, 1), (1, len(quantiles)))
        loss_benchmark = mean_pinball_loss(y_test, preds_benchmark, quantiles)
        print(f"  Official benchmark pinball loss: {loss_benchmark:.2f}")

        # --- Main model: LightGBM Quantile ---
        print("  Building features...")
        train_feat = full_featured[train_mask].copy()
        test_feat = full_featured[test_mask].loc[test_index].copy()

        # Drop rows with NaN features or target in training
        train_feat = train_feat.dropna(subset=["load"] + feature_cols)
        X_train = train_feat[feature_cols]
        y_train = train_feat["load"].values

        # Fill any remaining NaN test features with the training median
        X_test = test_feat[feature_cols].copy()
        for col in feature_cols:
            if X_test[col].isna().any():
                X_test[col] = X_test[col].fillna(X_train[col].median())

        # Validation set: last month of training, for early stopping
        val_cutoff = split["train_end"] - pd.DateOffset(months=1)
        val_mask = X_train.index >= val_cutoff
        if val_mask.sum() > 100:
            X_val, y_val = X_train[val_mask], y_train[val_mask]
            X_train_fit, y_train_fit = X_train[~val_mask], y_train[~val_mask]
        else:
            X_val, y_val = None, None
            X_train_fit, y_train_fit = X_train, y_train

        print(f"  Training LightGBM: {len(X_train_fit)} train, "
              f"{len(X_val) if X_val is not None else 0} val, "
              f"{len(X_test)} test samples")

        lgb_model = LightGBMQuantileModel(config, quantiles)
        lgb_model.fit(X_train_fit, y_train_fit, X_val, y_val, quantile_step=quantile_step)
        preds_lgb = lgb_model.predict(X_test)
        loss_lgb = mean_pinball_loss(y_test, preds_lgb, quantiles)
        print(f"  LightGBM pinball loss: {loss_lgb:.2f}")

        # --- Diebold-Mariano tests ---
        dm_naive_vs_clim = diebold_mariano_test(y_test, preds_naive, preds_clim, quantiles, "two-sided")
        dm_clim_vs_benchmark = diebold_mariano_test(y_test, preds_clim, preds_benchmark, quantiles, "two-sided")
        dm_lgb_vs_clim = diebold_mariano_test(y_test, preds_lgb, preds_clim, quantiles, "less")
        dm_lgb_vs_benchmark = diebold_mariano_test(y_test, preds_lgb, preds_benchmark, quantiles, "less")

        fold_results.append({
            "fold": fold,
            "train_end": str(split["train_end"]),
            "test_start": str(split["test_start"]),
            "test_end": str(split["test_end"]),
            "n_test": len(y_test),
            "loss_seasonal_naive": loss_naive,
            "loss_climatology": loss_clim,
            "loss_benchmark": loss_benchmark,
            "loss_lgb": loss_lgb,
            "dm_naive_vs_clim_stat": dm_naive_vs_clim["test_statistic"],
            "dm_naive_vs_clim_pval": dm_naive_vs_clim["p_value"],
            "dm_clim_vs_benchmark_stat": dm_clim_vs_benchmark["test_statistic"],
            "dm_clim_vs_benchmark_pval": dm_clim_vs_benchmark["p_value"],
            "dm_lgb_vs_clim_stat": dm_lgb_vs_clim["test_statistic"],
            "dm_lgb_vs_clim_pval": dm_lgb_vs_clim["p_value"],
            "dm_lgb_vs_benchmark_stat": dm_lgb_vs_benchmark["test_statistic"],
            "dm_lgb_vs_benchmark_pval": dm_lgb_vs_benchmark["p_value"],
        })

        shap_vals, shap_base_value = lgb_model.shap_values(X_test)
        last_fold_predictions = {
            "y_true": y_test,
            "preds_lgb": preds_lgb,
            "test_index": test_index,
            "X_test": X_test,
            "shap_values": shap_vals,
            "shap_base_value": shap_base_value,
        }

    # --- Aggregate results ---
    results_df = pd.DataFrame(fold_results)

    summary = {
        "n_folds": len(results_df),
        "seasonal_naive": {
            "mean_pinball": results_df["loss_seasonal_naive"].mean(),
            "std_pinball": results_df["loss_seasonal_naive"].std(),
        },
        "climatology": {
            "mean_pinball": results_df["loss_climatology"].mean(),
            "std_pinball": results_df["loss_climatology"].std(),
        },
        "benchmark": {
            "mean_pinball": results_df["loss_benchmark"].mean(),
            "std_pinball": results_df["loss_benchmark"].std(),
        },
        "lightgbm": {
            "mean_pinball": results_df["loss_lgb"].mean(),
            "std_pinball": results_df["loss_lgb"].std(),
        },
        "dm_naive_vs_clim_significant": (results_df["dm_naive_vs_clim_pval"] < 0.05).mean(),
        "dm_clim_vs_benchmark_significant": (results_df["dm_clim_vs_benchmark_pval"] < 0.05).mean(),
        "dm_lgb_vs_clim_significant": (results_df["dm_lgb_vs_clim_pval"] < 0.05).mean(),
        "dm_lgb_vs_benchmark_significant": (results_df["dm_lgb_vs_benchmark_pval"] < 0.05).mean(),
    }

    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)
    print(f"Folds: {summary['n_folds']}")
    print(f"Seasonal Naive:  {summary['seasonal_naive']['mean_pinball']:.2f} "
          f"+/- {summary['seasonal_naive']['std_pinball']:.2f}")
    print(f"Climatology:     {summary['climatology']['mean_pinball']:.2f} "
          f"+/- {summary['climatology']['std_pinball']:.2f}")
    print(f"Official benchmark: {summary['benchmark']['mean_pinball']:.2f} "
          f"+/- {summary['benchmark']['std_pinball']:.2f}")
    print(f"LightGBM:        {summary['lightgbm']['mean_pinball']:.2f} "
          f"+/- {summary['lightgbm']['std_pinball']:.2f}")
    print(f"DM test significant (naive vs climatology): "
          f"{summary['dm_naive_vs_clim_significant']*100:.0f}% of folds")
    print(f"DM test significant (climatology vs benchmark): "
          f"{summary['dm_clim_vs_benchmark_significant']*100:.0f}% of folds")
    print(f"DM test significant (LightGBM better than climatology): "
          f"{summary['dm_lgb_vs_clim_significant']*100:.0f}% of folds")
    print(f"DM test significant (LightGBM better than benchmark): "
          f"{summary['dm_lgb_vs_benchmark_significant']*100:.0f}% of folds")

    return {
        "fold_results": results_df,
        "summary": summary,
        "last_fold_predictions": last_fold_predictions,
    }
