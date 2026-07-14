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
from src.baselines import SeasonalNaiveBaseline, ClimatologyBaseline
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
) -> dict:
    """Run rolling-origin evaluation of the seasonal-naive and climatology baselines.

    Parameters
    ----------
    df : Prepared dataset (load + temperature, datetime index).
    config : Experiment configuration.

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

    fold_results = []

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

        # --- Diebold-Mariano tests ---
        dm_naive_vs_clim = diebold_mariano_test(y_test, preds_naive, preds_clim, quantiles, "two-sided")
        dm_clim_vs_benchmark = diebold_mariano_test(y_test, preds_clim, preds_benchmark, quantiles, "two-sided")

        fold_results.append({
            "fold": fold,
            "train_end": str(split["train_end"]),
            "test_start": str(split["test_start"]),
            "test_end": str(split["test_end"]),
            "n_test": len(y_test),
            "loss_seasonal_naive": loss_naive,
            "loss_climatology": loss_clim,
            "loss_benchmark": loss_benchmark,
            "dm_naive_vs_clim_stat": dm_naive_vs_clim["test_statistic"],
            "dm_naive_vs_clim_pval": dm_naive_vs_clim["p_value"],
            "dm_clim_vs_benchmark_stat": dm_clim_vs_benchmark["test_statistic"],
            "dm_clim_vs_benchmark_pval": dm_clim_vs_benchmark["p_value"],
        })

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
        "dm_naive_vs_clim_significant": (results_df["dm_naive_vs_clim_pval"] < 0.05).mean(),
        "dm_clim_vs_benchmark_significant": (results_df["dm_clim_vs_benchmark_pval"] < 0.05).mean(),
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
    print(f"DM test significant (naive vs climatology): "
          f"{summary['dm_naive_vs_clim_significant']*100:.0f}% of folds")
    print(f"DM test significant (climatology vs benchmark): "
          f"{summary['dm_clim_vs_benchmark_significant']*100:.0f}% of folds")

    return {
        "fold_results": results_df,
        "summary": summary,
    }
