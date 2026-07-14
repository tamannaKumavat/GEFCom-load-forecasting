"""Entry point: rolling-origin evaluation of the baselines and LightGBM."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import load_config, get_quantiles
from src.data import load_all_tasks
from src.evaluation import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="GEFCom2014 Load Forecasting")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to configuration file")
    parser.add_argument("--quantile-step", type=int, default=5,
                        help="Fit every N-th quantile for LightGBM (1=all 99, 5=every 5th)")
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = config["data"]["raw_dir"]

    print("=" * 60)
    print("GEFCom2014 Probabilistic Load Forecasting")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Quantile step: {args.quantile_step}")
    quantiles = get_quantiles(config)
    print(f"Quantiles: {len(quantiles)} ({quantiles[0]:.2f} to {quantiles[-1]:.2f})")
    print()

    print("Loading data...")
    df = load_all_tasks(data_dir)

    results = run_evaluation(df, config, quantile_step=args.quantile_step)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    results["fold_results"].to_csv(results_dir / "fold_results.csv", index=False)
    with open(results_dir / "summary.json", "w") as f:
        json.dump(results["summary"], f, indent=2, default=str)

    lf = results["last_fold_predictions"]
    if lf is not None:
        X_test = lf["X_test"]
        shap_vals = lf["shap_values"]

        # Aggregate: mean |SHAP| per feature -- which features move
        # predictions the most, on average, across the last fold's test month.
        shap_importance = (
            pd.DataFrame({
                "feature": X_test.columns,
                "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
            })
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        shap_importance.to_csv(results_dir / "shap_importance.csv", index=False)

        # Per-hour: exactly how much each feature pushed each individual
        # prediction up or down relative to the model's average prediction.
        shap_detail = pd.DataFrame(shap_vals, columns=X_test.columns, index=X_test.index)
        shap_detail.insert(0, "base_value", lf["shap_base_value"])
        shap_detail.to_csv(results_dir / "shap_values.csv")

        # Beeswarm summary plot: per-feature SHAP distribution across every
        # test-month hour, colored by that hour's feature value (red=high,
        # blue=low), so both importance and direction of effect are visible.
        shap.summary_plot(shap_vals, X_test, show=False)
        plt.tight_layout()
        plt.savefig(results_dir / "shap_summary.png", dpi=150)
        plt.close()

        # Bar chart: same mean |SHAP| ranking as shap_importance.csv, plotted.
        shap.summary_plot(shap_vals, X_test, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(results_dir / "shap_importance_bar.png", dpi=150)
        plt.close()

    print(f"\nResults saved to {results_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
