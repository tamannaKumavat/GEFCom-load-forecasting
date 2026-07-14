"""Entry point: rolling-origin evaluation of the seasonal-naive and climatology baselines."""

import argparse
import json
from pathlib import Path

from src.config import load_config, get_quantiles
from src.data import load_all_tasks
from src.evaluation import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="GEFCom2014 Load Forecasting")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to configuration file")
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = config["data"]["raw_dir"]

    print("=" * 60)
    print("GEFCom2014 Probabilistic Load Forecasting - Baselines")
    print("=" * 60)
    print(f"Config: {args.config}")
    quantiles = get_quantiles(config)
    print(f"Quantiles: {len(quantiles)} ({quantiles[0]:.2f} to {quantiles[-1]:.2f})")
    print()

    print("Loading data...")
    df = load_all_tasks(data_dir)

    results = run_evaluation(df, config)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    results["fold_results"].to_csv(results_dir / "fold_results.csv", index=False)
    with open(results_dir / "summary.json", "w") as f:
        json.dump(results["summary"], f, indent=2, default=str)

    print(f"\nResults saved to {results_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
