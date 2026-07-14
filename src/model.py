"""LightGBM quantile regression: one model per quantile level, interpolated in between."""

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from typing import Any, Optional


class LightGBMQuantileModel:
    def __init__(self, config: dict[str, Any], quantiles: np.ndarray):
        self.config = config
        self.quantiles = np.asarray(quantiles, dtype=float)
        self.models: dict[int, lgb.LGBMRegressor] = {}

    def _params(self) -> dict[str, Any]:
        model_cfg = self.config.get("model", {})
        return {
            "n_estimators": model_cfg.get("n_estimators", 500),
            "learning_rate": model_cfg.get("learning_rate", 0.05),
            "num_leaves": model_cfg.get("num_leaves", 63),
            "max_depth": model_cfg.get("max_depth", -1),
            "min_child_samples": model_cfg.get("min_child_samples", 20),
            "subsample": model_cfg.get("subsample", 0.8),
            "colsample_bytree": model_cfg.get("colsample_bytree", 0.8),
            "verbosity": -1,
        }

    def _median_model(self) -> lgb.LGBMRegressor:
        fitted_idx = np.array(sorted(self.models.keys()))
        closest = fitted_idx[np.argmin(np.abs(self.quantiles[fitted_idx] - 0.5))]
        return self.models[closest]

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        quantile_step: int = 1,
    ) -> "LightGBMQuantileModel":
        early_stopping_rounds = self.config.get("training", {}).get("early_stopping_rounds", 50)

        indices = set(range(0, len(self.quantiles), quantile_step))
        indices.add(len(self.quantiles) - 1)  # always fit the top quantile too

        self.models = {}
        for idx in sorted(indices):
            tau = self.quantiles[idx]
            model = lgb.LGBMRegressor(objective="quantile", alpha=tau, **self._params())

            fit_kwargs = {}
            if X_val is not None and y_val is not None:
                fit_kwargs["eval_set"] = [(X_val, y_val)]
                fit_kwargs["callbacks"] = [lgb.early_stopping(early_stopping_rounds, verbose=False)]

            model.fit(X_train, y_train, **fit_kwargs)
            self.models[idx] = model

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        n_q = len(self.quantiles)
        fitted_idx = np.array(sorted(self.models.keys()))
        fitted_preds = np.column_stack([self.models[idx].predict(X) for idx in fitted_idx])

        out = np.empty((len(X), n_q))
        for j in range(n_q):
            pos = np.searchsorted(fitted_idx, j)
            if fitted_idx[pos] == j:
                out[:, j] = fitted_preds[:, pos]
                continue
            lo, hi = fitted_idx[pos - 1], fitted_idx[pos]
            frac = (j - lo) / (hi - lo)
            out[:, j] = fitted_preds[:, pos - 1] + frac * (fitted_preds[:, pos] - fitted_preds[:, pos - 1])

        return np.sort(out, axis=1)  # avoid quantile crossing

    def shap_values(self, X: pd.DataFrame) -> tuple[np.ndarray, float]:
        # explains the median model only -- doing this for all ~20 quantile models would be
        # expensive and redundant, since the same features drive every quantile's prediction
        model = self._median_model()
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X)
        return values, explainer.expected_value
