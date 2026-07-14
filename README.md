# GEFCom Load Forecasting

Load forecasting on the GEFCom2014 electricity load track.

## Setup

```
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

## Data

Place the GEFCom2014 Load track under `data/`, so that it looks like:

```
data/
  Load/
    Task 1/L1-train.csv
    Task 2/L2-train.csv
    ...
```

## Usage

```
python run.py
```

Loads and concatenates all 15 task files (plus the Task 15 solution) into a
single continuous hourly series with load and temperature columns.

## Exploratory analysis

[notebooks/data_analyis.ipynb](notebooks/data_analyis.ipynb) covers missing
data, seasonality, the load-temperature relationship, trend, stationarity,
and autocorrelation, and motivates the feature choices in
`configs/default.yaml`.

## Evaluation methodology

**Rolling-origin (expanding-window) backtest.** GEFCom2014 released this
data in 15 rounds: round 1 is a full historical archive, and each
subsequent round adds one more month. We mirror that structure directly
(`src/evaluation.py`): fold *k* trains on every hour available through round
*k*, then forecasts round *k+1*'s entire month. This gives 15 folds total.

We use this instead of a single train/test split or a random split for two
reasons: (1) a random split would let the model train on data that is
chronologically *after* some of its test data — effectively letting it see
the future, which no real forecasting system could do; (2) repeating the
backtest across 15 different origins (rather than trusting one arbitrarily
chosen split) is what lets us tell whether a model's advantage is a
consistent, real effect or a one-off result on an easy/hard month (see
"Statistical comparison" below).

**No leakage into features.** All lagged/rolling load features are capped to
be `>= 744h` (the longest possible forecast month, ~31 days). A shorter lag
(e.g. 1 week) would, for most of the forecast month, resolve to a value that
is itself still inside the unobserved test period — see the discussion and
worked example in `src/features.py`'s module docstring. This is why the
final feature set only keeps a 1-year load lag rather than the more
obviously useful weekly/monthly lags.

## Assumptions and limitations

**Temperature is treated as perfectly known at forecast time.** The model
uses the actual observed `temp_mean` for the forecast month, not a weather
forecast. In real deployment you would only have a temperature *forecast*
(with its own error) at the time you make a load forecast, so this backtest
likely overstates real-world accuracy to some degree — the model is
implicitly assuming perfect foresight of weather. We did not quantify this
gap (e.g. by injecting synthetic forecast error into temperature and
re-running the backtest), which would be a natural next step if a tighter
estimate of real-world performance were needed.

**LightGBM's quantile objective fits one model per quantile level** (there
is no native multi-quantile output). Fitting all 99 levels per fold is
expensive, so `--quantile-step` (default 5) fits every Nth level directly and
linearly interpolates the rest, with predictions sorted per row afterward to
guarantee valid (non-crossing) quantiles.

## Statistical comparison

Beyond raw pinball loss, every fold's result is compared pairwise with a
Diebold-Mariano test (`src/metrics.py`), which asks whether a loss
difference is a real, consistent effect rather than noise from one
particular month. We report the fraction of the 15 folds where each
comparison is significant (p < 0.05), not just a single aggregate number.

## Results

Run `python run.py` to reproduce; current numbers (99 quantiles, 15 folds,
`--quantile-step 5`):

| Method | Mean pinball loss |
|---|---|
| Seasonal-naive | 13.00 +/- 4.35 |
| Climatology | 8.33 +/- 4.55 |
| Official GEFCom2014 benchmark | 15.14 +/- 7.60 |
| LightGBM | 3.88 +/- 1.51 |

LightGBM's improvement over both climatology and the official benchmark is
statistically significant (Diebold-Mariano, p < 0.05) in 100% of the 15
folds. See `results/summary.json` for full per-fold numbers, and
`results/shap_summary.png` for which features drive the model's predictions.
