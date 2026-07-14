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
