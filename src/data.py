""" Data loading and preprocessing for GEFCom2014 load-track data. """

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
 
 # Constants
N_TASKS = 15
WEATHER_COLS = [f"w{i}" for i in range(1, 26)]
 
 
# Timestamp parsing
def _resolve_date(
    date_digits: str, prev_date: Optional[pd.Timestamp]
) -> pd.Timestamp:
    """Turn the date portion of a TIMESTAMP into a pd.Timestamp.
 
    Tries every valid month/day split of the digits before the 4-digit year.
    If multiple splits are valid calendar dates, picks the one closest to
    ``prev_date`` (rows are chronological, so the nearest date wins).
    """
    year = int(date_digits[-4:])
    prefix = date_digits[:-4]  # month + day, 2–4 chars
 
    candidates = []
    for split in range(1, len(prefix)):
        m, d = int(prefix[:split]), int(prefix[split:])
        try:
            candidates.append(pd.Timestamp(year=year, month=m, day=d))
        except ValueError:
            continue
 
    if not candidates:
        raise ValueError(f"Cannot parse date from '{date_digits}'")
    if len(candidates) == 1 or prev_date is None:
        return candidates[0]
    # Pick the candidate closest to the previous date
    return min(candidates, key=lambda d: abs((d - prev_date).days))
 
 
def _parse_timestamps(
    ts_series: pd.Series, prev_date: Optional[pd.Timestamp]
) -> tuple[pd.Series, Optional[pd.Timestamp]]:
    """Parse a full TIMESTAMP column into datetimes.
 
    Returns (datetime_series, last_resolved_date) so that callers can
    thread the disambiguation context across consecutive files.
    """
    # Split each timestamp into date digits and hour
    parts = ts_series.str.extract(r"^(?P<date>\d+)\s+(?P<hour>\d{1,2}):00$")
    if parts["date"].isna().any():
        bad = ts_series[parts["date"].isna()].head(3).tolist()
        raise ValueError(f"Unrecognised TIMESTAMP format: {bad}")
 
    date_digits = parts["date"]
    hours = parts["hour"].astype(int)
 
    # The date part repeats for 24 consecutive rows (one per hour),
    # so resolve ambiguity once per day, not once per row.
    group_id = (date_digits != date_digits.shift()).cumsum()
    first_of_group = ~group_id.duplicated()
 
    resolved = {}
    for gid, digits in zip(
        group_id[first_of_group], date_digits[first_of_group]
    ):
        prev_date = _resolve_date(digits, prev_date)
        resolved[gid] = prev_date
 
    dates = group_id.map(resolved)
    datetimes = dates + pd.to_timedelta(hours, unit="h")
    return datetimes, prev_date
 
 
# Individual file readers
def _read_task_train(
    data_dir: Path, task: int, prev_date: Optional[pd.Timestamp]
) -> tuple[pd.DataFrame, Optional[pd.Timestamp]]:
    """Read one Task n/Ln-train.csv file.
 
    Returns a DataFrame with columns:
        datetime, load, w1, w2, ..., w25
    and the last resolved date for disambiguation threading.
    """
    path = data_dir / "Load" / f"Task {task}" / f"L{task}-train.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Expected the GEFCom2014 Load track under "
            f"{data_dir}/Load/Task <n>/L<n>-train.csv"
        )
 
    raw = pd.read_csv(path)
    datetimes, prev_date = _parse_timestamps(raw["TIMESTAMP"], prev_date)
 
    out = pd.DataFrame({"datetime": datetimes, "load": raw["LOAD"]})
    for col in WEATHER_COLS:
        if col in raw.columns:
            out[col] = raw[col]
 
    return out, prev_date
 
 
def load_benchmark(
    data_dir: str, task: int, prev_date: Optional[pd.Timestamp] = None
) -> pd.Series:
    """Load the official GEFCom2014 benchmark point forecast for one task.

    L<task>-benchmark.csv covers the same period as L<task+1>-train.csv,
    i.e. it's the reference forecast for the test month of the rolling-origin
    fold that trains through ``task``. All 99 quantile columns hold the same
    value per hour (it's a flat point forecast, not a real probabilistic
    spread), so this returns a single series rather than 99 redundant columns.

    ``prev_date`` disambiguates the file's month/day-ambiguous timestamps
    (e.g. "1012010" could mean Jan 1 or Oct 1, 2010) the same way
    ``load_all_tasks`` threads it across consecutive training files. Pass the
    corresponding fold's ``train_end`` so the benchmark month resolves next
    to it rather than defaulting to the first calendar-valid guess.
    """
    path = Path(data_dir) / "Load" / f"Task {task}" / f"L{task}-benchmark.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")

    raw = pd.read_csv(path)
    datetimes, _ = _parse_timestamps(raw["TIMESTAMP"], prev_date=prev_date)
    return pd.Series(raw["0.5"].values, index=datetimes, name="benchmark")


def _read_solution(data_dir: Path) -> pd.DataFrame:
    """Read the Task-15 solution (Dec 2011 actuals + temperature).
 
    This file uses slash-separated dates ('12/1/2011') so no ambiguity.
    """
    path = (
        data_dir / "Load" / "Solution to Task 15"
        / "solution15_L_temperature.csv"
    )
    if not path.exists():
        return pd.DataFrame(columns=["datetime", "load"] + WEATHER_COLS)
 
    raw = pd.read_csv(path)
    datetimes = (
        pd.to_datetime(raw["date"], format="%m/%d/%Y")
        + pd.to_timedelta(raw["hour"], unit="h")
    )
 
    out = pd.DataFrame({"datetime": datetimes, "load": raw["LOAD"]})
    for col in WEATHER_COLS:
        if col in raw.columns:
            out[col] = raw[col]
 
    return out
 
 
# Main public API
def load_all_tasks(
    data_dir: str,
    n_tasks: int = N_TASKS,
    include_solution: bool = True,
) -> pd.DataFrame:
    """Load and concatenate all task files into one continuous series.
 
    Parameters
    ----------
    data_dir : Root data directory (contains ``Load/`` subfolder).
    n_tasks : How many task files to load (1–15).
    include_solution : Whether to append Dec 2011 actuals from the
        Solution to Task 15 folder.
 
    Returns
    -------
    DataFrame with datetime index and columns:
        load, w1, w2, ..., w25, temp_mean
    """
    data_path = Path(data_dir)
    frames = []
    prev_date = None
 
    for task in range(1, n_tasks + 1):
        frame, prev_date = _read_task_train(data_path, task, prev_date)
        frames.append(frame)
        print(
            f"  Loaded Task {task:2d}: {len(frame):6d} rows  "
            f"({frame['datetime'].min()} to {frame['datetime'].max()})"
        )
 
    if include_solution:
        sol = _read_solution(data_path)
        if len(sol) > 0:
            frames.append(sol)
            print(
                f"  Loaded Solution:  {len(sol):4d} rows  "
                f"({sol['datetime'].min()} to {sol['datetime'].max()})"
            )
 
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["datetime"], keep="last")
    combined = combined.set_index("datetime").sort_index()
 
    # Add mean temperature across all 25 stations
    temp_cols = [c for c in WEATHER_COLS if c in combined.columns]
    combined["temp_mean"] = combined[temp_cols].mean(axis=1)
 
    n_total = len(combined)
    n_load = combined["load"].notna().sum()
    print(
        f"\n  Combined: {n_total} hours  "
        f"({combined.index.min()} to {combined.index.max()})"
    )
    print(
        f"  Load available: {n_load} hours  "
        f"({combined['load'].first_valid_index()} to "
        f"{combined['load'].last_valid_index()})"
    )
 
    return combined
 
 
def get_task_splits(
    data_dir: str, n_tasks: int = N_TASKS
) -> list[dict]:
    """Define the train/test boundary for each forecasting task.
 
    For task k:
        - Train on everything up to the end of Lk-train.csv
        - Test on the next month (Lk+1-train.csv, or solution for k=15)
 
    This is the rolling-origin evaluation scheme: each fold adds one
    month of training data and forecasts the following month.
 
    Returns
    -------
    List of dicts, each with keys:
        task, train_end, test_start, test_end
    """
    data_path = Path(data_dir)
 
    # Collect time range of each task file
    task_ranges = {}
    prev_date = None
    for i in range(1, n_tasks + 1):
        path = data_path / "Load" / f"Task {i}" / f"L{i}-train.csv"
        if not path.exists():
            continue
        raw = pd.read_csv(path)
        datetimes, prev_date = _parse_timestamps(raw["TIMESTAMP"], prev_date)
        task_ranges[i] = {"start": datetimes.min(), "end": datetimes.max()}
 
    # Collect solution range
    sol = _read_solution(data_path)
    sol_range = None
    if len(sol) > 0:
        sol_range = {"start": sol["datetime"].min(), "end": sol["datetime"].max()}
 
    # Build splits
    splits = []
    for k in range(1, n_tasks):
        if k not in task_ranges or (k + 1) not in task_ranges:
            continue
        splits.append({
            "task": k,
            "train_end": task_ranges[k]["end"],
            "test_start": task_ranges[k + 1]["start"],
            "test_end": task_ranges[k + 1]["end"],
        })
 
    # Task 15: test against solution
    if n_tasks in task_ranges and sol_range is not None:
        splits.append({
            "task": n_tasks,
            "train_end": task_ranges[n_tasks]["end"],
            "test_start": sol_range["start"],
            "test_end": sol_range["end"],
        })
 
    return splits