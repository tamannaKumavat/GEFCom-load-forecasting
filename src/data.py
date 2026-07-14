"""Data loading for GEFCom2014 load-track data."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

N_TASKS = 15
WEATHER_COLS = [f"w{i}" for i in range(1, 26)]


def _resolve_date(
    date_digits: str, prev_date: Optional[pd.Timestamp]
) -> pd.Timestamp:
    # date digits before the year are ambiguous (e.g. "1012010" = Jan 1 or Oct 1, 2010)
    # try every valid month/day split, pick whichever is closest to prev_date
    year = int(date_digits[-4:])
    prefix = date_digits[:-4]

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
    return min(candidates, key=lambda d: abs((d - prev_date).days))


def _parse_timestamps(
    ts_series: pd.Series, prev_date: Optional[pd.Timestamp]
) -> tuple[pd.Series, Optional[pd.Timestamp]]:
    parts = ts_series.str.extract(r"^(?P<date>\d+)\s+(?P<hour>\d{1,2}):00$")
    if parts["date"].isna().any():
        bad = ts_series[parts["date"].isna()].head(3).tolist()
        raise ValueError(f"Unrecognised TIMESTAMP format: {bad}")

    date_digits = parts["date"]
    hours = parts["hour"].astype(int)

    # date repeats for 24 rows (one per hour), so resolve once per day not per row
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


def _read_task_train(
    data_dir: Path, task: int, prev_date: Optional[pd.Timestamp]
) -> tuple[pd.DataFrame, Optional[pd.Timestamp]]:
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
    """Official benchmark forecast for the month after `task` (all 99 quantile cols are identical)."""
    path = Path(data_dir) / "Load" / f"Task {task}" / f"L{task}-benchmark.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")

    raw = pd.read_csv(path)
    # prev_date disambiguates the same way as train files -- pass the fold's train_end
    datetimes, _ = _parse_timestamps(raw["TIMESTAMP"], prev_date=prev_date)
    return pd.Series(raw["0.5"].values, index=datetimes, name="benchmark")


def _read_solution(data_dir: Path) -> pd.DataFrame:
    # Task 15 solution file uses slash-separated dates, no ambiguity here
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


def load_all_tasks(
    data_dir: str,
    n_tasks: int = N_TASKS,
    include_solution: bool = True,
) -> pd.DataFrame:
    """Load and concatenate all 15 task files (+ solution) into one hourly series."""
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
    """Rolling-origin folds: task k trains through Lk, tests on task k+1 (or the solution for k=15)."""
    data_path = Path(data_dir)

    task_ranges = {}
    prev_date = None
    for i in range(1, n_tasks + 1):
        path = data_path / "Load" / f"Task {i}" / f"L{i}-train.csv"
        if not path.exists():
            continue
        raw = pd.read_csv(path)
        datetimes, prev_date = _parse_timestamps(raw["TIMESTAMP"], prev_date)
        task_ranges[i] = {"start": datetimes.min(), "end": datetimes.max()}

    sol = _read_solution(data_path)
    sol_range = None
    if len(sol) > 0:
        sol_range = {"start": sol["datetime"].min(), "end": sol["datetime"].max()}

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

    if n_tasks in task_ranges and sol_range is not None:
        splits.append({
            "task": n_tasks,
            "train_end": task_ranges[n_tasks]["end"],
            "test_start": sol_range["start"],
            "test_end": sol_range["end"],
        })

    return splits
