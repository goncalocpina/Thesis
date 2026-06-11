"""
data_loader.py
==============
Load, validate and align the time-series input data required by the
optimization model.

Expected CSV format for each file
----------------------------------
Single numeric column (no header required, or header in the first row that
will be auto-detected).  The column must contain exactly N_HOURS rows
(default 8 760 for one non-leap year).

Profiles
--------
* PV capacity factor (CF)  – fraction of installed capacity available each hour
  Range: [0, 1]
* Wind capacity factor (CF) – same as above
* H2 demand                 – MWh of H2 required per hour

If the CSV files are not available, ``generate_synthetic_data()`` returns
realistic synthetic profiles for development and testing purposes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from parameters import N_HOURS


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def load_profiles(
    pv_path: str | Path | None = None,
    wind_path: str | Path | None = None,
    demand_path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """
    Load PV, wind and H2 demand profiles from CSV files.

    If a path is ``None`` or the file does not exist, a synthetic profile is
    generated and a warning is printed.

    Parameters
    ----------
    pv_path : str or Path, optional
        Path to PV capacity-factor CSV (values in [0, 1]).
    wind_path : str or Path, optional
        Path to wind capacity-factor CSV (values in [0, 1]).
    demand_path : str or Path, optional
        Path to hourly H2 demand CSV (MWh / hour).

    Returns
    -------
    dict with keys ``'pv'``, ``'wind'``, ``'demand'`` – each a 1-D
    ``numpy.ndarray`` of length ``N_HOURS``.
    """
    pv = _load_or_synthetic(pv_path, "pv")
    wind = _load_or_synthetic(wind_path, "wind")
    demand = _load_or_synthetic(demand_path, "demand")

    _validate(pv, "PV capacity factor", lb=0.0, ub=1.0)
    _validate(wind, "Wind capacity factor", lb=0.0, ub=1.0)
    _validate(demand, "H2 demand", lb=0.0)

    return {"pv": pv, "wind": wind, "demand": demand}


def generate_synthetic_data(seed: int = 42) -> dict[str, np.ndarray]:
    """
    Generate realistic synthetic hourly profiles for a full year.

    Profiles are deterministic given the seed so results are reproducible.

    Returns
    -------
    dict with keys ``'pv'``, ``'wind'``, ``'demand'``.
    """
    rng = np.random.default_rng(seed)
    hours = np.arange(N_HOURS)

    # ---- PV: daytime sinusoid with seasonal envelope + small noise ----------
    hour_of_day = hours % 24
    day_of_year = hours // 24
    # Seasonal factor: peak in summer (day 172), trough in winter
    seasonal = 0.5 + 0.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    # Daytime sinusoid (non-negative): zero at night, peak around noon
    daytime = np.maximum(0, np.sin(np.pi * (hour_of_day - 6) / 12))
    pv = seasonal * daytime
    pv += rng.normal(0, 0.02, N_HOURS)  # small weather noise
    pv = np.clip(pv, 0, 1)

    # ---- Wind: Weibull-shaped with seasonal autocorrelation ----------------
    # Seasonal envelope: stronger in winter
    wind_seasonal = 0.55 + 0.20 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
    wind_raw = rng.weibull(2.0, N_HOURS) * wind_seasonal * 0.45
    # Low-pass filter to create hour-to-hour autocorrelation
    alpha = 0.85
    wind = np.empty(N_HOURS)
    wind[0] = wind_raw[0]
    for t in range(1, N_HOURS):
        wind[t] = alpha * wind[t - 1] + (1 - alpha) * wind_raw[t]
    wind = np.clip(wind, 0, 1)

    # ---- H2 demand: flat industrial demand with weekly/daily pattern -------
    base_demand = 10.0  # MWh / hour average
    # Slightly higher on weekdays
    day_of_week = (hours // 24) % 7
    weekday_factor = np.where(day_of_week < 5, 1.05, 0.90)
    # Small daily variation (lower at night)
    daily_var = 1.0 + 0.10 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
    demand = base_demand * weekday_factor * daily_var
    demand += rng.normal(0, 0.5, N_HOURS)  # small stochastic variation
    demand = np.maximum(0, demand)

    return {"pv": pv, "wind": wind, "demand": demand}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_csv_series(path: str | Path) -> np.ndarray:
    """Read the first numeric column from a CSV and return as 1-D array."""
    df = pd.read_csv(path, header=None)
    # Try to detect if first row is a text header
    try:
        float(df.iloc[0, 0])
        data = df.iloc[:, 0].astype(float).values
    except (ValueError, TypeError):
        data = df.iloc[1:, 0].astype(float).values
    return data


def _load_or_synthetic(
    path: str | Path | None,
    key: str,
) -> np.ndarray:
    """Load from file if available, else use synthetic data."""
    if path is not None:
        p = Path(path)
        if p.exists():
            arr = _load_csv_series(p)
            arr = _align_to_n_hours(arr, key)
            return arr
        else:
            print(f"[data_loader] WARNING: file not found: {p}. Using synthetic data.")
    else:
        print(f"[data_loader] INFO: no path provided for '{key}'. Using synthetic data.")
    return generate_synthetic_data()[key]


def _align_to_n_hours(arr: np.ndarray, name: str) -> np.ndarray:
    """Truncate or repeat to exactly N_HOURS samples."""
    n = len(arr)
    if n == N_HOURS:
        return arr
    if n > N_HOURS:
        print(
            f"[data_loader] WARNING: '{name}' has {n} rows; "
            f"truncating to {N_HOURS}."
        )
        return arr[:N_HOURS]
    # Repeat (tile) if shorter
    reps = int(np.ceil(N_HOURS / n))
    arr_tiled = np.tile(arr, reps)[:N_HOURS]
    print(
        f"[data_loader] WARNING: '{name}' has {n} rows; "
        f"tiled to {N_HOURS}."
    )
    return arr_tiled


def _validate(
    arr: np.ndarray,
    name: str,
    lb: float | None = None,
    ub: float | None = None,
) -> None:
    """Raise ValueError if array contains NaN/Inf or out-of-bound values."""
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        raise ValueError(f"[data_loader] '{name}' contains NaN or Inf values.")
    if lb is not None and np.any(arr < lb - 1e-9):
        raise ValueError(
            f"[data_loader] '{name}' has values below {lb} "
            f"(min found: {arr.min():.4f})."
        )
    if ub is not None and np.any(arr > ub + 1e-9):
        raise ValueError(
            f"[data_loader] '{name}' has values above {ub} "
            f"(max found: {arr.max():.4f})."
        )
