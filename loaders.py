"""
Chrona Data Loaders
Handles: CSV / Parquet / NumPy input, normalization, windowing,
time-feature extraction, covariate alignment.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, List, Union
from sklearn.preprocessing import StandardScaler


def extract_time_features(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Returns (T, 4): [hour/24, dow/7, doy/365, month/12] normalised to [-1,1]."""
    feats = np.stack([
        timestamps.hour / 23.0 * 2 - 1,
        timestamps.dayofweek / 6.0 * 2 - 1,
        timestamps.dayofyear / 365.0 * 2 - 1,
        (timestamps.month - 1) / 11.0 * 2 - 1,
    ], axis=-1).astype(np.float32)
    return feats


class ChronaDataset(Dataset):
    """
    Sliding-window dataset for (input_sequence → forecast_horizon) training.

    Parameters
    ----------
    series      : (T, D) array-like — target time series
    context_len : length of input window
    horizon     : number of future steps to predict
    covariates  : optional (T, C) array — external regressors
    timestamps  : optional DatetimeIndex — enables time features
    stride      : step between consecutive windows (default 1)
    """

    def __init__(
        self,
        series: Union[np.ndarray, pd.DataFrame],
        context_len: int = 512,
        horizon: int = 48,
        covariates: Optional[np.ndarray] = None,
        timestamps: Optional[pd.DatetimeIndex] = None,
        stride: int = 1,
        normalize: bool = True,
    ):
        if isinstance(series, pd.DataFrame):
            if timestamps is None and isinstance(series.index, pd.DatetimeIndex):
                timestamps = series.index
            series = series.values

        series = np.array(series, dtype=np.float32)
        if series.ndim == 1:
            series = series[:, None]

        self.context_len = context_len
        self.horizon = horizon
        self.stride = stride

        # Fit scaler on training portion (all data here; caller should split before)
        self.scaler = StandardScaler()
        if normalize:
            series = self.scaler.fit_transform(series)
        else:
            self.scaler.fit(series)   # still fit so inverse_transform works

        self.series = series
        self.covariates = covariates
        self.time_features = extract_time_features(timestamps) if timestamps is not None else None

        total = len(series)
        self.indices = list(range(0, total - context_len - horizon + 1, stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        end   = start + self.context_len
        t_end = end + self.horizon

        ts      = torch.tensor(self.series[start:end],  dtype=torch.float32)
        targets = torch.tensor(self.series[end:t_end, 0], dtype=torch.float32)  # first channel as target

        item = {"ts": ts, "targets": targets}

        if self.covariates is not None:
            item["covariates"] = torch.tensor(self.covariates[start:end], dtype=torch.float32)

        if self.time_features is not None:
            item["time_features"] = torch.tensor(self.time_features[start:end], dtype=torch.float32)

        return item

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """Un-normalise predictions back to original scale."""
        dummy = np.zeros((len(values), self.series.shape[1]))
        dummy[:, 0] = values
        return self.scaler.inverse_transform(dummy)[:, 0]


# ---------------------------------------------------------------------------
# Convenience loaders
# ---------------------------------------------------------------------------

def load_csv(
    path: Union[str, Path],
    target_col: str,
    covariate_cols: Optional[List[str]] = None,
    timestamp_col: Optional[str] = None,
    **dataset_kwargs,
) -> ChronaDataset:
    df = pd.read_csv(path, parse_dates=[timestamp_col] if timestamp_col else False)
    ts = df[[target_col]]
    cov = df[covariate_cols].values.astype(np.float32) if covariate_cols else None
    idx = pd.DatetimeIndex(df[timestamp_col]) if timestamp_col else None
    return ChronaDataset(ts, covariates=cov, timestamps=idx, **dataset_kwargs)


def synthetic_dataset(
    T: int = 5000,
    context_len: int = 256,
    horizon: int = 48,
    num_series: int = 3,
    seed: int = 42,
) -> ChronaDataset:
    """Generate a synthetic multi-series dataset for smoke-testing."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, T)
    series = np.stack([
        10 * np.sin(t + rng.uniform(0, np.pi)) + rng.normal(0, 0.5, T) + np.linspace(0, 5, T)
        for _ in range(num_series)
    ], axis=1).astype(np.float32)
    timestamps = pd.date_range("2020-01-01", periods=T, freq="H")
    return ChronaDataset(series, context_len=context_len, horizon=horizon, timestamps=timestamps)
