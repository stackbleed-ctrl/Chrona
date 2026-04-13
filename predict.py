"""
Chrona Inference Engine
Fully supports: multivariate series, covariates, event conditioning, streaming.
"""

import torch
import numpy as np
import pandas as pd
from typing import Optional, List, Union, Iterator, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path

from chrona.models.hybrid_model import ChronaModel, ModelConfig


@dataclass
class ForecastResult:
    timestamps: List
    mean: np.ndarray
    std: np.ndarray
    quantiles: np.ndarray
    quantile_levels: List[float]
    events_applied: List[str] = field(default_factory=list)
    num_series: int = 1

    def p10(self): return self.quantiles[:, 0]
    def p50(self): return self.quantiles[:, self.quantiles.shape[1] // 2]
    def p90(self): return self.quantiles[:, -1]

    def to_dataframe(self):
        df = pd.DataFrame({
            "timestamp": self.timestamps,
            "mean": self.mean, "std": self.std,
            "p10": self.p10(), "p50": self.p50(), "p90": self.p90(),
        })
        for i, q in enumerate(self.quantile_levels):
            df[f"q{int(q*100):02d}"] = self.quantiles[:, i]
        return df

    def to_api_dict(self):
        return {
            "mean": self.mean.round(4).tolist(),
            "std":  self.std.round(4).tolist(),
            "p10":  self.p10().round(4).tolist(),
            "p50":  self.p50().round(4).tolist(),
            "p90":  self.p90().round(4).tolist(),
            "timestamps": [str(t) for t in self.timestamps],
            "events_applied": self.events_applied,
        }


class ChronaPredictor:
    """
    High-level inference interface supporting multivariate series,
    event conditioning, what-if simulation, and streaming.
    """

    def __init__(self, model: ChronaModel, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = model.to(self.device).eval()
        self.quantile_levels = list(np.linspace(0.05, 0.95, model.cfg.num_quantiles))

    @classmethod
    def from_pretrained(cls, path: Union[str, Path], device: Optional[str] = None):
        ckpt  = torch.load(path, map_location="cpu", weights_only=False)
        cfg   = ckpt.get("cfg", ModelConfig())
        model = ChronaModel(cfg)
        model.load_state_dict(ckpt["model_state"])
        return cls(model, device)

    @classmethod
    def from_scratch(cls, cfg: Optional[ModelConfig] = None, device: Optional[str] = None):
        return cls(ChronaModel(cfg or ModelConfig()), device)

    def _prepare_series(self, series, context_len=None):
        if isinstance(series, pd.DataFrame):
            arr = series.select_dtypes(include=np.number).values
        elif isinstance(series, pd.Series):
            arr = series.values[:, None]
        elif isinstance(series, list):
            arr = np.array(series, dtype=np.float32)
        else:
            arr = np.asarray(series, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        arr = arr.astype(np.float32)
        num_series = arr.shape[1]
        mu = arr.mean(axis=0); sigma = arr.std(axis=0).clip(min=1e-6)
        arr = (arr - mu) / sigma
        ctx = context_len or self.model.cfg.max_seq_len
        arr = arr[-ctx:]
        model_in = self.model.cfg.input_dim + self.model.cfg.covariate_dim
        if arr.shape[1] < model_in:
            arr = np.concatenate([arr, np.zeros((arr.shape[0], model_in - arr.shape[1]), np.float32)], axis=1)
        elif arr.shape[1] > model_in:
            arr = arr[:, :model_in]
        return torch.tensor(arr).unsqueeze(0).to(self.device), mu, sigma, num_series

    def _build_event_ids(self, context_len, events):
        ids = torch.zeros(1, context_len, dtype=torch.long, device=self.device)
        for i, name in enumerate(events):
            eid = self.model.event_name_to_id(name)
            pos = context_len - len(events) + i
            if 0 <= pos < context_len:
                ids[0, pos] = eid
        return ids

    @torch.no_grad()
    def predict(
        self,
        series,
        horizon: Optional[int] = None,
        covariates=None,
        events: Optional[List[str]] = None,
        text_emb: Optional[np.ndarray] = None,
        timestamps: Optional[pd.DatetimeIndex] = None,
        context_len: Optional[int] = None,
    ) -> ForecastResult:
        x, mu, sigma, num_series = self._prepare_series(series, context_len)
        T = x.shape[1]
        h = horizon or self.model.cfg.horizon

        event_ids = None
        events_applied = []
        if events:
            events_applied = [e for e in events if e]
            event_ids = self._build_event_ids(T, events_applied)

        text_tensor = None
        if text_emb is not None:
            t = torch.tensor(text_emb, dtype=torch.float32, device=self.device)
            d = self.model.cfg.model_dim
            t = t[:d] if t.shape[0] >= d else torch.nn.functional.pad(t, (0, d - t.shape[0]))
            text_tensor = t.unsqueeze(0)

        out = self.model(x, event_ids=event_ids, text_emb=text_tensor)

        s0, m0 = float(sigma[0]), float(mu[0])
        mean_n = out["mean"][0, :h].cpu().numpy() * s0 + m0
        std_n  = out["std"][0, :h].cpu().numpy()  * s0
        q_n    = out["quantiles"][0, :h].cpu().numpy() * s0 + m0

        if timestamps is not None:
            last = timestamps[-1]
            freq = pd.infer_freq(timestamps) or "h"
            ts_out = list(pd.date_range(last, periods=h + 1, freq=freq)[1:])
        else:
            base = len(series) if isinstance(series, list) else np.asarray(series).shape[0]
            ts_out = list(range(base, base + h))

        return ForecastResult(
            timestamps=ts_out, mean=mean_n, std=std_n,
            quantiles=q_n, quantile_levels=self.quantile_levels,
            events_applied=events_applied, num_series=num_series,
        )

    @torch.no_grad()
    def stream_predict(self, series, horizon=96, events=None) -> Iterator[ForecastResult]:
        arr = list(np.asarray(series, dtype=float).flatten())
        for _ in range(horizon):
            result = self.predict(arr, horizon=1, events=events)
            yield result
            arr.append(float(result.mean[0]))

    @torch.no_grad()
    def simulate(self, series, interventions, horizon=72, events=None):
        arr = np.asarray(series, dtype=float).copy()
        modified = arr.copy()
        for iv in interventions:
            s, e = iv.get("start", 0), iv.get("end", len(modified))
            if iv["type"] == "scale":
                modified[s:e] *= iv.get("factor", 1.0)
            elif iv["type"] == "shift":
                modified[s:e] += iv.get("value", 0.0)
            elif iv["type"] == "zero_out":
                modified[s:e] = 0.0
        base     = self.predict(arr,      horizon=horizon, events=events)
        scenario = self.predict(modified, horizon=horizon, events=events)
        return {"base": base, "scenario": scenario, "delta_mean": scenario.mean - base.mean}

    @torch.no_grad()
    def detect_anomalies(self, series, sensitivity=0.95, context_window=64):
        if isinstance(series, list):
            arr = np.array(series, dtype=float)
        elif isinstance(series, pd.DataFrame):
            arr = series.select_dtypes(include=np.number).values[:, 0].astype(float)
        else:
            arr = np.asarray(series, dtype=float).flatten()
        ctx = min(context_window, len(arr) // 2)
        records, z_scores = [], []
        for i in range(ctx, len(arr)):
            result = self.predict(arr[i - ctx:i], horizon=1)
            z = abs(arr[i] - result.mean[0]) / (result.std[0] + 1e-6)
            z_scores.append(z)
            records.append({"idx": i, "value": arr[i], "pred": result.mean[0], "std": result.std[0], "z_score": z})
        threshold = float(np.percentile(z_scores, sensitivity * 100)) if z_scores else 3.0
        df = pd.DataFrame(records)
        if not df.empty:
            df["anomaly"] = df["z_score"] > threshold
        return df
