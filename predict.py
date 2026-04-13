"""
Chrona Inference Engine
Zero-shot and few-shot forecasting, simulation, anomaly detection.
"""

import torch
import numpy as np
import pandas as pd
from typing import Optional, List, Union, Iterator
from dataclasses import dataclass
from pathlib import Path

from chrona.models.hybrid_model import ChronaModel, ModelConfig


@dataclass
class ForecastResult:
    timestamps: List
    mean: np.ndarray
    std: np.ndarray
    quantiles: np.ndarray      # (H, Q)
    quantile_levels: List[float]

    def p10(self): return self.quantiles[:, 0]
    def p50(self): return self.quantiles[:, 4]
    def p90(self): return self.quantiles[:, -1]

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "timestamp": self.timestamps,
            "mean": self.mean,
            "std": self.std,
            "p10": self.p10(),
            "p50": self.p50(),
            "p90": self.p90(),
        })
        for i, q in enumerate(self.quantile_levels):
            df[f"q{int(q*100)}"] = self.quantiles[:, i]
        return df


class ChronaPredictor:
    """
    High-level inference interface.

    Usage
    -----
    predictor = ChronaPredictor.from_pretrained("checkpoints/best.pt")
    result = predictor.predict(series=my_df, horizon=48)
    print(result.to_dataframe())
    """

    def __init__(self, model: ChronaModel, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.quantile_levels = list(np.linspace(0.05, 0.95, model.cfg.num_quantiles))

    @classmethod
    def from_pretrained(cls, path: Union[str, Path], device: Optional[str] = None):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg  = ckpt.get("cfg", ModelConfig())
        model = ChronaModel(cfg)
        model.load_state_dict(ckpt["model_state"])
        return cls(model, device)

    @classmethod
    def from_scratch(cls, cfg: Optional[ModelConfig] = None, device: Optional[str] = None):
        """Zero-shot (untrained) — useful for API scaffolding / demos."""
        return cls(ChronaModel(cfg or ModelConfig()), device)

    def _prepare_input(
        self,
        series: Union[np.ndarray, pd.DataFrame, List],
        context_len: Optional[int] = None,
    ) -> torch.Tensor:
        if isinstance(series, pd.DataFrame):
            arr = series.select_dtypes(include=np.number).values
        elif isinstance(series, list):
            arr = np.array(series, dtype=np.float32)
        else:
            arr = np.array(series, dtype=np.float32)

        if arr.ndim == 1:
            arr = arr[:, None]

        # Normalize with running stats
        mu, sigma = arr.mean(0), arr.std(0).clip(min=1e-6)
        self._norm_mu, self._norm_sigma = mu, sigma
        arr = (arr - mu) / sigma

        ctx = context_len or self.model.cfg.max_seq_len
        arr = arr[-ctx:]    # take last ctx steps

        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(
        self,
        series: Union[np.ndarray, pd.DataFrame, List],
        horizon: Optional[int] = None,
        covariates: Optional[np.ndarray] = None,
        events: Optional[List[str]] = None,
        num_samples: int = 0,    # if >0, use Monte-Carlo sampling
        timestamps: Optional[pd.DatetimeIndex] = None,
    ) -> ForecastResult:
        x = self._prepare_input(series)
        out = self.model(x)

        h = horizon or self.model.cfg.horizon
        mean_n = out["mean"][0, :h].cpu().numpy()
        std_n  = out["std"][0, :h].cpu().numpy()
        q_n    = out["quantiles"][0, :h].cpu().numpy()

        # Denormalize
        s0 = self._norm_sigma[0]
        m0 = self._norm_mu[0]
        mean_n = mean_n * s0 + m0
        std_n  = std_n  * s0
        q_n    = q_n    * s0 + m0

        if timestamps is not None:
            last = timestamps[-1]
            freq = pd.infer_freq(timestamps) or "H"
            fut  = pd.date_range(last, periods=h+1, freq=freq)[1:]
            ts_out = list(fut)
        else:
            n = len(series) if not isinstance(series, (np.ndarray, torch.Tensor)) else len(series)
            ts_out = list(range(n, n + h))

        return ForecastResult(
            timestamps=ts_out,
            mean=mean_n,
            std=std_n,
            quantiles=q_n,
            quantile_levels=self.quantile_levels,
        )

    @torch.no_grad()
    def stream_predict(
        self,
        series: Union[np.ndarray, List],
        horizon: int = 96,
        step: int = 1,
    ) -> Iterator[ForecastResult]:
        """Yield rolling forecasts as new data arrives."""
        arr = list(series)
        for i in range(horizon):
            result = self.predict(arr, horizon=step)
            yield result
            arr.append(result.mean[0])

    @torch.no_grad()
    def detect_anomalies(
        self,
        series: Union[np.ndarray, pd.DataFrame, List],
        sensitivity: float = 0.95,
    ) -> pd.DataFrame:
        """Flag points that fall outside the model's predictive interval."""
        if isinstance(series, list):
            arr = np.array(series)
        elif isinstance(series, pd.DataFrame):
            arr = series.select_dtypes(include=np.number).values[:, 0]
        else:
            arr = np.array(series)

        scores = []
        ctx = min(64, len(arr) // 2)
        for i in range(ctx, len(arr)):
            window = arr[i-ctx:i]
            result = self.predict(window, horizon=1)
            z = abs(arr[i] - result.mean[0]) / (result.std[0] + 1e-6)
            scores.append({"idx": i, "value": arr[i], "z_score": z,
                           "anomaly": z > np.percentile([s["z_score"] for s in scores + [{"z_score": z}]], sensitivity * 100)})
        return pd.DataFrame(scores)

    @torch.no_grad()
    def simulate(
        self,
        series: Union[np.ndarray, List],
        interventions: List[dict],
        horizon: int = 72,
    ) -> dict:
        """
        What-if simulation.
        interventions = [{"type": "scale", "factor": 1.2, "start": 10, "end": 20}, ...]
        """
        base = self.predict(series, horizon=horizon)

        modified = np.array(series, dtype=float).copy()
        for iv in interventions:
            if iv["type"] == "scale":
                s, e = iv.get("start", 0), iv.get("end", len(modified))
                modified[s:e] *= iv.get("factor", 1.0)
            elif iv["type"] == "shift":
                modified += iv.get("value", 0)

        scenario = self.predict(modified, horizon=horizon)
        return {"base": base, "scenario": scenario,
                "delta_mean": scenario.mean - base.mean}
