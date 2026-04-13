"""
Chrona Python SDK
pip install chrona
"""

import requests
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass


@dataclass
class ForecastResult:
    mean: List[float]
    p10: List[float]
    p50: List[float]
    p90: List[float]
    latency_ms: float

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({"mean": self.mean, "p10": self.p10, "p50": self.p50, "p90": self.p90})


class Chrona:
    """
    Chrona SDK client.

    Usage
    -----
    client = Chrona(api_key="ck_...")
    result = client.forecast(series=[10,12,15,14,18], horizon=24)
    print(result.to_dataframe())
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.chrona.ai/v1",
        timeout: int = 30,
    ):
        import os
        self.api_key  = api_key or os.environ.get("CHRONA_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.session  = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
            "X-SDK-Version": "1.0.0",
        })
        self.timeout = timeout

    def _post(self, endpoint: str, payload: dict) -> dict:
        r = self.session.post(f"{self.base_url}{endpoint}", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def forecast(
        self,
        series: Union[List, np.ndarray, pd.Series, pd.DataFrame],
        horizon: int = 48,
        covariates: Optional[Dict[str, List]] = None,
        events: Optional[List[str]] = None,
        quantiles: List[float] = [0.1, 0.5, 0.9],
    ) -> ForecastResult:
        if isinstance(series, (pd.Series, pd.DataFrame)):
            series = series.values.tolist()
        elif isinstance(series, np.ndarray):
            series = series.tolist()
        # Ensure list-of-rows format
        if series and not isinstance(series[0], list):
            series = [[v] for v in series]

        payload = {
            "series": series,
            "horizon": horizon,
            "quantiles": quantiles,
        }
        if covariates:
            payload["covariates"] = covariates
        if events:
            payload["events"] = [{"name": e} for e in events]

        data = self._post("/forecast", payload)
        fc = data["forecast"]
        return ForecastResult(
            mean=fc.get("mean", []),
            p10=fc.get("p10", []),
            p50=fc.get("p50", []),
            p90=fc.get("p90", []),
            latency_ms=data.get("metadata", {}).get("latency_ms", 0),
        )

    def simulate(
        self,
        series: List[float],
        interventions: List[Dict[str, Any]],
        horizon: int = 72,
    ) -> dict:
        return self._post("/simulate", {
            "base_series": series,
            "interventions": interventions,
            "horizon": horizon,
        })

    def detect_anomalies(self, series: List[float], sensitivity: float = 0.95) -> pd.DataFrame:
        data = self._post("/anomaly", {"series": series, "sensitivity": sensitivity})
        return pd.DataFrame(data.get("anomalies", []))

    def embed(self, text: str) -> List[float]:
        return self._post("/embed", {"text": text})["embedding"]
