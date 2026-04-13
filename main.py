"""
Chrona API — Production FastAPI Server
Endpoints: /forecast  /simulate  /anomaly  /embed  /forecast/stream
"""

import os
import time
import asyncio
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import torch

from chrona.inference.predict import ChronaPredictor
from chrona.models.hybrid_model import ModelConfig

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Chrona Forecasting API",
    description="Multimodal, probabilistic time-series forecasting foundation model.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=500)

# Lazy-loaded predictor
_predictor: Optional[ChronaPredictor] = None

def get_predictor() -> ChronaPredictor:
    global _predictor
    if _predictor is None:
        ckpt = os.environ.get("CHRONA_CHECKPOINT", "")
        if ckpt and Path(ckpt).exists():
            _predictor = ChronaPredictor.from_pretrained(ckpt)
        else:
            _predictor = ChronaPredictor.from_scratch()
    return _predictor


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class EventInput(BaseModel):
    name: str
    time: Optional[str] = None
    magnitude: float = 1.0

class ForecastRequest(BaseModel):
    series: List[List[float]] = Field(..., description="(T, D) multivariate series as list-of-rows")
    timestamps: Optional[List[str]] = None
    covariates: Optional[Dict[str, List[float]]] = None
    events: Optional[List[EventInput]] = None
    horizon: int = Field(48, ge=1, le=720)
    quantiles: List[float] = Field([0.1, 0.5, 0.9])
    num_samples: int = Field(0, ge=0, le=500)

class SimulateRequest(BaseModel):
    base_series: List[float]
    interventions: List[Dict[str, Any]]
    horizon: int = Field(72, ge=1, le=720)

class AnomalyRequest(BaseModel):
    series: List[float]
    sensitivity: float = Field(0.95, ge=0.5, le=0.999)

class EmbedRequest(BaseModel):
    text: str

class ForecastResponse(BaseModel):
    forecast: Dict[str, List[float]]
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model": "chrona-1.0", "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))}


# ---------------------------------------------------------------------------
# POST /forecast
# ---------------------------------------------------------------------------

@app.post("/forecast", response_model=ForecastResponse)
async def forecast(req: ForecastRequest, predictor: ChronaPredictor = Depends(get_predictor)):
    t0 = time.perf_counter()
    try:
        arr = np.array(req.series, dtype=np.float32)
        result = predictor.predict(arr, horizon=req.horizon)
        q_map = {}
        for q in req.quantiles:
            qi = int(round((q - 0.05) / 0.10))
            qi = max(0, min(qi, result.quantiles.shape[1] - 1))
            q_map[f"p{int(q*100)}"] = result.quantiles[:, qi].tolist()

        return ForecastResponse(
            forecast={
                "mean": result.mean.tolist(),
                **q_map,
            },
            metadata={
                "horizon": req.horizon,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                "model": "chrona-1.0",
            }
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


# ---------------------------------------------------------------------------
# POST /simulate
# ---------------------------------------------------------------------------

@app.post("/simulate")
async def simulate(req: SimulateRequest, predictor: ChronaPredictor = Depends(get_predictor)):
    try:
        result = predictor.simulate(req.base_series, req.interventions, req.horizon)
        return {
            "base": {
                "mean": result["base"].mean.tolist(),
                "p10":  result["base"].p10().tolist(),
                "p90":  result["base"].p90().tolist(),
            },
            "scenario": {
                "mean": result["scenario"].mean.tolist(),
                "p10":  result["scenario"].p10().tolist(),
                "p90":  result["scenario"].p90().tolist(),
            },
            "delta_mean": result["delta_mean"].tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


# ---------------------------------------------------------------------------
# POST /anomaly
# ---------------------------------------------------------------------------

@app.post("/anomaly")
async def anomaly(req: AnomalyRequest, predictor: ChronaPredictor = Depends(get_predictor)):
    try:
        df = predictor.detect_anomalies(req.series, req.sensitivity)
        anomalies = df[df["anomaly"]].to_dict(orient="records")
        return {"anomalies": anomalies, "total_checked": len(df), "total_flagged": len(anomalies)}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


# ---------------------------------------------------------------------------
# POST /embed
# ---------------------------------------------------------------------------

@app.post("/embed")
async def embed(req: EmbedRequest):
    """
    Returns a mock embedding. Wire up to your LLM provider
    (OpenAI, Grok, local model) for real multimodal conditioning.
    """
    words = req.text.lower().split()
    seed  = sum(ord(c) for c in req.text) % 2**32
    rng   = np.random.default_rng(seed)
    vec   = rng.standard_normal(256).astype(np.float32)
    vec  /= np.linalg.norm(vec) + 1e-8
    return {"embedding": vec.tolist(), "dim": 256, "text": req.text}


# ---------------------------------------------------------------------------
# GET /forecast/stream  (Server-Sent Events)
# ---------------------------------------------------------------------------

@app.get("/forecast/stream")
async def forecast_stream(
    series: str,
    horizon: int = 10,
    predictor: ChronaPredictor = Depends(get_predictor),
):
    """
    SSE streaming endpoint. Call with ?series=1,2,3,4,5&horizon=10
    """
    arr = [float(v) for v in series.split(",")]

    async def event_generator():
        for i, result in enumerate(predictor.stream_predict(arr, horizon=horizon)):
            data = f"data: {{\"step\": {i}, \"mean\": {result.mean[0]:.4f}, \"p10\": {result.p10()[0]:.4f}, \"p90\": {result.p90()[0]:.4f}}}\n\n"
            yield data
            await asyncio.sleep(0.05)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
# uvicorn chrona.api.main:app --host 0.0.0.0 --port 8000 --reload
