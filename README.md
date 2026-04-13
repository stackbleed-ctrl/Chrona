# Chrona ⏱️

**Multimodal Probabilistic Time-Series Foundation Model**

[![CI](https://github.com/chrona-ai/chrona/actions/workflows/ci.yml/badge.svg)](https://github.com/chrona-ai/chrona/actions)
[![PyPI](https://img.shields.io/pypi/v/chrona)](https://pypi.org/project/chrona)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)

---

> Forecast, simulate, and reason over time — with context, uncertainty, and real-world signals built in.

Traditional forecasting models are blind. They ignore events, struggle with multiple signals, and break in real-world systems.

**Chrona** is a hybrid Transformer + Mamba foundation model that handles multivariate series, external covariates, event signals, and probabilistic outputs out of the box — zero retraining required.

---

## Features

| Capability | Chrona | Classical models |
|---|---|---|
| Multivariate + hierarchical | ✅ | ⚠️ |
| Text / event conditioning | ✅ | ❌ |
| Probabilistic (P10/P50/P90) | ✅ | ⚠️ |
| 4k+ context length | ✅ | ❌ |
| In-context learning | ✅ | ❌ |
| ONNX / edge export | ✅ | ⚠️ |

---

## Quick Start

```bash
pip install chrona
```

### Local inference

```python
from chrona import ChronaPredictor
import numpy as np

predictor = ChronaPredictor.from_scratch()   # or .from_pretrained("checkpoints/best.pt")

series = np.sin(np.linspace(0, 4*np.pi, 200)) + np.random.normal(0, 0.1, 200)
result = predictor.predict(series, horizon=48)

print(result.to_dataframe().head())
```

### API client

```python
from chrona import Chrona

client = Chrona(api_key="ck_...")   # or set CHRONA_API_KEY env var
result = client.forecast(
    series=data,
    horizon=48,
    events=["Black Friday", "Rate Hike"],
    quantiles=[0.1, 0.5, 0.9],
)
print(result.to_dataframe())
```

---

## Run the API server

```bash
# Install
pip install "chrona[all]"

# Start
uvicorn chrona.api.main:app --reload --port 8000

# Try it
curl http://localhost:8000/docs
```

### Key endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/forecast` | Probabilistic multi-step forecast |
| POST | `/simulate` | What-if scenario analysis |
| POST | `/anomaly` | Anomaly detection |
| POST | `/embed` | Text → conditioning embedding |
| GET  | `/forecast/stream` | SSE rolling forecast |

---

## Train on your own data

```python
from chrona.models.hybrid_model import ModelConfig
from chrona.data.loaders import load_csv, synthetic_dataset
from chrona.training.train import Trainer

cfg = ModelConfig(input_dim=1, model_dim=256, num_layers=8, horizon=48)
dataset = load_csv("data/energy.csv", target_col="load", timestamp_col="timestamp")

trainer = Trainer(cfg, dataset, epochs=50, batch_size=64)
trainer.fit()
```

---

## Export to ONNX (edge / production)

```python
from chrona.deployment.onnx_export import export_onnx, onnx_inference
from chrona import ChronaModel

model = ChronaModel.base()
export_onnx(model, "chrona.onnx")

# Inference (2-5x faster than PyTorch CPU)
result = onnx_inference("chrona.onnx", series=my_data)
```

---

## Docker

```bash
docker build -t chrona-api .
docker run -p 8000:8000 -e CHRONA_CHECKPOINT=checkpoints/best.pt chrona-api
```

---

## Architecture

```
Input (multivariate + covariates + events + text)
         ↓
  Multimodal Encoder
         ↓
  Hybrid Backbone (8 layers)
  ┌─────────────────────────┐
  │  Transformer Block (even) │  ← global cross-series attention
  │  Mamba Block (odd)        │  ← efficient long-range SSM
  └─────────────────────────┘
         ↓
  Probabilistic Head (MDN)
         ↓
  P10 / P50 / P90 + full distribution
```

---

## Repository Structure

```
chrona/
├── src/chrona/
│   ├── models/hybrid_model.py     # Transformer + Mamba core
│   ├── training/{train,losses}.py # Training + probabilistic loss
│   ├── inference/predict.py       # Forecast, simulate, anomaly
│   ├── data/loaders.py            # Dataset + preprocessing
│   ├── api/main.py                # FastAPI server
│   ├── deployment/onnx_export.py  # ONNX + quantization
│   └── sdk/python/client.py       # Python SDK
├── tests/test_all.py
├── configs/{model,training}.yaml
├── Dockerfile
└── pyproject.toml
```

---

## License

Apache 2.0 — free for research and commercial use.

---

## Citation

```bibtex
@software{chrona2025,
  title  = {Chrona: Multimodal Probabilistic Time-Series Foundation Model},
  year   = {2025},
  url    = {https://github.com/chrona-ai/chrona}
}
```
