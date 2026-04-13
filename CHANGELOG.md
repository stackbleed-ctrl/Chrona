# Changelog

All notable changes to Chrona will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com).

## [1.0.0] - 2025-04-13

### Added
- Hybrid Transformer + Mamba backbone (`models/hybrid_model.py`)
- Multimodal encoder: time series, covariates, events, text embeddings
- Probabilistic head: MDN + pinball + CRPS loss
- Full training loop with cosine LR, AMP, gradient clipping
- Inference engine: `predict`, `simulate`, `detect_anomalies`, `stream_predict`
- FastAPI server with `/forecast`, `/simulate`, `/anomaly`, `/embed`, `/forecast/stream`
- Python SDK client (`Chrona`)
- ONNX + TorchScript export
- Post-training quantization
- Sliding-window dataset with time feature extraction
- CI workflow (lint + test + build)
- Dockerfile (CPU + GPU variants)
- Apache 2.0 license
