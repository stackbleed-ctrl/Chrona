"""
Chrona Test Suite
pytest tests/ -v
"""
import math
import torch
import numpy as np
import pytest

from chrona.models.hybrid_model import ChronaModel, ModelConfig
from chrona.training.losses import ChronaLoss, pinball_loss, crps_loss
from chrona.inference.predict import ChronaPredictor
from chrona.data.loaders import ChronaDataset, synthetic_dataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_cfg():
    return ModelConfig(input_dim=1, model_dim=32, num_layers=2, num_heads=2, horizon=10)


@pytest.fixture
def model(small_cfg):
    return ChronaModel(small_cfg)


@pytest.fixture
def predictor(model):
    return ChronaPredictor(model, device="cpu")


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestModel:
    def test_forward_shape(self, model, small_cfg):
        B, T, D = 2, 64, 1
        x = torch.randn(B, T, D)
        out = model(x)
        assert out["mean"].shape   == (B, small_cfg.horizon)
        assert out["std"].shape    == (B, small_cfg.horizon)
        assert out["quantiles"].shape[0] == B
        assert out["quantiles"].shape[1] == small_cfg.horizon

    def test_std_positive(self, model):
        x = torch.randn(1, 64, 1)
        out = model(x)
        assert (out["std"] > 0).all(), "std must be positive"

    def test_multivariate(self, small_cfg):
        cfg = ModelConfig(input_dim=3, model_dim=32, num_layers=2, num_heads=2, horizon=5)
        m = ChronaModel(cfg)
        x = torch.randn(2, 64, 3)
        out = m(x)
        assert out["mean"].shape == (2, 5)

    def test_variants_exist(self):
        assert ChronaModel.small().num_params() < ChronaModel.base().num_params()

    def test_no_nan_forward(self, model):
        x = torch.randn(4, 128, 1)
        out = model(x)
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                assert not torch.isnan(v).any(), f"NaN in {k}"


# ---------------------------------------------------------------------------
# Loss tests
# ---------------------------------------------------------------------------

class TestLoss:
    def test_chrona_loss_keys(self, model, small_cfg):
        x = torch.randn(2, 64, 1)
        y = torch.randn(2, small_cfg.horizon)
        out = model(x)
        loss_fn = ChronaLoss()
        losses = loss_fn(out, y)
        assert {"total", "nll", "pinball", "crps"} <= losses.keys()

    def test_loss_finite(self, model, small_cfg):
        x = torch.randn(4, 64, 1)
        y = torch.randn(4, small_cfg.horizon)
        out = model(x)
        loss = ChronaLoss()(out, y)
        assert math.isfinite(loss["total"].item())

    def test_pinball_shape(self):
        preds = torch.randn(2, 10, 9)
        targets = torch.randn(2, 10)
        qs = torch.linspace(0.05, 0.95, 9)
        loss = pinball_loss(preds, targets, qs)
        assert loss.ndim == 0


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestDataset:
    def test_synthetic_dataset_len(self):
        ds = synthetic_dataset(T=500, context_len=64, horizon=16)
        assert len(ds) > 0

    def test_item_shapes(self):
        ds = synthetic_dataset(T=500, context_len=64, horizon=16)
        item = ds[0]
        assert item["ts"].shape == (64, 3)
        assert item["targets"].shape == (16,)

    def test_time_features_present(self):
        ds = synthetic_dataset(T=500, context_len=64, horizon=16)
        assert "time_features" in ds[0]
        assert ds[0]["time_features"].shape == (64, 4)


# ---------------------------------------------------------------------------
# Predictor tests
# ---------------------------------------------------------------------------

class TestPredictor:
    def test_predict_returns_result(self, predictor):
        series = np.random.randn(100)
        result = predictor.predict(series, horizon=10)
        assert len(result.mean) == 10
        assert len(result.p10()) == 10
        assert len(result.p90()) == 10

    def test_p10_lt_p90(self, predictor):
        series = np.random.randn(100)
        result = predictor.predict(series, horizon=10)
        assert (result.p10() < result.p90()).all(), "p10 must be < p90"

    def test_to_dataframe(self, predictor):
        series = np.random.randn(100)
        result = predictor.predict(series, horizon=5)
        df = result.to_dataframe()
        assert "mean" in df.columns
        assert len(df) == 5

    def test_simulate_delta(self, predictor):
        series = list(np.random.randn(80))
        result = predictor.simulate(series, [{"type": "scale", "factor": 2.0}], horizon=5)
        assert "delta_mean" in result
        assert len(result["delta_mean"]) == 5
