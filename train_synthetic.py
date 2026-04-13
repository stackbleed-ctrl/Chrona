"""
examples/train_synthetic.py
Train Chrona on synthetic multi-series data from scratch.

    python examples/train_synthetic.py
"""

from chrona.models.hybrid_model import ModelConfig
from chrona.data.loaders import synthetic_dataset
from chrona.training.train import Trainer

cfg = ModelConfig(
    input_dim=3,
    model_dim=128,
    num_layers=4,
    num_heads=4,
    horizon=24,
)

dataset = synthetic_dataset(T=5000, context_len=128, horizon=24, num_series=3)

trainer = Trainer(
    cfg=cfg,
    dataset=dataset,
    output_dir="checkpoints",
    epochs=20,
    batch_size=32,
    max_lr=3e-4,
)

trainer.fit()
print("\nDone. Checkpoint saved to checkpoints/best.pt")
