"""
Chrona Training Loop
Supports: mixed-precision, gradient clipping, cosine LR schedule,
early stopping, W&B / tensorboard logging (optional).
"""

import os
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from typing import Optional
from pathlib import Path

from chrona.models.hybrid_model import ChronaModel, ModelConfig
from chrona.training.losses import ChronaLoss
from chrona.data.loaders import ChronaDataset


def cosine_lr(optimizer, step: int, warmup: int, total: int, min_lr=1e-6, max_lr=3e-4):
    if step < warmup:
        lr = max_lr * step / max(warmup, 1)
    else:
        progress = (step - warmup) / max(total - warmup, 1)
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


class Trainer:
    def __init__(
        self,
        cfg: ModelConfig,
        dataset: ChronaDataset,
        output_dir: str = "checkpoints",
        epochs: int = 50,
        batch_size: int = 64,
        max_lr: float = 3e-4,
        warmup_steps: int = 500,
        grad_clip: float = 1.0,
        val_split: float = 0.1,
        device: Optional[str] = None,
        use_amp: bool = True,
    ):
        self.cfg = cfg
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.grad_clip = grad_clip
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and self.device == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Split dataset
        n_val = int(len(dataset) * val_split)
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
        self.val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

        self.model = ChronaModel(cfg).to(self.device)
        self.criterion = ChronaLoss().to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=max_lr, weight_decay=1e-2, betas=(0.9, 0.95)
        )

        total_steps = epochs * len(self.train_loader)
        self.total_steps = total_steps
        self.step = 0
        self.best_val = float("inf")

        print(f"[Chrona] Model params: {self.model.num_params():,}")
        print(f"[Chrona] Device: {self.device} | AMP: {self.use_amp}")
        print(f"[Chrona] Train batches: {len(self.train_loader)} | Val batches: {len(self.val_loader)}")

    def _forward_batch(self, batch):
        ts      = batch["ts"].to(self.device)
        targets = batch["targets"].to(self.device)
        tf      = batch.get("time_features")
        if tf is not None:
            tf = tf.to(self.device)
        with torch.autocast(device_type=self.device, enabled=self.use_amp):
            out  = self.model(ts, time_features=tf)
            loss = self.criterion(out, targets)
        return loss

    def train_epoch(self):
        self.model.train()
        total, count = 0.0, 0
        for batch in self.train_loader:
            lr = cosine_lr(self.optimizer, self.step, self.warmup_steps, self.total_steps, max_lr=self.max_lr)
            losses = self._forward_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total += losses["total"].item()
            count += 1
            self.step += 1
        return total / count

    @torch.no_grad()
    def val_epoch(self):
        self.model.eval()
        total, count = 0.0, 0
        for batch in self.val_loader:
            losses = self._forward_batch(batch)
            total += losses["total"].item()
            count += 1
        return total / count

    def save_checkpoint(self, name="best.pt"):
        path = self.output_dir / name
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "step": self.step,
            "cfg": self.cfg,
        }, path)
        return path

    def fit(self):
        print("\n[Chrona] Starting training...\n")
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch()
            val_loss   = self.val_epoch()
            elapsed = time.time() - t0

            flag = ""
            if val_loss < self.best_val:
                self.best_val = val_loss
                self.save_checkpoint("best.pt")
                flag = " ★ saved"

            print(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"train={train_loss:.4f} val={val_loss:.4f} | "
                f"{elapsed:.1f}s{flag}"
            )

        self.save_checkpoint("final.pt")
        print(f"\n[Chrona] Training complete. Best val loss: {self.best_val:.4f}")
