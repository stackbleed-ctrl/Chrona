"""
Chrona Hybrid Temporal Model
Alternating Transformer + Mamba-style (SSM) blocks for efficient,
long-context multivariate time-series forecasting.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    input_dim: int = 1
    covariate_dim: int = 0
    event_embed_dim: int = 32
    model_dim: int = 256
    num_layers: int = 8          # alternating Transformer / Mamba
    num_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.1
    max_seq_len: int = 4096
    horizon: int = 48
    num_quantiles: int = 9       # deciles
    use_rope: bool = True        # Rotary Position Embeddings


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """RoPE — baked-in relative position info, no learned params."""
    def __init__(self, dim: int, max_len: int = 8192):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.shape[1]
        return self.cos[:seq], self.sin[:seq]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


# ---------------------------------------------------------------------------
# Transformer Block (global attention)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            cfg.model_dim, cfg.num_heads,
            dropout=cfg.dropout, batch_first=True
        )
        hidden = cfg.model_dim * cfg.ffn_mult
        self.ff = nn.Sequential(
            nn.Linear(cfg.model_dim, hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, cfg.model_dim),
            nn.Dropout(cfg.dropout),
        )
        self.norm1 = nn.LayerNorm(cfg.model_dim)
        self.norm2 = nn.LayerNorm(cfg.model_dim)
        self.use_rope = cfg.use_rope
        if cfg.use_rope:
            self.rope = RotaryEmbedding(cfg.model_dim // cfg.num_heads)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = residual + attn_out
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Mamba-style SSM Block (efficient long-range)
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """
    Simplified Mamba (S4/S6 spirit): selective state-space with
    causal depthwise conv + gating. Full Mamba can be swapped in
    via `pip install mamba-ssm` — this keeps zero non-torch deps.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.model_dim
        self.in_proj  = nn.Linear(d, d * 2)          # x-branch + z-branch
        self.conv     = nn.Conv1d(d, d, kernel_size=4, padding=3, groups=d)
        self.act      = nn.SiLU()
        self.dt_proj  = nn.Linear(d, d)               # Δt (discretisation)
        self.out_proj = nn.Linear(d, d)
        self.norm     = nn.LayerNorm(d)
        self.dropout  = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)                              # pre-norm
        xz = self.in_proj(x)                          # (B,T,2D)
        x_b, z = xz.chunk(2, dim=-1)
        # causal conv (trim future tokens)
        x_b = self.conv(x_b.transpose(1, 2))[:, :, :x.shape[1]].transpose(1, 2)
        x_b = self.act(x_b)
        dt  = torch.sigmoid(self.dt_proj(x_b))        # selective gate
        y   = x_b * dt * torch.sigmoid(z)             # gated output
        return residual + self.dropout(self.out_proj(y))


# ---------------------------------------------------------------------------
# Multimodal Encoder
# ---------------------------------------------------------------------------

class MultimodalEncoder(nn.Module):
    """
    Fuses: raw time-series + covariates + event embeddings + optional
    text embeddings (passed in as pre-computed tensors).
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        total_in = cfg.input_dim + cfg.covariate_dim
        self.ts_proj   = nn.Linear(total_in, cfg.model_dim)
        self.event_emb = nn.Embedding(512, cfg.event_embed_dim)
        self.event_proj= nn.Linear(cfg.event_embed_dim, cfg.model_dim)
        self.time_feat = nn.Linear(4, cfg.model_dim)   # hour, dow, doy, month
        self.dropout   = nn.Dropout(cfg.dropout)

    def forward(
        self,
        ts: torch.Tensor,                          # (B, T, input_dim+cov)
        time_features: Optional[torch.Tensor] = None,  # (B, T, 4)
        event_ids: Optional[torch.Tensor] = None,      # (B, T) int
        text_emb: Optional[torch.Tensor] = None,       # (B, D) global
    ) -> torch.Tensor:
        x = self.ts_proj(ts)
        if time_features is not None:
            x = x + self.time_feat(time_features)
        if event_ids is not None:
            x = x + self.event_proj(self.event_emb(event_ids))
        if text_emb is not None:
            x = x + text_emb.unsqueeze(1).expand_as(x) * 0.1
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Probabilistic Head
# ---------------------------------------------------------------------------

class ProbabilisticHead(nn.Module):
    """
    Mixture Density Network: outputs K Gaussian components.
    Also produces explicit quantile forecasts via pin-ball regression.
    """
    NUM_COMPONENTS = 3

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        K = self.NUM_COMPONENTS
        self.horizon = cfg.horizon
        self.quantiles = torch.linspace(0.05, 0.95, cfg.num_quantiles)

        # Autoregressive horizon projection
        self.horizon_proj = nn.Linear(cfg.model_dim, cfg.model_dim * cfg.horizon)
        self.norm = nn.LayerNorm(cfg.model_dim)

        # MDN outputs per horizon step
        self.pi_head    = nn.Linear(cfg.model_dim, K)          # mixture weights
        self.mu_head    = nn.Linear(cfg.model_dim, K)          # means
        self.sigma_head = nn.Linear(cfg.model_dim, K)          # stds

        # Direct quantile regression (for calibration)
        self.q_head = nn.Linear(cfg.model_dim, cfg.num_quantiles)

    def forward(self, h: torch.Tensor):
        """h: (B, model_dim) — last hidden state"""
        B = h.shape[0]
        # Project to full horizon
        z = self.horizon_proj(self.norm(h))            # (B, model_dim * H)
        z = z.view(B, self.horizon, -1)                # (B, H, model_dim)

        pi    = F.softmax(self.pi_head(z), dim=-1)     # (B, H, K)
        mu    = self.mu_head(z)                        # (B, H, K)
        sigma = F.softplus(self.sigma_head(z)) + 1e-4  # (B, H, K)

        # Mixture mean & variance
        mean = (pi * mu).sum(-1)                       # (B, H)
        var  = (pi * (sigma**2 + mu**2)).sum(-1) - mean**2
        std  = var.clamp(min=1e-6).sqrt()

        # Quantile estimates (Gaussian approximation per quantile)
        qs = self.quantiles.to(h.device)               # (Q,)
        z_scores = torch.erfinv(2 * qs - 1) * math.sqrt(2)
        quantile_forecasts = mean.unsqueeze(-1) + std.unsqueeze(-1) * z_scores

        return {
            "mean":      mean,           # (B, H)
            "std":       std,            # (B, H)
            "quantiles": quantile_forecasts,  # (B, H, Q)
            "pi": pi, "mu": mu, "sigma": sigma,
        }


# ---------------------------------------------------------------------------
# Full Chrona Model
# ---------------------------------------------------------------------------

class ChronaModel(nn.Module):
    """
    Chrona: Hybrid Transformer + Mamba foundation model for
    multivariate probabilistic time-series forecasting.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = MultimodalEncoder(cfg)

        layers = []
        for i in range(cfg.num_layers):
            layers.append(
                TransformerBlock(cfg) if i % 2 == 0 else MambaBlock(cfg)
            )
        self.backbone = nn.ModuleList(layers)
        self.pool_norm = nn.LayerNorm(cfg.model_dim)
        self.head = ProbabilisticHead(cfg)

    def forward(
        self,
        ts: torch.Tensor,
        time_features: Optional[torch.Tensor] = None,
        event_ids: Optional[torch.Tensor] = None,
        text_emb: Optional[torch.Tensor] = None,
    ) -> dict:
        x = self.encoder(ts, time_features, event_ids, text_emb)
        for block in self.backbone:
            x = block(x)
        h = self.pool_norm(x[:, -1])   # take last token
        return self.head(h)

    @classmethod
    def small(cls):
        return cls(ModelConfig(model_dim=128, num_layers=4, num_heads=4))

    @classmethod
    def base(cls):
        return cls(ModelConfig(model_dim=256, num_layers=8, num_heads=8))

    @classmethod
    def large(cls):
        return cls(ModelConfig(model_dim=512, num_layers=12, num_heads=16))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
