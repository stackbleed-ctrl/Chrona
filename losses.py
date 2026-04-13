"""
Chrona Loss Functions
Pinball / quantile loss, NLL for MDN, CRPS for calibration.
"""
import torch
import torch.nn as nn
import math


def pinball_loss(preds: torch.Tensor, targets: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """
    preds:     (B, H, Q)
    targets:   (B, H)
    quantiles: (Q,)
    """
    targets = targets.unsqueeze(-1).expand_as(preds)
    q = quantiles.to(preds.device).unsqueeze(0).unsqueeze(0)
    errors = targets - preds
    loss = torch.max(q * errors, (q - 1) * errors)
    return loss.mean()


def gaussian_nll(mean: torch.Tensor, std: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    dist = torch.distributions.Normal(mean, std.clamp(min=1e-4))
    return -dist.log_prob(targets).mean()


def mdn_nll(pi, mu, sigma, targets):
    """Negative log likelihood for Mixture Density Network."""
    targets = targets.unsqueeze(-1).expand_as(mu)
    log_probs = torch.distributions.Normal(mu, sigma).log_prob(targets)
    log_mix   = torch.log(pi.clamp(min=1e-8)) + log_probs
    return -torch.logsumexp(log_mix, dim=-1).mean()


def crps_loss(mean: torch.Tensor, std: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Continuous Ranked Probability Score — proper scoring rule."""
    z = (targets - mean) / std.clamp(min=1e-4)
    phi = torch.distributions.Normal(0, 1).log_prob(z).exp()
    Phi = torch.distributions.Normal(0, 1).cdf(z)
    crps = std * (z * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))
    return crps.mean()


class ChronaLoss(nn.Module):
    """Combined loss: MDN NLL + pinball + CRPS."""
    def __init__(self, quantiles=None, w_nll=1.0, w_pin=0.5, w_crps=0.5):
        super().__init__()
        if quantiles is None:
            quantiles = torch.linspace(0.05, 0.95, 9)
        self.register_buffer("quantiles", quantiles)
        self.w_nll, self.w_pin, self.w_crps = w_nll, w_pin, w_crps

    def forward(self, output: dict, targets: torch.Tensor) -> dict:
        nll  = mdn_nll(output["pi"], output["mu"], output["sigma"], targets)
        pin  = pinball_loss(output["quantiles"], targets, self.quantiles)
        crps = crps_loss(output["mean"], output["std"], targets)
        total = self.w_nll * nll + self.w_pin * pin + self.w_crps * crps
        return {"total": total, "nll": nll, "pinball": pin, "crps": crps}
