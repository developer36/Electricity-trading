"""Differentiable, single-block Timestral adaptation with isolated daily tasks."""
import math

import numpy as np
import torch
from torch import nn
from timestral.forecaster import TimestralForecaster
from timestral.loading import load_model

try:
    from .foundation_models import make_input
except ImportError:
    from foundation_models import make_input


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.factor = alpha / rank

    def forward(self, x):
        return self.base(x) + nn.functional.linear(nn.functional.linear(x, self.A), self.B) * self.factor


def inject_lora(model, rank=8, alpha=16):
    replaced = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and name.rsplit('.', 1)[-1] in ('wQKV', 'wO'):
            parent, child = name.rsplit('.', 1)
            setattr(model.get_submodule(parent), child, LoRALinear(module, rank, alpha))
            replaced.append(name)
    if not replaced:
        raise ValueError("No Timestral attention projections found")
    return replaced


class AdaptedTimestral(nn.Module):
    def __init__(self, checkpoint, mode, rank=8, alpha=16):
        super().__init__()
        self.network = load_model(checkpoint, device='cpu')
        self.network.requires_grad_(False)
        self.mode = mode
        if mode == 'Head':
            self.network.head.requires_grad_(True)
        elif mode == 'LoRA':
            self.lora_modules = inject_lora(self.network, rank, alpha)
        elif mode != 'ZeroShot':
            raise ValueError(mode)
        # Same differentiable raw path as the official forecaster; no decorators,
        # no nonnegative clipping, no extreme-value cap, and one 32-step patch.
        self.forecaster = TimestralForecaster(self.network, quantile_real_cap_k=0)
        self.levels = tuple(self.network.config.quantile_knots)

    def forward(self, context, known):
        observed = torch.isfinite(context)
        q = self.forecaster._forecast_raw(context, observed, 24, known_covariates=known)
        mirrored = self.forecaster._forecast_raw(-context, observed, 24, known_covariates=-known)
        return (q - mirrored.flip(-1))[:, 0, :, :] * .5

    def trainable_state(self):
        return {n: p.detach().cpu().clone() for n, p in self.named_parameters() if p.requires_grad}

    def load_trainable(self, state):
        params = dict(self.named_parameters())
        if set(state) != {n for n, p in params.items() if p.requires_grad}:
            raise ValueError('Trainable parameter keys differ')
        with torch.no_grad():
            for n, value in state.items():
                params[n].copy_(value)


def pinball(prediction, target, levels):
    if prediction.shape[:2] != target.shape or prediction.shape[-1] != len(levels):
        raise ValueError('Pinball shape mismatch')
    error = target[..., None] - prediction
    q = prediction.new_tensor(levels)
    return torch.maximum(q * error, (q-1) * error).mean()


def build_dataset(prices, calendar, days):
    samples = [make_input(prices, calendar, day) for day in days]
    return (torch.from_numpy(np.stack([s.target for s in samples])),
            torch.from_numpy(np.stack([s.known for s in samples])),
            torch.from_numpy(prices.loc[days].to_numpy(dtype=np.float32)))


def predict(model, context, known, batch_size=8):
    model.eval()
    with torch.no_grad():
        result = torch.cat([model(context[i:i+batch_size], known[i:i+batch_size])
                            for i in range(0, len(context), batch_size)]).cpu().numpy()
    if not np.isfinite(result).all() or np.any(np.diff(result, axis=-1) < -1e-4):
        raise ValueError('Invalid quantile predictions')
    return result
