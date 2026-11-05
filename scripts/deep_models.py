"""CPU PyTorch day-ahead models. No target price enters the decoder inputs.

TFT follows Lim et al. (https://arxiv.org/abs/1912.09363): variable selection,
gated residual networks, local LSTM encoder/decoder and shared-value interpretable
causal attention. Single-market data has no static covariates; static encoders
are omitted. Point forecasts use the median quantile (L1), not prediction bands.
"""
import math

import numpy as np
import pandas as pd
import torch
from torch import nn


CALENDAR_FEATURES = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
                     "year_sin", "year_cos", "holiday", "rest_day"]
PAST_FEATURES = ["price"] + CALENDAR_FEATURES
FUTURE_FEATURES = CALENDAR_FEATURES + ["price_d1", "price_d7", "mean7", "std7"]


def calendar_features(day, calendar):
    h = np.arange(1, 25)
    return np.column_stack([
        np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
        np.full(24, np.sin(2 * np.pi * day.dayofweek / 7)),
        np.full(24, np.cos(2 * np.pi * day.dayofweek / 7)),
        np.full(24, np.sin(2 * np.pi * (day.dayofyear - 1) / 365.25)),
        np.full(24, np.cos(2 * np.pi * (day.dayofyear - 1) / 365.25)),
        np.full(24, float(calendar.loc[day, "是否节假日"] == "是")),
        np.full(24, float(calendar.loc[day, "是否周末休息日"] == "是"))])


def build_sequences(prices, calendar, history_days=7):
    if history_days != 7:
        raise ValueError("This experiment fixes history_days=7 for baseline alignment")
    past, future, targets, days = [], [], [], []
    for day in prices.index:
        history = pd.date_range(day - pd.Timedelta(days=history_days), periods=history_days)
        if not history.isin(prices.index).all() or not history.isin(calendar.index).all():
            continue
        values = prices.loc[history].to_numpy(dtype=np.float32)
        target = prices.loc[day].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all() or not np.isfinite(target).all():
            continue
        hist_calendar = np.concatenate([calendar_features(d, calendar) for d in history])
        past.append(np.column_stack([values.ravel(), hist_calendar]))
        future.append(np.column_stack([calendar_features(day, calendar), values[-1],
                                       values[0], values.mean(axis=0), values.std(axis=0)]))
        targets.append(target)
        days.append(day)
    return (pd.DatetimeIndex(days), np.asarray(past, dtype=np.float32),
            np.asarray(future, dtype=np.float32), np.asarray(targets, dtype=np.float32))


def fit_scaler(past, future, target):
    """Called on fitting targets only; reuse unchanged for later predictions."""
    return {"past_mean": past.mean(axis=(0, 1)),
            "past_std": np.maximum(past.std(axis=(0, 1)), 1e-6),
            "future_mean": future.mean(axis=(0, 1)),
            "future_std": np.maximum(future.std(axis=(0, 1)), 1e-6),
            "target_mean": float(target.mean()),
            "target_std": max(float(target.std()), 1e-6)}


def transform(past, future, scaler):
    return (torch.from_numpy((past - scaler["past_mean"]) / scaler["past_std"]),
            torch.from_numpy((future - scaler["future_mean"]) / scaler["future_std"]))


class GateAddNorm(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.gate = nn.Sequential(nn.Dropout(dropout), nn.Linear(size, 2 * size), nn.GLU())
        self.norm = nn.LayerNorm(size)

    def forward(self, x, skip):
        return self.norm(skip + self.gate(x))


class GRN(nn.Module):
    def __init__(self, input_size, size, dropout):
        super().__init__()
        self.skip = nn.Linear(input_size, size) if input_size != size else nn.Identity()
        self.net = nn.Sequential(nn.Linear(input_size, size), nn.ELU(), nn.Linear(size, size))
        self.output = GateAddNorm(size, dropout)

    def forward(self, x):
        return self.output(self.net(x), self.skip(x))


class VariableSelection(nn.Module):
    def __init__(self, count, size, dropout):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Linear(1, size) for _ in range(count)])
        self.transforms = nn.ModuleList([GRN(size, size, dropout) for _ in range(count)])
        self.weights = nn.Sequential(GRN(count * size, size, dropout), nn.Linear(size, count))

    def forward(self, x):
        embedded = [layer(x[..., i:i+1]) for i, layer in enumerate(self.embeddings)]
        weights = self.weights(torch.cat(embedded, dim=-1)).softmax(dim=-1)
        values = torch.stack([grn(e) for grn, e in zip(self.transforms, embedded)], dim=-2)
        return (weights.unsqueeze(-1) * values).sum(dim=-2), weights


class LSTM(nn.Module):
    def __init__(self, past_size, future_size, size=16, dropout=.1, **kwargs):
        super().__init__()
        self.encoder = nn.LSTM(past_size, size, batch_first=True)
        self.decoder = nn.LSTM(future_size, size, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(size, 1))

    def forward(self, past, future):
        _, state = self.encoder(past)
        decoded, _ = self.decoder(future, state)
        return self.head(decoded).squeeze(-1)


class InterpretableAttention(nn.Module):
    """Head-specific Q/K, shared V, head averaging as in TFT (not concatenation)."""
    def __init__(self, size, heads, dropout):
        super().__init__()
        if size % heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        self.heads, self.dim = heads, size // heads
        self.query = nn.Linear(size, size, bias=False)
        self.key = nn.Linear(size, size, bias=False)
        self.value = nn.Linear(size, self.dim, bias=False)
        self.output = nn.Linear(self.dim, size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence, history_length):
        batch, length, _ = sequence.shape
        q = self.query(sequence[:, history_length:]).reshape(batch, -1, self.heads, self.dim).transpose(1, 2)
        k = self.key(sequence).reshape(batch, length, self.heads, self.dim).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.dim)
        mask = torch.arange(length)[None, :] > torch.arange(history_length, length)[:, None]
        weights = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
        combined = self.dropout(weights).mean(dim=1) @ self.value(sequence)
        return self.output(combined), weights.mean(dim=1)


class TFT(nn.Module):
    def __init__(self, past_size, future_size, size=16, dropout=.1, heads=2):
        super().__init__()
        self.past_selection = VariableSelection(past_size, size, dropout)
        self.future_selection = VariableSelection(future_size, size, dropout)
        self.encoder = nn.LSTM(size, size, batch_first=True)
        self.decoder = nn.LSTM(size, size, batch_first=True)
        self.local_gate = GateAddNorm(size, dropout)
        self.enrichment = GRN(size, size, dropout)
        self.attention = InterpretableAttention(size, heads, dropout)
        self.attention_gate = GateAddNorm(size, dropout)
        self.feed_forward = GRN(size, size, dropout)
        self.output_gate = GateAddNorm(size, dropout)
        self.head = nn.Linear(size, 1)

    def forward(self, past, future, interpret=False):
        p, pw = self.past_selection(past)
        f, fw = self.future_selection(future)
        encoded, state = self.encoder(p)
        decoded, _ = self.decoder(f, state)
        local = self.local_gate(torch.cat([encoded, decoded], dim=1), torch.cat([p, f], dim=1))
        enriched = self.enrichment(local)
        attended, attention = self.attention(enriched, past.shape[1])
        fused = self.attention_gate(attended, enriched[:, past.shape[1]:])
        output = self.head(self.output_gate(self.feed_forward(fused), local[:, past.shape[1]:])).squeeze(-1)
        if interpret:
            return output, {"past_weights": pw, "future_weights": fw, "attention": attention}
        return output


def make_model(name, config):
    cls = {"LSTM": LSTM, "TFT": TFT}[name]
    return cls(len(PAST_FEATURES), len(FUTURE_FEATURES), size=config["hidden_size"],
               dropout=config["dropout"], heads=config["attention_heads"]).cpu()
