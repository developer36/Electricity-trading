"""Official foundation-model adapters; input construction never reads target-day prices."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

KNOWN_FEATURES = ("hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
                  "year_sin", "year_cos", "holiday", "rest_day")


@dataclass
class ForecastInput:
    day: pd.Timestamp
    target: np.ndarray  # (targets, context); this task has one price target
    known: np.ndarray   # (covariates, context + horizon)
    history_start: pd.Timestamp
    history_end: pd.Timestamp


def make_input(prices, calendar, day, history_days=7):
    """Calendar-day slicing preserves hour-end 24:00 and cannot shift across gaps."""
    day = pd.Timestamp(day)
    if history_days < 1:
        raise ValueError("history_days must be positive")
    history = pd.date_range(day - pd.Timedelta(days=history_days), periods=history_days)
    values = prices.reindex(history).to_numpy(dtype=np.float32).reshape(1, -1)
    if values.shape != (1, history_days * 24) or not np.isfinite(values).all():
        raise ValueError(f"Missing/invalid historical labels for {day.date()}")
    dates = history.append(pd.DatetimeIndex([day]))
    cal = calendar.reindex(dates)
    holiday = cal["是否节假日"].map({"是": 1., "否": 0.}).to_numpy()
    rest = cal["是否周末休息日"].map({"是": 1., "否": 0.}).to_numpy()
    hours = np.tile(np.arange(1, 25), len(dates))
    weekday = np.repeat(dates.dayofweek.to_numpy(), 24)
    year_day = np.repeat(dates.dayofyear.to_numpy() - 1, 24)
    known = np.stack([np.sin(2*np.pi*hours/24), np.cos(2*np.pi*hours/24),
                      np.sin(2*np.pi*weekday/7), np.cos(2*np.pi*weekday/7),
                      np.sin(2*np.pi*year_day/365.25), np.cos(2*np.pi*year_day/365.25),
                      np.repeat(holiday, 24), np.repeat(rest, 24)]).astype(np.float32)
    if not np.isfinite(known).all():
        raise ValueError("Missing calendar covariates")
    assert history[-1] < day
    return ForecastInput(day, values, known, history[0], history[-1])


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class OfficialAdapter:
    """Only official inference APIs. All adapters return (batch, 24, 3) q10/q50/q90."""
    def __init__(self, name, local_path):
        self.name = "Timestral" if name in ("Timestral-5M", "Timestral-28M", "Timestral-123M", "Timestral-209M") else name
        name = self.name
        local_path = str(Path(local_path).resolve())
        if name == "Chronos-2":
            from chronos import Chronos2Pipeline
            self.model = Chronos2Pipeline.from_pretrained(local_path, device_map="cpu")
        elif name == "Timestral":
            from timestral import TimestralForecaster
            self.model = TimestralForecaster.from_pretrained(local_path, device="cpu")
        elif name == "TiRex-2":
            from tirex2 import load_model
            self.model = load_model(local_path, device="cpu")
        else:
            raise ValueError(f"Unknown model {name}")

    def predict(self, inputs):
        import torch
        with torch.inference_mode():
            if self.name == "Chronos-2":
                tasks = []
                for item in inputs:
                    length = item.target.shape[-1]
                    tasks.append({"target": item.target,
                                  "past_covariates": {k: item.known[i, :length] for i, k in enumerate(KNOWN_FEATURES)},
                                  "future_covariates": {k: item.known[i, length:] for i, k in enumerate(KNOWN_FEATURES)}})
                quantiles, _ = self.model.predict_quantiles(tasks, prediction_length=24,
                                                          quantile_levels=[.1, .5, .9], cross_learning=False)
                result = np.stack([as_numpy(q)[0] for q in quantiles])
            elif self.name == "Timestral":
                forecast = self.model.forecast(np.stack([i.target for i in inputs]),
                                              prediction_length=24,
                                              known_covariates=np.stack([i.known for i in inputs]))
                result = np.stack([as_numpy(forecast.quantile(q))[:, 0, :] for q in (.1, .5, .9)], axis=-1)
            else:
                from tirex2 import TimeseriesType
                tasks = [TimeseriesType(target=torch.from_numpy(i.target.copy()),
                                        past_covariates=None,
                                        future_covariates=torch.from_numpy(i.known.copy())) for i in inputs]
                forecasts = self.model.forecast(tasks, prediction_length=24, output_type="numpy")
                result = np.stack([as_numpy(q)[0, [0, 4, 8], :].T for q in forecasts])
        if result.shape != (len(inputs), 24, 3) or not np.isfinite(result).all():
            raise ValueError(f"Invalid output from {self.name}: {result.shape}")
        if np.any(np.diff(result, axis=-1) < -1e-4):
            raise ValueError(f"Crossed quantiles from {self.name}")
        return result
