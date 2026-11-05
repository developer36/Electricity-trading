"""Availability boundaries, causal attention and checkpoint reproducibility."""
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from scripts import deep_models as d
from scripts import run_deep_models as r


def synthetic():
    days = pd.date_range("2024-01-01", periods=90)
    prices = pd.DataFrame(np.arange(90 * 24, dtype=float).reshape(90, 24),
                          index=days, columns=range(1, 25))
    calendar = pd.DataFrame({"是否节假日": "否", "是否周末休息日": "否"}, index=days)
    return prices, calendar


class DeepModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = json.loads((r.ROOT / "configs/deep_models.json").read_text())

    def test_inputs_exclude_target_and_future_prices(self):
        prices, calendar = synthetic()
        days, past, future, _ = d.build_sequences(prices, calendar)
        day = days[20]
        idx = days.get_loc(day)
        self.assertEqual(past.shape[1:], (168, len(d.PAST_FEATURES)))
        np.testing.assert_allclose(future[idx, :, 8], prices.loc[day - pd.Timedelta(days=1)])
        prices.loc[day:] += 99999
        _, p2, f2, _ = d.build_sequences(prices, calendar)
        np.testing.assert_array_equal(past[idx], p2[idx])
        np.testing.assert_array_equal(future[idx], f2[idx])

    def test_missing_day_or_label_invalidates_full_history(self):
        prices, calendar = synthetic()
        missing = prices.index[20]
        for removed in (prices.drop(missing), prices.mask(prices.index.to_series().eq(missing), axis=0)):
            days, _, _, _ = d.build_sequences(removed, calendar)
            for delta in range(8):
                self.assertNotIn(missing + pd.Timedelta(days=delta), days)
            self.assertIn(missing + pd.Timedelta(days=8), days)

    def test_scaler_fits_training_subset_only(self):
        prices, calendar = synthetic()
        _, p, f, y = d.build_sequences(prices, calendar)
        s = d.fit_scaler(p[:60], f[:60], y[:60])
        p[60:] += 1e6
        s2 = d.fit_scaler(p[:60], f[:60], y[:60])
        for key in s:
            np.testing.assert_array_equal(s[key], s2[key])

    def test_models_cpu_gradients_and_attention_causality(self):
        for name in ("LSTM", "TFT"):
            r.seed_everything(42)
            model = d.make_model(name, self.config).eval()
            past = torch.randn(2, 168, len(d.PAST_FEATURES))
            future = torch.randn(2, 24, len(d.FUTURE_FEATURES))
            prediction = model(past, future)
            self.assertEqual(prediction.shape, (2, 24))
            self.assertEqual(prediction.device.type, "cpu")
            prediction.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            changed = future.clone()
            changed[:, 12:] += 100
            torch.testing.assert_close(prediction[:, :12], model(past, changed)[:, :12])
            if name == "TFT":
                _, weights = model(past, future, interpret=True)
                torch.testing.assert_close(weights["past_weights"].sum(-1), torch.ones(2, 168))
                torch.testing.assert_close(weights["future_weights"].sum(-1), torch.ones(2, 24))
                self.assertEqual(float(weights["attention"][:, 0, 169:].abs().sum().detach()), 0.)

    def test_fit_cutoff_determinism_and_checkpoint_reload(self):
        prices, calendar = synthetic()
        days, p, f, y = d.build_sequences(prices, calendar)
        cutoff = days[65]
        config = dict(self.config, batch_size=32)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "checkpoints").mkdir()
            model, scaler, audit = r.fit("LSTM", config, 42, p, f, y, days, cutoff,
                                          output, io.StringIO(), epochs=2)
            xp, xf = d.transform(p[65:66], f[65:66], scaler)
            expected = r.predict(model, xp, xf, scaler)
            y[65:] += 99999
            changed, s2, _ = r.fit("LSTM", config, 42, p, f, y, days, cutoff,
                                    output, io.StringIO(), epochs=2)
            np.testing.assert_array_equal(expected, r.predict(changed, xp, xf, s2))
            self.assertLess(pd.Timestamp(audit["last_training_target"]), cutoff)
            checkpoint = torch.load(next((output / "checkpoints").glob("*.pt")), weights_only=True)
            loaded = d.make_model(checkpoint["name"], checkpoint["config"])
            loaded.load_state_dict(checkpoint["state_dict"])
            saved_scaler = {k: np.asarray(v, dtype=np.float32) for k, v in checkpoint["scaler"].items()}
            pp, ff = d.transform(p[65:66], f[65:66], saved_scaler)
            np.testing.assert_allclose(expected, r.predict(loaded, pp, ff, saved_scaler), atol=1e-4)


if __name__ == "__main__":
    unittest.main()
