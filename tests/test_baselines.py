"""Tests for label semantics, availability boundaries, and metric correctness."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts import run_baselines as b


def synthetic(days=90):
    index = pd.date_range("2024-01-01", periods=days)
    prices = pd.DataFrame(np.arange(days * 24, dtype=float).reshape(-1, 24),
                          index=index, columns=range(1, 25))
    calendar = pd.DataFrame({"是否节假日": "否", "是否周末休息日": "否"}, index=index)
    return prices, calendar


class BaselineTests(unittest.TestCase):
    def test_hourly_labels_and_repeated_quarter_hours(self):
        rows = []
        for day, repeated in (("2024-01-01", False), ("2024-01-02", True)):
            for slot in range(1, 97):
                hour = (slot - 1) // 4 + 1
                rows.append({"日期": day, "时刻": f"{slot // 4:02d}:{slot % 4 * 15:02d}",
                             "日前价格": -hour if repeated or slot % 4 == 0 else np.nan,
                             "是否节假日": "否", "是否周末休息日": "否"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prices.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            prices, _, counts = b.load_hourly(path)
            self.assertEqual(counts, {"2024-01-01": 24, "2024-01-02": 96})
            self.assertEqual(prices.shape, (2, 24))
            self.assertEqual(prices.loc["2024-01-01", 24], -24)
            rows[-1]["日前价格"] = 1000  # inconsistent 96-point day must be rejected
            pd.DataFrame(rows).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                b.load_hourly(path)

    def test_features_never_read_target_or_future_prices(self):
        prices, calendar = synthetic()
        x, _, ref = b.build_samples(prices, calendar)
        day = prices.index[20]
        self.assertEqual(x.loc[day, "price_d1_h24"], prices.iloc[19, 23])
        self.assertEqual(ref.loc[day, 1], prices.iloc[13, 0])
        self.assertAlmostEqual(x.loc[day, "mean7_h01"], prices.iloc[13:20, 0].mean())
        prices.loc[day:] += 99999
        changed, _, _ = b.build_samples(prices, calendar)
        np.testing.assert_array_equal(x.loc[day].to_numpy(), changed.loc[day].to_numpy())

    def test_missing_calendar_day_is_not_replaced_by_previous_row(self):
        prices, calendar = synthetic()
        missing = prices.index[15]
        prices = prices.drop(missing)
        x, _, _ = b.build_samples(prices, calendar)
        for offset in range(8):
            self.assertNotIn(missing + pd.Timedelta(days=offset), x.index)
        self.assertIn(missing + pd.Timedelta(days=8), x.index)

    def test_metrics_and_negative_event_counts(self):
        score = b.metrics([-10, 0, 20], [-5, -5, 30], [0, 10, 30], 15)
        self.assertAlmostEqual(score["MAE"], 20 / 3)
        self.assertAlmostEqual(score["RMSE"], np.sqrt(50))
        self.assertAlmostEqual(score["rMAE"], 2 / 3)
        self.assertAlmostEqual(score["Bias"], 10 / 3)
        self.assertEqual(score["peak_MAE"], 10)
        self.assertEqual(score["negative_precision"], .5)
        self.assertEqual(score["negative_recall"], 1)
        self.assertAlmostEqual(score["negative_F1"], 2 / 3)
        self.assertIsNone(b.metrics([0], [1], [0], 1)["rMAE"])
        with self.assertRaises(ValueError):
            b.metrics([np.nan], [1], [2], 1)

    def test_pooled_hour_features_match_day_and_hour(self):
        prices, calendar = synthetic()
        x, _, _ = b.build_samples(prices, calendar)
        pooled = b.pooled_features(x.iloc[:2])
        self.assertEqual(len(pooled), 48)
        self.assertEqual(pooled.loc[23, "hour_end"], 24)
        self.assertEqual(pooled.loc[24, "hour_end"], 1)
        self.assertEqual(pooled.loc[23, "same_hour_price_d1"], x.iloc[0]["price_d1_h24"])
        self.assertEqual(pooled.loc[24, "same_hour_price_d1"], x.iloc[1]["price_d1_h01"])

    def test_walk_forward_excludes_current_day_labels(self):
        prices, calendar = synthetic()
        x, y, _ = b.build_samples(prices, calendar)
        start, end = x.index[65], x.index[67]
        args = ("LEAR", {"alpha": 1.}, x)
        pred, _, audit = b.walk_forward(*args, y, start, end, 7, 42)
        changed = y.copy()
        changed.loc[start:] += 100000
        pred2, _, _ = b.walk_forward(*args, changed, start, end, 7, 42)
        np.testing.assert_allclose(pred, pred2)
        self.assertEqual(len(audit["retraining"]), 1)
        self.assertLess(pd.Timestamp(audit["retraining"][0]["last_training_target"]), start)


if __name__ == "__main__":
    unittest.main()
