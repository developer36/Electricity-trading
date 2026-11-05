"""Availability tests use interventions on forbidden data, without model downloads."""
import unittest
import numpy as np
import pandas as pd

from scripts.foundation_models import make_input, KNOWN_FEATURES
from tests.test_baselines import synthetic


class FoundationAvailabilityTests(unittest.TestCase):
    def test_target_and_future_price_intervention_cannot_change_inputs(self):
        prices, calendar = synthetic()
        day = prices.index[30]
        before = make_input(prices, calendar, day)
        prices.loc[day:] = -999999
        after = make_input(prices, calendar, day)
        np.testing.assert_array_equal(before.target, after.target)
        np.testing.assert_array_equal(before.known, after.known)
        self.assertEqual(before.target.shape, (1, 168))
        self.assertEqual(before.known.shape, (len(KNOWN_FEATURES), 192))

    def test_hour_24_stays_in_previous_delivery_day(self):
        prices, calendar = synthetic()
        day = prices.index[30]
        item = make_input(prices, calendar, day)
        self.assertEqual(item.target[0, -1], prices.loc[day-pd.Timedelta(days=1), 24])
        self.assertEqual(item.target[0, 0], prices.loc[day-pd.Timedelta(days=7), 1])
        np.testing.assert_allclose(item.known[:2, 168], [np.sin(2*np.pi/24), np.cos(2*np.pi/24)])

    def test_missing_history_is_rejected_not_filled_or_time_compressed(self):
        prices, calendar = synthetic()
        day = prices.index[30]
        prices = prices.drop(day-pd.Timedelta(days=3))
        with self.assertRaises(ValueError):
            make_input(prices, calendar, day)

    def test_calendar_is_the_only_future_input(self):
        prices, calendar = synthetic()
        day = prices.index[30]
        before = make_input(prices, calendar, day)
        calendar.loc[day, "是否节假日"] = "是"
        after = make_input(prices, calendar, day)
        np.testing.assert_array_equal(before.target, after.target)
        np.testing.assert_array_equal(before.known[:, :168], after.known[:, :168])
        np.testing.assert_array_equal(after.known[-2, 168:], np.ones(24))


if __name__ == "__main__":
    unittest.main()
