import importlib.util
from pathlib import Path
import unittest
spec = importlib.util.spec_from_file_location('solar', Path(__file__).resolve().parents[1]/'scripts/run_solar_backtest.py')
solar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solar)

class SolarTests(unittest.TestCase):
    def test_shortfall_settlement(self):
        self.assertEqual(solar.settle(40,50,300,400),11000)
    def test_equal_prices_remove_nomination_effect(self):
        self.assertEqual(solar.settle(40,80,-20,-20),-800)
    def test_capacity_and_threshold(self):
        self.assertEqual(solar.nomination(95,100,30,25),100)
        self.assertEqual(solar.nomination(50,100,-30,25),40)
        self.assertEqual(solar.nomination(50,100,25,25),50)
    def test_ledger(self):
        import csv
        p=Path(__file__).resolve().parents[1]/'outputs/solar_v1/ledger.csv'
        with p.open() as f:
            for r in csv.DictReader(f):
                self.assertLess(r['latest_feature_day'],r['forecast_origin'][:10])
                self.assertTrue(0<=float(r['cleared_mwh'])<=100)
                self.assertAlmostEqual(float(r['revenue']),float(r['rt_energy_revenue'])+float(r['da_spread_revenue']),places=7)

if __name__=='__main__':unittest.main()
