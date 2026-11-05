"""Offline integrity and alignment tests; no external service or real dataset needed."""

from datetime import date, datetime, timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import download_dataset as data


LOCATIONS = [{"id": "jinan", "name": "济南", "latitude": 36.65, "longitude": 117.12},
             {"id": "qingdao", "name": "青岛", "latitude": 36.07, "longitude": 120.38}]


def payload(first=date(2024, 1, 1), last=date(2024, 1, 1)):
    hours = (last - first).days * 24 + 24
    start = datetime.combine(first, datetime.min.time())
    times = [(start + timedelta(hours=i)).isoformat(timespec="minutes") for i in range(hours)]
    return [{"latitude": loc["latitude"], "longitude": loc["longitude"],
             "timezone": "Asia/Shanghai", "utc_offset_seconds": 28800,
             "hourly_units": dict(data.VARIABLES),
             "hourly": {"time": times, **{v: [i + city * 10 for i in range(hours)]
                                          for v in data.VARIABLES}}}
            for city, loc in enumerate(LOCATIONS)]


def row(day="2024-01-01", time="00:15", price="100"):
    return {"日期": day, "时刻": time, "日前价格": price, "实时价格": "90"}


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "data"

    def test_delivery_midnight_and_invalid_times(self):
        self.assertEqual(data.delivery_end(row(time="24:00")).isoformat(),
                         "2024-01-02T00:00:00+08:00")
        for time in ("00:00", "24:15", "01:10", "25:00"):
            with self.subTest(time=time), self.assertRaises(data.DataError):
                data.delivery_end(row(time=time))

    def test_leap_year_month_chunks(self):
        self.assertEqual(list(data.month_ranges(date(2024, 2, 28), date(2024, 3, 1))),
                         [(date(2024, 2, 28), date(2024, 2, 29)),
                          (date(2024, 3, 1), date(2024, 3, 1))])

    def test_instant_vs_hourly_interval_and_no_price_imputation(self):
        aggregate, cities = data.weather_tables([payload(last=date(2024, 1, 2))], LOCATIONS)
        result = data.merge_prices([row(price=""), row(time="01:00"), row(time="24:00")], aggregate)
        self.assertEqual(result[0]["日前价格"], "")
        self.assertEqual(result[0]["weather_temperature_2m_reanalysis_mean"], 5)
        self.assertEqual(result[0]["weather_precipitation_reanalysis_mean"], 6)
        self.assertEqual(result[1]["weather_temperature_2m_reanalysis_mean"], 5)
        self.assertEqual(result[2]["weather_precipitation_reanalysis_mean"], 29)
        self.assertEqual(result[2]["weather_interval_end"], "2024-01-02T00:00:00+08:00")
        self.assertEqual(len(cities), 96)
        self.assertEqual(result[0]["weather_available_at"], "")
        self.assertEqual(result[0]["weather_kind"], "reanalysis_ex_post")

    def test_missing_weather_or_duplicate_delivery_rejected(self):
        with self.assertRaises(data.DataError):
            data.merge_prices([row()], {})
        aggregate, _ = data.weather_tables([payload()], LOCATIONS)
        with self.assertRaises(data.DataError):
            data.merge_prices([row(), row()], aggregate)

    def test_weather_null_units_grid_and_time_coverage_rejected(self):
        for mutation in ("null", "unit", "time", "grid"):
            value = payload()
            if mutation == "null":
                value[0]["hourly"]["temperature_2m"][0] = None
            elif mutation == "unit":
                value[0]["hourly_units"]["wind_speed_10m"] = "km/h"
            elif mutation == "time":
                value[0]["hourly"]["time"].pop()
            else:
                value[0]["latitude"] = 0
            with self.subTest(mutation=mutation), self.assertRaises(data.DataError):
                data.validate_weather(value, LOCATIONS, date(2024, 1, 1), date(2024, 1, 1))

    def test_price_integrity_skip_and_failed_force_preserves_original(self):
        content = b"test price data"
        blob = hashlib.sha1(b"blob 15\x00" + content).hexdigest()
        with patch.object(data, "PRICE_SIZE", len(content)), patch.object(data, "PRICE_BLOB", blob):
            with patch.object(data, "fetch", return_value=content) as fetch:
                path = data.download_prices(self.directory)
                data.download_prices(self.directory)
                self.assertEqual(fetch.call_count, 1)
            with patch.object(data, "fetch", return_value=b"invalid"):
                with self.assertRaises(data.DataError):
                    data.download_prices(self.directory, force=True)
            self.assertEqual(path.read_bytes(), content)
            self.assertFalse(list(self.directory.rglob("*.part")))

    def test_weather_cache_is_reused_and_invalid_payload_not_cached(self):
        body = json.dumps(payload()).encode()
        with patch.object(data, "fetch", return_value=body) as fetch, patch.object(data.time, "sleep"):
            for _ in range(2):
                data.download_weather(self.directory, LOCATIONS, date(2024, 1, 1), date(2024, 1, 1))
            self.assertEqual(fetch.call_count, 1)
        with patch.object(data, "fetch", return_value=b'{}'), patch.object(data.time, "sleep"):
            with self.assertRaises(data.DataError):
                data.download_weather(self.directory, LOCATIONS, date(2024, 2, 1), date(2024, 2, 1))
        self.assertEqual(len(list((self.directory / "raw/weather").glob("*.json"))), 1)

    def test_build_writes_complete_report_and_preserves_labels(self):
        path = self.directory / "raw/prices.csv"
        rows = [row(price=""), row(time="24:00")]
        data.write_csv(path, list(rows[0]), rows)
        body = json.dumps(payload(last=date(2024, 1, 2))).encode()
        with patch.object(data, "fetch", return_value=body), patch.object(data.time, "sleep"):
            data.build(self.directory, path, LOCATIONS)
        report = json.loads((self.directory / "metadata/quality_report.json").read_text())
        self.assertEqual(report["merged_rows"], 2)
        self.assertEqual(report["weather_missing_values"], 0)
        output = self.directory / "processed/shandong/prices_with_weather.csv"
        self.assertEqual(data.read_prices(output)[1][0]["日前价格"], "")
        self.assertEqual(report["output_sha256"], data.sha256(output))


if __name__ == "__main__":
    unittest.main()
