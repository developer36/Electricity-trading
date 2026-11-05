#!/usr/bin/env python3
"""Download pinned Shandong prices, add ERA5 weather, and build a research table."""

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
from http.client import HTTPException
import json
import math
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "96ddcf9b020e98be35c9b8eb1c0f8b6e1fb8dc6f"
PRICE_BLOB = "d02840aaf039b1cbf5f230f6d301fa4e7d9e72a7"
PRICE_SIZE = 11297296
PRICE_URL = (f"https://raw.githubusercontent.com/ronaldowzy/price_forecast/{COMMIT}/data/"
             + quote("价格预测数据集.csv"))
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
VARIABLES = {
    "temperature_2m": "°C", "relative_humidity_2m": "%",
    "precipitation": "mm", "cloud_cover": "%",
    "wind_speed_10m": "m/s", "wind_speed_100m": "m/s",
    "shortwave_radiation": "W/m²",
}
# These quantities describe the preceding hour, timestamped at its END.
INTERVAL_VARIABLES = {"precipitation", "shortwave_radiation"}
LOCAL_TZ = timezone(timedelta(hours=8))


class DataError(Exception):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_bytes(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


def fetch(url):
    if (urlsplit(url).scheme != "https" or urlsplit(url).hostname not in
            {"raw.githubusercontent.com", "archive-api.open-meteo.com"}):
        raise DataError("拒绝非数据源 HTTPS 地址。")
    for attempt in range(4):
        try:
            request = Request(url, headers={"User-Agent": "Electricity-trading-research/1.0"})
            with urlopen(request, timeout=60) as response:
                if "text/html" in response.headers.get("Content-Type", "").lower():
                    raise DataError("数据接口返回 HTML 页面。")
                return response.read()
        except HTTPError as exc:
            status = exc.code
            exc.close()
            if (status == 429 or status >= 500) and attempt < 3:
                time.sleep(5 * (attempt + 1))
                continue
            raise DataError(f"数据接口返回 HTTP {status}。") from None
        except (URLError, TimeoutError, HTTPException):
            if attempt < 3:
                time.sleep(2 ** attempt)
                continue
            raise DataError("下载失败，请检查网络/代理；重跑可复用已校验缓存。") from None


def validate_price(content):
    blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\x00" + content).hexdigest()
    if len(content) != PRICE_SIZE or blob != PRICE_BLOB:
        raise DataError("山东价格文件大小或 Git blob 校验失败。")


def download_prices(data_dir, force=False):
    path = data_dir / "raw/shandong/prices.csv"
    if path.exists() and not force:
        validate_price(path.read_bytes())
        print("山东价格：已校验，复用本地文件。", flush=True)
    else:
        content = fetch(PRICE_URL)
        validate_price(content)
        atomic_bytes(path, content)
        print("山东价格：下载并校验完成。", flush=True)
    write_json(data_dir / "metadata/price_source.json", {
        "repository": "https://github.com/ronaldowzy/price_forecast",
        "commit": COMMIT, "url": PRICE_URL, "git_blob_sha1": PRICE_BLOB,
        "sha256": sha256(path), "size": path.stat().st_size, "checked_at_utc": utc_now(),
        "note": "社区聚合数据；字段发布时间与价格口径尚需进一步核验。",
    })
    return path


def read_prices(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields or not {"日期", "时刻", "日前价格", "实时价格"}.issubset(fields):
            raise DataError("价格文件缺少必需列。")
        rows = list(reader)
    if not rows:
        raise DataError("价格文件为空。")
    return fields, rows


def delivery_end(row):
    day = date.fromisoformat(row["日期"])
    hour, minute = map(int, row["时刻"].split(":"))
    if (not 0 <= hour <= 24 or minute not in (0, 15, 30, 45)
            or (hour == 24 and minute != 0) or (hour == minute == 0)):
        raise DataError("价格时刻必须在 00:15 至 24:00 的15分钟网格上。")
    return datetime.combine(day, datetime.min.time(), LOCAL_TZ) + timedelta(hours=hour, minutes=minute)


def month_ranges(start, end):
    while start <= end:
        next_month = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
        stop = min(end, next_month - timedelta(days=1))
        yield start, stop
        start = stop + timedelta(days=1)


def weather_url(locations, start, end):
    return ARCHIVE_URL + "?" + urlencode({
        "latitude": ",".join(str(x["latitude"]) for x in locations),
        "longitude": ",".join(str(x["longitude"]) for x in locations),
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "hourly": ",".join(VARIABLES), "models": "era5",
        "timezone": "Asia/Shanghai", "wind_speed_unit": "ms",
    })


def validate_weather(payload, locations, start, end):
    items = payload if isinstance(payload, list) else [payload]
    if len(items) != len(locations):
        raise DataError("天气返回地点数量不符。")
    expected = []
    t = datetime.combine(start, datetime.min.time())
    while t.date() <= end:
        expected.append(t.isoformat(timespec="minutes"))
        t += timedelta(hours=1)
    for item, location in zip(items, locations):
        if item.get("timezone") != "Asia/Shanghai" or item.get("utc_offset_seconds") != 28800:
            raise DataError("天气时区不符。")
        # ERA5 snaps points to its grid; catch swapped or unrelated locations.
        if (abs(item.get("latitude", 999) - location["latitude"]) > 0.5
                or abs(item.get("longitude", 999) - location["longitude"]) > 0.5):
            raise DataError("天气网格位置与请求地点不符。")
        hourly = item.get("hourly", {})
        if hourly.get("time") != expected:
            raise DataError("天气小时序列不完整或重复。")
        for variable, unit in VARIABLES.items():
            values = hourly.get(variable, [])
            if item.get("hourly_units", {}).get(variable) != unit or len(values) != len(expected):
                raise DataError(f"天气字段 {variable} 的单位或长度不符。")
            if any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in values):
                raise DataError(f"天气字段 {variable} 含缺失或非有限值，停止生成完整表。")
    return items


def download_weather(data_dir, locations, start, end, force=False):
    # Include locations/model/variables in the cache key, avoiding stale cache reuse.
    manifest = []
    monthly = []
    for first, last in month_ranges(start, end):
        url = weather_url(locations, first, last)
        key = hashlib.sha256(url.encode()).hexdigest()[:16]
        path = data_dir / f"raw/weather/era5_{first}_{last}_{key}.json"
        if path.exists() and not force:
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            content = fetch(url)
            payload = json.loads(content)
            validate_weather(payload, locations, first, last)
            atomic_bytes(path, content)
            # Limit the request rate; each multi-location query counts as multiple calls.
            time.sleep(1)
        items = validate_weather(payload, locations, first, last)
        monthly.append(items)
        manifest.append({"url": url, "path": str(path.relative_to(data_dir)),
                         "sha256": sha256(path)})
        print(f"ERA5 天气：{first} 至 {last}，{len(items)} 地点，已校验。", flush=True)
    write_json(data_dir / "metadata/weather_source.json", {
        "source": "Open-Meteo Historical Weather API / ECMWF ERA5", "model": "era5",
        "data_kind": "reanalysis_ex_post", "available_at": None,
        "note": "事后再分析，不是日前天气预报；不能直接作为预测交付日的未来已知特征。",
        "license": "CC BY 4.0", "api_terms": "https://open-meteo.com/en/terms",
        "attribution": "Weather data by Open-Meteo.com, ERA5 by ECMWF/Copernicus",
        "units": VARIABLES, "locations": locations, "checked_at_utc": utc_now(),
        "requests": manifest,
    })
    return monthly


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def weather_tables(monthly, locations):
    by_city = []
    aggregate = {}
    for items in monthly:
        for index, timestamp in enumerate(items[0]["hourly"]["time"]):
            timestamp = datetime.fromisoformat(timestamp).replace(tzinfo=LOCAL_TZ)
            values = {}
            for variable in VARIABLES:
                values[variable] = sum(x["hourly"][variable][index] for x in items) / len(items)
            if timestamp in aggregate:
                raise DataError("天气分片包含重复时刻。")
            aggregate[timestamp] = values
            for location, item in zip(locations, items):
                by_city.append({"weather_time": timestamp.isoformat(), "location_id": location["id"],
                                "location_name": location["name"],
                                **{v: item["hourly"][v][index] for v in VARIABLES}})
    return aggregate, by_city


def merge_prices(rows, aggregate):
    """Ex-post alignment. Interval weather is NOT a quarter-hour measurement."""
    result = []
    seen = set()
    for row in rows:
        end = delivery_end(row)
        start = end - timedelta(minutes=15)
        if end in seen:
            raise DataError("价格存在重复交付时段。")
        seen.add(end)
        instant = start.replace(minute=0, second=0, microsecond=0)
        interval_end = instant + timedelta(hours=1)
        try:
            weather = {
                f"weather_{v}_reanalysis_mean": aggregate[
                    interval_end if v in INTERVAL_VARIABLES else instant][v]
                for v in VARIABLES
            }
        except KeyError:
            raise DataError("天气覆盖不足，不能完成全部价格时段对齐。") from None
        result.append({**row, "delivery_start": start.isoformat(), "delivery_end": end.isoformat(),
                       "weather_instant_time": instant.isoformat(),
                       "weather_interval_end": interval_end.isoformat(),
                       "weather_kind": "reanalysis_ex_post", "weather_available_at": "",
                       **weather})
    return result


def build(data_dir, price_path, locations, force=False):
    fields, rows = read_prices(price_path)
    endpoints = [delivery_end(r) for r in rows]
    start = (min(endpoints) - timedelta(minutes=15)).date()
    # 24:00 belongs to the following day; include its hourly interval weather.
    end = max(endpoints).date()
    monthly = download_weather(data_dir, locations, start, end, force)
    aggregate, cities = weather_tables(monthly, locations)
    merged = merge_prices(rows, aggregate)
    processed = data_dir / "processed/shandong"
    write_csv(processed / "weather_hourly_by_city.csv",
              ["weather_time", "location_id", "location_name", *VARIABLES], cities)
    write_csv(processed / "weather_hourly_mean.csv", ["weather_time", *VARIABLES],
              ({"weather_time": t.isoformat(), **values} for t, values in sorted(aggregate.items())))
    output = processed / "prices_with_weather.csv"
    write_csv(output, list(merged[0]), merged)
    nonempty = lambda key: sum(bool(r[key].strip()) for r in rows)
    day_ahead_counts = {}
    for r in rows:
        day_ahead_counts.setdefault(r["日期"], 0)
        day_ahead_counts[r["日期"]] += bool(r["日前价格"].strip())
    report = {
        "generated_at_utc": utc_now(), "price_rows": len(rows), "original_columns": len(fields),
        "merged_rows": len(merged), "merged_columns": len(merged[0]),
        "date_start": min(r["日期"] for r in rows), "date_end": max(r["日期"] for r in rows),
        "unique_delivery_intervals": len(set(endpoints)), "weather_locations": len(locations),
        "weather_hours": len(aggregate), "weather_city_rows": len(cities),
        "weather_missing_values": 0, "day_ahead_nonempty": nonempty("日前价格"),
        "real_time_nonempty": nonempty("实时价格"),
        "day_ahead_points_per_day": {str(n): list(day_ahead_counts.values()).count(n)
                                    for n in sorted(set(day_ahead_counts.values()))},
        "output_sha256": sha256(output),
        "warnings": [
            "日前价格混有小时标签、重复填充的15分钟标签及空值，保留原样，不插值、不前填标签。",
            "原始时刻按交付结束解释：00:15为00:00—00:15，24:00为次日00:00；仍需向源方核验。",
            "天气为16城市代表点等权均值，不是全省面积、装机或负荷加权真值。",
            "温湿风云使用区间起点所在整点；降水和辐射取对应小时结束标签，未伪造15分钟分辨率。",
            "所有天气列标记reanalysis_ex_post，available_at未知；不可直接用于日前未来输入。",
        ],
    }
    write_json(data_dir / "metadata/quality_report.json", report)
    print(f"完成：{len(merged)} 行，天气缺失 0，输出 {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices-only", action="store_true", help="仅下载并校验山东原始价格数据")
    parser.add_argument("--force", action="store_true", help="重新下载所有原始文件")
    args = parser.parse_args()
    data_dir = ROOT / "data"
    try:
        prices = download_prices(data_dir, args.force)
        if not args.prices_only:
            locations = json.loads((ROOT / "configs/weather_locations.json").read_text())['locations']
            build(data_dir, prices, locations, args.force)
    except (DataError, OSError, ValueError) as exc:
        print(f"数据准备失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
