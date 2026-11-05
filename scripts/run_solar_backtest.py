"""Synthetic PV, price-taking quantity nominations; not a market clearing simulator."""
import csv
import hashlib
import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def settle(actual, nominated, da, rt):
    return actual * rt + nominated * (da - rt)

def nomination(predicted, capacity, spread, threshold):
    factor = 1.2 if spread > threshold else 0.8 if spread < -threshold else 1.0
    return min(capacity, max(0.0, predicted * factor))

def run():
    out = ROOT / 'outputs/solar_v1'
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'data/raw/shandong/prices.csv'
    groups = defaultdict(list)
    with source.open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            h, m = map(int, r['时刻'].split(':'))
            groups[(r['日期'], h + (m > 0))].append(r)
    hourly = {}
    for key, rows in groups.items():
        if len(rows) != 4:
            continue
        try:
            hourly[key] = (sum(max(0.0, float(r['光伏总加(实际)'])) for r in rows)/4,
                           sum(float(r['实时价格']) for r in rows)/4)
        except ValueError:
            continue
    capacity = 100.0
    # Calibration uses only the pre-validation period, never future maxima.
    scale = capacity / max(pv for (d, _), (pv, _) in hourly.items() if d < '2025-01-01')
    def prepare(split):
        path = ROOT / f'outputs/baselines_v1/{split}_predictions.csv'
        data, excluded = [], 0
        with path.open() as f:
            for r in csv.DictReader(f):
                d, h = r['delivery_date'], int(r['hour_end'])
                day = date.fromisoformat(d)
                history = [hourly.get(((day-timedelta(days=k)).isoformat(), h)) for k in range(2, 9)]
                if (d, h) not in hourly or any(v is None for v in history):
                    excluded += 1
                    continue
                pv, rt = hourly[(d, h)]
                data.append(dict(delivery_date=d, hour_end=h, forecast_origin=r['forecast_origin'],
                    actual_mwh=min(capacity, pv*scale), predicted_mwh=sum(min(capacity,v[0]*scale) for v in history)/7,
                    da_actual=float(r['actual']), rt_actual=rt, da_predicted=float(r['LightGBM']),
                    rt_predicted=sum(v[1] for v in history)/7,
                    latest_feature_day=(day-timedelta(days=2)).isoformat()))
        return data, excluded
    validation, val_excluded = prepare('validation')
    test, test_excluded = prepare('test')
    candidates = [0, 25, 50, 100, 200]
    def score(rows, threshold):
        return sum(settle(r['actual_mwh'], nomination(r['predicted_mwh'],capacity,r['da_predicted']-r['rt_predicted'],threshold),r['da_actual'],r['rt_actual']) for r in rows)
    scores = {t: score(validation,t) for t in candidates}
    chosen = max(candidates,key=lambda t:scores[t])
    ledger, metrics = [], {}
    for strategy in ['actual_rt_only','forecast_nomination','fixed_80pct','price_signal','oracle_quantity']:
        daily = defaultdict(float)
        deviation = 0.0
        for r in test:
            pred, spread = r['predicted_mwh'], r['da_predicted']-r['rt_predicted']
            q = {'actual_rt_only':0.0, 'forecast_nomination':pred, 'fixed_80pct':0.8*pred,
                 'price_signal':nomination(pred,capacity,spread,chosen),
                 'oracle_quantity':nomination(pred,capacity,r['da_actual']-r['rt_actual'],0)}[strategy]
            revenue = settle(r['actual_mwh'],q,r['da_actual'],r['rt_actual'])
            daily[r['delivery_date']] += revenue
            deviation += abs(r['actual_mwh']-q)
            ledger.append(dict(r,strategy=strategy,nominated_mwh=q,cleared_mwh=q,
                               rt_energy_revenue=r['actual_mwh']*r['rt_actual'],
                               da_spread_revenue=q*(r['da_actual']-r['rt_actual']),revenue=revenue))
        metrics[strategy] = dict(revenue=sum(daily.values()),worst_day=min(daily.values()),
                                 negative_days=sum(v<0 for v in daily.values()),absolute_deviation_mwh=deviation)
    base = metrics['forecast_nomination']['revenue']
    for v in metrics.values():
        v['increment_vs_forecast'] = v['revenue']-base
    with (out/'ledger.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(ledger[0]));writer.writeheader();writer.writerows(ledger)
    manifest=dict(capacity_mw=capacity,scale=scale,scale_fit_end='2024-12-31',selected_threshold=chosen,
                  validation_scores=scores,test_hours=len(test),excluded_test_hours=test_excluded,
                  excluded_validation_hours=val_excluded,metrics=metrics,
                  assumptions=['source price assumed CNY/MWh, power MW; not verified',
                  'synthetic capped regional output; negative raw PV clipped to zero',
                  'D-2 actuals assumed published by D-1 noon; publication not verified',
                  '100% nomination acceptance; no bid-price or clearing model',
                  'generic two-settlement, not Shandong historical settlement',
                  'no contracts, fees, subsidy, curtailment decision or operating costs'],
                  inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,ROOT/'outputs/baselines_v1/test_predictions.csv',ROOT/'outputs/baselines_v1/validation_predictions.csv']})
    (out/'metrics.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    report=['# 模拟光伏电站报量回测 v1','',
            '**研究模拟，不代表实际电厂收益或山东规则下的可执行策略。**','',
            f'100 MW 模拟电站；测试 {len(test)} 小时，剔除 {test_excluded} 小时；验证集选定价差信号阈值 {chosen}。',
            '价格假设为元/MWh，因此以下金额单位为假设元；未核验原始单位。','',
            '|策略|模拟收入|相对预测出力申报增量|最差日收入|','|---|---:|---:|---:|']
    for k,v in metrics.items():
        report.append(f"|{k}|{v['revenue']:.2f}|{v['increment_vs_forecast']:.2f}|{v['worst_day']:.2f}|")
    report += ['', 'oracle_quantity 使用真实未来价差，只是同一 80%–120% 报量范围内的事后上限，不可执行。',
               'actual_rt_only 为纯实时结算数学参照，不声称该申报方式在真实市场可用。',
               '最差日基于有效小时，缺失小时未补齐；收入不是扣除成本后的利润。',
               '复现：`.venv/bin/python scripts/run_solar_backtest.py`。逐时流水见 `outputs/solar_v1/ledger.csv`，配置、输入哈希和指标见同目录 `metrics.json`。']
    (ROOT/'reports/solar_v1.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    run()
