#!/usr/bin/env python3
"""Hourly day-ahead benchmarks: weekly naive, LASSO-LEAR variant, LightGBM.

Only lagged day-ahead prices and calendar data are used. No target-day actuals,
reanalysis weather, or supply/demand fields of unknown publication time enter X.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("WeeklyNaive", "LEAR", "LightGBM")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_hourly(path):
    """Validate the 96-row grid; select observed hour-end labels, never fill y."""
    raw = pd.read_csv(path)
    raw["date"] = pd.to_datetime(raw["日期"], format="%Y-%m-%d")
    expected = {f"{i // 4:02d}:{i % 4 * 15:02d}" for i in range(1, 97)}
    day_counts = {}
    for day, group in raw.groupby("date", sort=True):
        if len(group) != 96 or set(group["时刻"]) != expected:
            raise ValueError(f"Invalid or duplicate 15-minute grid: {day}")
        n = int(group["日前价格"].notna().sum())
        if n not in (0, 24, 96):
            raise ValueError(f"Unexpected label count {n}: {day}")
        if n == 24 and not group.loc[group["日前价格"].notna(), "时刻"].str.endswith(":00").all():
            raise ValueError(f"Non-hourly day-ahead labels: {day}")
        if n == 96:
            g = group.copy()
            g["slot"] = g["时刻"].map(lambda s: (int(s[:2]) * 60 + int(s[3:]) - 1) // 60)
            if not g.groupby("slot")["日前价格"].nunique().eq(1).all():
                raise ValueError(f"96-label day is not repeated hourly data: {day}")
        for name in ("是否节假日", "是否周末休息日"):
            if group[name].nunique() != 1 or group[name].iloc[0] not in ("是", "否"):
                raise ValueError(f"Invalid daily calendar field {name}: {day}")
        day_counts[str(day.date())] = n
    rows = raw.loc[raw["时刻"].str.endswith(":00")].copy()
    rows["hour_end"] = rows["时刻"].str[:2].astype(int)
    prices = rows.pivot(index="date", columns="hour_end", values="日前价格")
    if list(prices.columns) != list(range(1, 25)):
        raise ValueError("Expected hours 01:00 through 24:00")
    dates = pd.date_range(prices.index.min(), prices.index.max(), freq="D")
    prices = prices.reindex(dates)
    calendar = raw.groupby("date")[["是否节假日", "是否周末休息日"]].first().reindex(dates)
    if not np.isfinite(prices.to_numpy()[prices.notna().to_numpy()]).all():
        raise ValueError("Nonfinite price labels")
    return prices, calendar, day_counts


def build_samples(prices, calendar):
    """One sample/day; lag(1) means previous *calendar* day, not previous row."""
    prices = prices.reindex(pd.date_range(prices.index.min(), prices.index.max(), freq="D"))
    features = {}
    for lag in (1, 2, 3, 7):
        for hour in range(1, 25):
            features[f"price_d{lag}_h{hour:02d}"] = prices[hour].shift(lag)
    # Rolling statistics exclude the target day and require all seven past days.
    for hour in range(1, 25):
        history = prices[hour].shift(1).rolling(7, min_periods=7)
        features[f"mean7_h{hour:02d}"] = history.mean()
        features[f"std7_h{hour:02d}"] = history.std(ddof=0)
    for weekday in range(7):
        features[f"weekday_{weekday}"] = (prices.index.dayofweek == weekday).astype(float)
    features["year_sin"] = np.sin(2 * np.pi * (prices.index.dayofyear - 1) / 365.25)
    features["year_cos"] = np.cos(2 * np.pi * (prices.index.dayofyear - 1) / 365.25)
    for name, field in (("holiday", "是否节假日"), ("rest_day", "是否周末休息日")):
        features[name] = calendar[field].reindex(prices.index).map({"是": 1., "否": 0.})
    x = pd.DataFrame(features, index=prices.index)
    valid = x.notna().all(axis=1) & prices.notna().all(axis=1)
    return x.loc[valid], prices.loc[valid], prices.shift(7).loc[valid]


def pooled_features(x):
    """Pool hours for LightGBM, preserving the same daily information set."""
    base = np.repeat(np.asarray(x, dtype=float), 24, axis=0)
    hour = np.tile(np.arange(1, 25), len(x))
    extra = [hour, np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)]
    for prefix in ("price_d1", "price_d2", "price_d3", "price_d7", "mean7", "std7"):
        extra.append(x[[f"{prefix}_h{h:02d}" for h in range(1, 25)]].to_numpy().ravel())
    names = list(x.columns) + ["hour_end", "hour_sin", "hour_cos"] + [
        "same_hour_" + p for p in ("price_d1", "price_d2", "price_d3", "price_d7", "mean7", "std7")]
    return pd.DataFrame(np.column_stack([base] + extra), columns=names)


def make_model(name, params, seed):
    if name == "LEAR":
        return [make_pipeline(StandardScaler(), Lasso(alpha=params["alpha"],
                max_iter=20000, tol=1e-4)) for _ in range(24)]
    from lightgbm import LGBMRegressor
    return LGBMRegressor(objective="regression_l1", learning_rate=0.03,
                         reg_lambda=1., verbosity=-1, n_jobs=2,
                         random_state=seed, deterministic=True, force_col_wise=True,
                         **params)


def walk_forward(name, params, x, y, start, end, retrain_days, seed):
    days = x.index[(x.index >= start) & (x.index <= end)]
    if len(days) == 0:
        raise ValueError(f"No evaluation dates: {start} to {end}")
    result = pd.DataFrame(index=days, columns=y.columns, dtype=float)
    fit_log = []
    elapsed_fit = elapsed_predict = 0.
    model = None
    last_fit = None
    with warnings.catch_warnings():
        # A convergence failure invalidates the experiment instead of silently passing.
        warnings.simplefilter("error", ConvergenceWarning)
        for day in days:
            if model is None or (day - last_fit).days >= retrain_days:
                train = x.index < day
                if int(train.sum()) < 60:
                    raise ValueError("At least 60 training days required")
                before = time.perf_counter()
                model = make_model(name, params, seed)
                if name == "LEAR":
                    for h, estimator in enumerate(model):
                        estimator.fit(x.loc[train].to_numpy(), y.loc[train].iloc[:, h].to_numpy())
                else:
                    model.fit(pooled_features(x.loc[train]), y.loc[train].to_numpy().ravel())
                elapsed_fit += time.perf_counter() - before
                last_fit = day
                fit_log.append({"forecast_day": str(day.date()),
                                "last_training_target": str(x.index[train][-1].date()),
                                "training_days": int(train.sum())})
            before = time.perf_counter()
            if name == "LEAR":
                prediction = [estimator.predict(x.loc[[day]].to_numpy())[0] for estimator in model]
            else:
                prediction = model.predict(pooled_features(x.loc[[day]]))
            result.loc[day] = prediction
            elapsed_predict += time.perf_counter() - before
    return result, model, {"fit_seconds": elapsed_fit, "predict_seconds": elapsed_predict,
                           "retraining": fit_log}


def metrics(actual, predicted, reference, peak_threshold):
    actual, predicted, reference = [np.asarray(a, dtype=float).ravel() for a in (actual, predicted, reference)]
    if not (len(actual) == len(predicted) == len(reference)) or not len(actual):
        raise ValueError("Metrics require equally sized, nonempty arrays")
    if not all(np.isfinite(a).all() for a in (actual, predicted, reference)):
        raise ValueError("Metrics reject missing/nonfinite labels or predictions")
    error = predicted - actual
    mae = np.mean(np.abs(error))
    baseline_mae = np.mean(np.abs(reference - actual))
    peak = actual >= peak_threshold
    negative = actual < 0
    guess_negative = predicted < 0
    tp = int(np.sum(negative & guess_negative))
    fp = int(np.sum(~negative & guess_negative))
    fn = int(np.sum(negative & ~guess_negative))
    return {"n": len(actual), "MAE": float(mae), "RMSE": float(np.sqrt(np.mean(error ** 2))),
            "rMAE": float(mae / baseline_mae) if baseline_mae > 0 else None,
            "improvement_pct": float(100 * (1 - mae / baseline_mae)) if baseline_mae > 0 else None,
            "Bias": float(np.mean(error)), "peak_n": int(peak.sum()),
            "peak_MAE": float(np.abs(error[peak]).mean()) if peak.any() else None,
            "negative_n": int(negative.sum()),
            "negative_MAE": float(np.abs(error[negative]).mean()) if negative.any() else None,
            "negative_precision": tp / (tp + fp) if tp + fp else None,
            "negative_recall": tp / (tp + fn) if tp + fn else None,
            "negative_F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def evaluate_frame(predictions, peak_threshold):
    return {name: metrics(predictions.actual, predictions[name], predictions.WeeklyNaive, peak_threshold)
            for name in MODELS}


def prediction_frame(y, reference, forecasts):
    frames = []
    for name, prediction in forecasts.items():
        if not prediction.index.equals(y.index) or not prediction.columns.equals(y.columns):
            raise ValueError(f"Unaligned predictions: {name}")
    for day in y.index:
        for hour in range(1, 25):
            frames.append({"delivery_date": str(day.date()), "hour_end": hour,
                           "forecast_origin": (day - pd.Timedelta(days=1)).strftime("%Y-%m-%dT12:00:00+08:00"),
                           "actual": y.loc[day, hour], "WeeklyNaive": reference.loc[day, hour],
                           **{name: forecast.loc[day, hour] for name, forecast in forecasts.items()}})
    return pd.DataFrame(frames)


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def plot_results(frame, output):
    os.environ.setdefault("MPLCONFIGDIR", str(output / "mpl-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    for name in MODELS:
        error = np.abs(frame[name] - frame.actual)
        axes[0].plot(frame.assign(error=error).groupby("hour_end").error.mean(), label=name)
    axes[0].set(xlabel="Delivery hour end (1-24)", ylabel="MAE (source price unit)", title="Test error by hour")
    axes[0].legend()
    last = frame.tail(7 * 24).reset_index(drop=True)
    axes[1].plot(last.actual, color="black", linewidth=1.5, label="Actual")
    for name in MODELS:
        axes[1].plot(last[name], alpha=.8, linewidth=.9, label=name)
    axes[1].set(xlabel="Hour in final test week", ylabel="Price (source unit)", title="Final seven delivery days")
    axes[1].legend(ncol=4)
    fig.savefig(output / "test_comparison.png", dpi=160)
    plt.close(fig)


def report_text(meta, val, test):
    lines = ["# 小时级日前电价基线测评", "",
             "本报告由 `scripts/run_baselines.py` 实际运行生成。", "",
             "## 实验口径", "",
             f"- 初始训练：{meta['initial_train_start']} 至 {meta['initial_train_end']}；验证：{meta['config']['validation_start']} 至 {(pd.Timestamp(meta['config']['test_start']) - pd.Timedelta(days=1)).date()}；测试：{meta['config']['test_start']} 至 {meta['config']['test_end']}。",
             f"- 验证 {meta['validation_days']} 天，测试 {meta['test_days']} 天；每天预测 24 点，所有模型评分样本完全相同。",
             f"- 扩展训练窗口，每 {meta['config']['retrain_days']} 天重训；当日 24 点一次生成，不使用当日真实价格递推。测试过程中允许按既定频率使用已到达的历史标签更新模型。",
             "- 预测时点约定为 D-1 12:00（北京时间），假设 D-1 及更早交付日的日前价格届时已发布；源数据未附发布时间，因此属于带明确可得性假设的研究回测。",
             "- 小时标签取 01:00 至 24:00 的原始日前价格，按交付结束解释；该时段定义及目标结算口径仍待源方核验。不对标签插值，不裁剪负价或尖峰。",
             "- 输入：D-1/2/3/7 日完整价格曲线、过去 7 日逐小时均值/标准差、星期独热、年周期、节假日/休息日。目标日供需实际值、供需预测字段及 ERA5 天气均未使用。",
             "- LEAR 为本地价格+日历 LASSO 自回归变体：每小时独立模型，训练窗口内标准化；不是 epftoolbox 原样复刻，未接入供需外生变量或其非线性价格变换。",
             "- LightGBM 为跨小时共享模型，加小时编码及同小时滞后摘要；与 LEAR 可得信息相同，建模结构不同。L1 目标，不使用测试集早停。",
             f"- 尖峰阈值固定为初始训练集价格 95% 分位：{meta['peak_threshold']:.6f}；负价定义为实际价格 < 0。",
             "- 不报告 MAPE；MAE/RMSE/Bias 使用源价格单位（具体单位仍待源方核验），rMAE 为同一评分集上相对前一周同小时基线的 MAE 比值；Bias=预测−实际。", "",
             "## 验证与参数选择", "",
             f"选择结果：`{json.dumps(meta['selected'], ensure_ascii=False)}`。所有候选均按完整验证期滚动 MAE 比较，测试集不参与选择。", ""]
    for title, scores in (("验证集", val), ("测试集", test)):
        lines.extend([f"## {title}", "", "| 模型 | MAE | RMSE | rMAE | 相对改善 | Bias | 尖峰 MAE | 负价 F1 |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for name, s in scores.items():
            f1 = f"{s['negative_F1']:.4f}" if s['negative_F1'] is not None else "N/A"
            peak = f"{s['peak_MAE']:.3f}" if s['peak_MAE'] is not None else "N/A"
            relative = f"{s['rMAE']:.4f}" if s['rMAE'] is not None else "N/A"
            gain = f"{s['improvement_pct']:.2f}%" if s['improvement_pct'] is not None else "N/A"
            lines.append(f"| {name} | {s['MAE']:.3f} | {s['RMSE']:.3f} | {relative} | {gain} | {s['Bias']:.3f} | {peak} | {f1} |")
        first = next(iter(scores.values()))
        lines.extend(["", f"样本数 {first['n']}；尖峰 {first['peak_n']}；负价 {first['negative_n']}。", ""])
    lines.extend(["## 产物与限制", "",
                  "- `validation_predictions.csv` / `test_predictions.csv`：逐日逐小时真实值及三模型预测。",
                  "- `metrics.json`、`selection.csv`、`timing.json`：全量指标、参数比较与每次拟合的数据截止时间。",
                  "- `group_metrics.csv`：按小时、月份、季节、节假日/非节假日及工作日/周末分组评估。",
                  "- `hourly_labels.csv`、`metadata.json`：原始整点标签、特征定义、数据/代码哈希及依赖版本。",
                  "- `models.joblib`：最后一次重训的模型，用于审计，不是截至测试末日再训练的部署模型。",
                  "- `test_comparison.png`：逐小时 MAE 与最后七天价格曲线。",
                  "- 2025-12-31 起连续 9 天日前标签缺失，随后 6 天口径变为重复填充值；首轮测试截止 2025-12-30，未用这段尾部数据。",
                  "- 仅半年独立测试，未覆盖完整年度测试季节；本轮仅价格+日历，不能代表加入供需预测后的结果，也不是交易收益回测。",
                  "- 参考：[LEAR 官方定义](https://epftoolbox.readthedocs.io/en/latest/modules/lear_model.html)、[LightGBM](https://lightgbm.readthedocs.io/)。", ""])
    return "\n".join(lines)


def run(args):
    config = json.loads(args.config.read_text())
    vs, ts, te = [pd.Timestamp(config[k]) for k in ("validation_start", "test_start", "test_end")]
    if not vs < ts <= te or config["retrain_days"] < 1:
        raise ValueError("Invalid split/retraining configuration")
    # Refuse accidental reuse of an existing report directory.
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output directory is nonempty; choose a new --output")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output.resolve() / "mpl-cache"))
    # Preserve the executed source/config even if the working tree is later edited.
    source_bytes = Path(__file__).read_bytes()
    (args.output / "run_baselines_snapshot.py").write_bytes(source_bytes)
    save_json(args.output / "config_snapshot.json", config)
    prices, calendar, day_counts = load_hourly(args.data)
    x, y, reference = build_samples(prices, calendar)
    if te > prices.index.max():
        raise ValueError("Test end exceeds source coverage")
    validation_days = x.index[(x.index >= vs) & (x.index < ts)]
    test_days = x.index[(x.index >= ts) & (x.index <= te)]
    if len(validation_days) == 0 or len(test_days) == 0:
        raise ValueError("Empty validation/test split")
    initial = y.loc[y.index < vs]
    threshold = float(np.quantile(initial.to_numpy(), .95))
    selection, selected, val_forecasts, timings = [], {}, {}, {}
    for name, candidates in (("LEAR", [{"alpha": a} for a in config["lear_alphas"]]),
                             ("LightGBM", config["lightgbm_candidates"])):
        best = float("inf")
        for params in candidates:
            print(f"Validation {name} {params}", flush=True)
            pred, _, timing = walk_forward(name, params, x, y, vs, ts - pd.Timedelta(days=1),
                                           config["retrain_days"], config["seed"])
            score = metrics(y.loc[pred.index], pred, reference.loc[pred.index], threshold)["MAE"]
            selection.append({"model": name, "params": json.dumps(params), "validation_MAE": score})
            timings[f"validation_{name}_{json.dumps(params, sort_keys=True)}"] = timing
            print(f"  MAE={score:.6f}", flush=True)
            if score < best:
                best, selected[name], val_forecasts[name] = score, params, pred
    # Parameters are frozen before any test predictions or evaluation are computed.
    save_json(args.output / "selected_params.json", selected)
    test_forecasts, fitted = {}, {}
    for name in ("LEAR", "LightGBM"):
        print(f"Test {name} {selected[name]}", flush=True)
        pred, model, timing = walk_forward(name, selected[name], x, y, ts, te,
                                          config["retrain_days"], config["seed"])
        test_forecasts[name], fitted[name], timings[f"test_{name}"] = pred, model, timing
    val_frame = prediction_frame(y.loc[validation_days], reference.loc[validation_days], val_forecasts)
    test_frame = prediction_frame(y.loc[test_days], reference.loc[test_days], test_forecasts)
    val_scores, test_scores = evaluate_frame(val_frame, threshold), evaluate_frame(test_frame, threshold)
    groups = []
    dates = pd.to_datetime(test_frame.delivery_date)
    labels = {"hour": test_frame.hour_end, "month": dates.dt.strftime("%Y-%m"),
              "season": dates.dt.month.map({12:"winter",1:"winter",2:"winter",3:"spring",4:"spring",5:"spring",6:"summer",7:"summer",8:"summer",9:"autumn",10:"autumn",11:"autumn"}),
              "holiday": dates.map(calendar["是否节假日"]),
              "weekday_or_weekend": dates.dt.dayofweek.map(lambda d: "weekend" if d >= 5 else "weekday")}
    for kind, values in labels.items():
        for label, frame in test_frame.groupby(values):
            for name, score in evaluate_frame(frame, threshold).items():
                groups.append({"group": kind, "value": label, "model": name, **score})
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except subprocess.CalledProcessError:
        revision = "uncommitted repository"
    meta = {"config": config, "selected": selected, "peak_threshold": threshold,
            "initial_train_start": str(initial.index.min().date()), "initial_train_end": str(initial.index.max().date()),
            "initial_training_days": len(initial), "validation_days": len(validation_days), "test_days": len(test_days),
            "validation_missing_days": [str(d.date()) for d in pd.date_range(vs, ts - pd.Timedelta(days=1)).difference(validation_days)],
            "test_missing_days": [str(d.date()) for d in pd.date_range(ts, te).difference(test_days)],
            "source_sha256": sha256(args.data), "source_file": str(args.data.resolve()),
            "script_sha256": hashlib.sha256(source_bytes).hexdigest(), "git_revision": revision,
            "python": platform.python_version(), "platform": platform.platform(),
            "dependencies": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "matplotlib")},
            "daily_features": list(x.columns), "day_label_counts": day_counts,
            "forecast_origin_assumption": "D-1 12:00 Asia/Shanghai; prices of D-1 and earlier assumed published",
            "hour_convention": "source delivery day, end labels 01:00..24:00; pending source verification"}
    save_json(args.output / "metadata.json", meta)
    save_json(args.output / "metrics.json", {"validation": val_scores, "test": test_scores})
    save_json(args.output / "timing.json", timings)
    pd.DataFrame(selection).to_csv(args.output / "selection.csv", index=False)
    pd.DataFrame(groups).to_csv(args.output / "group_metrics.csv", index=False)
    val_frame.to_csv(args.output / "validation_predictions.csv", index=False)
    test_frame.to_csv(args.output / "test_predictions.csv", index=False)
    prices.to_csv(args.output / "hourly_labels.csv", index_label="delivery_date")
    import joblib
    joblib.dump({"models": fitted, "features": list(x.columns), "params": selected}, args.output / "models.joblib")
    plot_results(test_frame, args.output)
    (args.output / "report.md").write_text(report_text(meta, val_scores, test_scores), encoding="utf-8")
    print(json.dumps(test_scores, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/shandong/prices.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baselines.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/baselines_v1")
    run(parser.parse_args())
