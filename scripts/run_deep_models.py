#!/usr/bin/env python3
"""Train/evaluate PyTorch LSTM and TFT on CPU, with auditable weekly refits."""
import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

try:
    from . import run_baselines as baseline
    from .deep_models import (build_sequences, fit_scaler, transform, make_model,
                              PAST_FEATURES, FUTURE_FEATURES)
except ImportError:
    import run_baselines as baseline
    from deep_models import (build_sequences, fit_scaler, transform, make_model,
                             PAST_FEATURES, FUTURE_FEATURES)

ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def predict(model, past, future, scaler, batch_size=64):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(past), batch_size):
            outputs.append(model(past[start:start+batch_size], future[start:start+batch_size]).numpy())
    return np.concatenate(outputs) * scaler["target_std"] + scaler["target_mean"]


def fit(name, config, seed, past, future, target, days, cutoff, output, log,
        epochs=None, validation=None, stage="test"):
    """Fresh model and scaler at each origin. Validation never contributes gradients."""
    started = time.perf_counter()
    train = days < cutoff
    if train.sum() < 60:
        raise ValueError("At least 60 fitting days required")
    scaler = fit_scaler(past[train], future[train], target[train])
    p, f = transform(past[train], future[train], scaler)
    y = torch.from_numpy((target[train] - scaler["target_mean"]) / scaler["target_std"])
    seed_everything(seed)
    model = make_model(name, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    loss_fn = nn.L1Loss()
    generator = torch.Generator().manual_seed(seed)
    best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
    if validation is not None:
        vp, vf = transform(past[validation], future[validation], scaler)
    fit_id = f"{stage}_{name}_seed{seed}_{cutoff.date()}"
    for epoch in range(1, (epochs if epochs is not None else config["max_epochs"]) + 1):
        model.train()
        permutation = torch.randperm(len(y), generator=generator)
        total = 0.
        for idx in permutation.split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(p[idx], f[idx]), y[idx])
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss: {fit_id}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        record = {"fit_id": fit_id, "stage": stage, "model": name, "seed": seed,
                  "cutoff": str(cutoff.date()), "epoch": epoch, "training_days": int(train.sum()),
                  "last_training_target": str(days[train][-1].date()),
                  "train_MAE": total / len(y) * scaler["target_std"],
                  "elapsed_seconds": time.perf_counter() - started}
        if validation is not None:
            val_prediction = predict(model, vp, vf, scaler)
            val_loss = float(np.abs(val_prediction - target[validation]).mean())
            record["validation_MAE"] = val_loss
            if val_loss < best_loss - config["min_delta"]:
                best_loss, best_epoch = val_loss, epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
        log.write(json.dumps(record, allow_nan=False) + "\n")
        log.flush()
        if validation is not None and stale >= config["patience"]:
            break
    if validation is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs
    checkpoint = {"state_dict": model.state_dict(), "scaler": {k: np.asarray(v).tolist() for k, v in scaler.items()},
                  "name": name, "config": config, "seed": seed, "epoch": best_epoch,
                  "last_training_target": str(days[train][-1].date()),
                  "past_features": PAST_FEATURES, "future_features": FUTURE_FEATURES}
    torch.save(checkpoint, output / "checkpoints" / f"{fit_id}.pt")
    audit = {"fit_id": fit_id, "best_epoch": best_epoch, "epochs_run": epoch,
             "last_training_target": str(days[train][-1].date()), "cutoff": str(cutoff.date()),
             "training_days": int(train.sum()), "seconds": time.perf_counter() - started,
             "parameters": sum(p.numel() for p in model.parameters()), "device": "cpu"}
    print(f"{fit_id}: epochs={best_epoch}/{epoch}, fit={audit['seconds']:.1f}s", flush=True)
    return model, scaler, audit


def aligned_baselines(frame, directory, source_hash):
    meta = json.loads((directory / "metadata.json").read_text())
    if meta["source_sha256"] != source_hash:
        raise ValueError("Baseline source hash differs")
    old = pd.read_csv(directory / "test_predictions.csv")
    keys = ["delivery_date", "hour_end"]
    joined = frame.merge(old[keys + ["actual", "WeeklyNaive", "LEAR", "LightGBM"]],
                         on=keys, how="outer", validate="one_to_one", suffixes=("", "_baseline"), indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("Baseline test coverage differs; cannot claim paired comparison")
    for column in ("actual", "WeeklyNaive"):
        np.testing.assert_allclose(joined[column], joined[column + "_baseline"], rtol=1e-6, atol=1e-4)
    return joined.drop(columns=["_merge", "actual_baseline", "WeeklyNaive_baseline"])


def plots(output, test, names):
    os.environ.setdefault("MPLCONFIGDIR", str(output.resolve() / "mpl-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    history = pd.read_json(output / "training.jsonl", lines=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for name, ax in zip(("LSTM", "TFT"), axes):
        for seed, group in history[(history.stage == "validation") & (history.model == name)].groupby("seed"):
            ax.plot(group.epoch, group.train_MAE, linestyle="--", label=f"train {seed}")
            ax.plot(group.epoch, group.validation_MAE, label=f"val {seed}")
        ax.set(title=name, xlabel="Epoch", ylabel="MAE (source price unit)")
        ax.legend(fontsize=8)
    fig.savefig(output / "training_curves.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    axes[1].plot(test.actual.to_numpy()[-168:], color="black", label="Actual")
    for name in names:
        axes[0].plot(test.assign(error=np.abs(test[name] - test.actual)).groupby("hour_end").error.mean(), label=name)
        axes[1].plot(test[name].to_numpy()[-168:], label=name, alpha=.8)
    axes[0].set(xlabel="Delivery hour end", ylabel="MAE", title="Paired test error by hour")
    axes[1].set(xlabel="Hour in final test week", ylabel="Price", title="Final test week")
    for ax in axes:
        ax.legend(ncol=5)
    fig.savefig(output / "test_comparison.png", dpi=160)
    plt.close(fig)


def report(meta, scores, stability):
    c = meta["config"]
    lines = ["# PyTorch CPU：LSTM 与 TFT 实测", "", "本报告由实际训练与预测生成。", "",
             f"- 设备 CPU，PyTorch {meta['dependencies']['torch']}，线程 {c['threads']}；随机种子 {c['seeds']}。",
             f"- 初始训练为 2024 年（扣除首 7 天历史）；验证 {c['validation_start']} 至测试开始前一天；测试 {c['test_start']} 至 {c['test_end']}。",
             "- 输入过去 168 小时价格及日历；未来 24 小时日历、D-1/D-7 同小时价格与前 7 天均值/标准差。供需字段和 ERA5 天气不进入模型。假设 D-1 12:00 已知 D-1 日前价格，发布时间与小时标签口径仍待核验。",
             "- LSTM 为历史编码器与未来协变量解码器。TFT 含变量选择、门控残差、LSTM 编解码、共享 V 的可解释多头因果注意力；单市场无静态变量，省略静态编码器。使用 L1（中位数点预测），未训练概率区间。",
             "- 验证采用固定训练截止时间，逐 epoch 检查完整验证集并早停，选最佳 epoch；验证误差属于模型选择结果，不能当作独立泛化指标，也不能与基线滚动验证直接比较。",
             f"- 进入测试前冻结各模型各种子的 epoch；测试每 {c['retrain_days']} 天从头训练，使用截止前全部有效标签，标准化仅拟合本次训练数据。测试标签不用于早停或调参。测试评估完成后未再次调参。",
             "- 表中 LSTM/TFT 为三个种子的等权预测均值；同时报告每个种子与波动，集成成本高于单模型。传统基线来自既有实测，已核对数据哈希、测试日期、真实标签与周基线。",
             f"- 初始训练 95% 尖峰阈值 {meta['peak_threshold']:.4f}，所有模型同样本评分；价格保留负值和尖峰，不填补标签。", "",
             "| 模型 | 样本 | MAE | RMSE | rMAE | 相对周基线改善 | 负价 F1 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, s in scores.items():
        f1 = f"{s['negative_F1']:.4f}" if s["negative_F1"] is not None else "N/A"
        lines.append(f"| {name} | {s['n']} | {s['MAE']:.3f} | {s['RMSE']:.3f} | {s['rMAE']:.4f} | {s['improvement_pct']:.2f}% | {f1} |")
    lines += ["", "## 种子波动与训练成本", ""]
    for name, s in stability.items():
        lines.append(f"- {name} 单种子测试 MAE 均值±总体标准差：{s['MAE_mean']:.3f} ± {s['MAE_std']:.3f}；范围 {s['MAE_min']:.3f}–{s['MAE_max']:.3f}。")
    best = min((n for n in ("LSTM", "TFT")), key=lambda n: scores[n]["MAE"])
    difference = 100 * (1 - scores[best]["MAE"] / scores["LightGBM"]["MAE"]) if "LightGBM" in scores else None
    lines += [f"- 本轮深度模型中 {best} 集成 MAE 最低。" + (f" 相对 LightGBM 的 MAE 改善 {difference:.2f}%（负数表示退步）。" if difference is not None else ""),
              f"- 全流程耗时 {meta['elapsed_seconds'] / 60:.2f} 分钟；详细各次训练耗时见 timing.json。",
              "", "## 产物与限制", "",
              "- training.jsonl / training.csv：每轮训练 MAE、验证 MAE、耗时、样本数、标签截止时间；training_curves.png 为初始选择阶段曲线。",
              "- checkpoints/：每个模型、种子与训练截止日期的 CPU state_dict、标准化参数、特征与配置；selected_epochs.json 在测试训练前落盘。",
              "- test_predictions.csv / validation_predictions.csv：逐小时预测，包含单种子与集成；metrics.json / group_metrics.csv 为整体与小时、月份、季节、节假日和周末分组指标。",
              "- metadata.json / config_snapshot.json / *_snapshot.py / requirements_snapshot.txt：环境、数据和代码哈希、配置与代码快照。",
              "- tft_interpretability.npz：各种子最后一次重训模型在末个测试日的变量权重及注意力；注意力不等于因果解释。",
              "- 仅半年测试，训练样本不足一年起步；本轮仅一个小型结构配置、三个种子，无供需/天气消融，结果不代表充分调参后的性能或交易收益。",
              "- 架构参考：[TFT 原论文](https://arxiv.org/abs/1912.09363)。", ""]
    return "\n".join(lines)


def run(args):
    started = time.perf_counter()
    config = json.loads(args.config.read_text())
    vs, ts, te = [pd.Timestamp(config[k]) for k in ("validation_start", "test_start", "test_end")]
    if not vs < ts <= te or any(config[k] < 1 for k in ("retrain_days", "max_epochs", "patience", "threads", "batch_size")):
        raise ValueError("Invalid split or training configuration")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Seeds must be nonempty and distinct")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output is nonempty; choose a new --output")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "checkpoints").mkdir()
    torch.set_num_threads(config["threads"])
    torch.set_num_interop_threads(1)
    baseline.save_json(args.output / "config_snapshot.json", config)
    sources = [Path(__file__), Path(__file__).with_name("deep_models.py"), Path(baseline.__file__)]
    for source in sources:
        (args.output / (source.stem + "_snapshot.py")).write_bytes(source.read_bytes())
    (args.output / "requirements_snapshot.txt").write_text(subprocess.check_output(
        [os.sys.executable, "-m", "pip", "freeze"], text=True))
    prices, calendar, _ = baseline.load_hourly(args.data)
    days, past, future, target = build_sequences(prices, calendar, config["history_days"])
    _, y, reference = baseline.build_samples(prices, calendar)
    if not days.equals(y.index):
        raise ValueError("Sequence and baseline sample alignment differs")
    if te > days.max():
        raise ValueError("Test end exceeds valid sample coverage")
    validation = (days >= vs) & (days < ts)
    test = (days >= ts) & (days <= te)
    if not validation.any() or not test.any():
        raise ValueError("Empty validation/test split")
    threshold = float(np.quantile(target[days < vs], .95))
    selected, timings, forecasts = {}, [], {"validation": {}, "test": {}}
    with (args.output / "training.jsonl").open("w") as log:
        # Freeze ALL epoch choices before the first test refit.
        for name in ("LSTM", "TFT"):
            selected[name] = {}
            for seed in config["seeds"]:
                model, scaler, audit = fit(name, config, seed, past, future, target, days, vs,
                                          args.output, log, validation=validation, stage="validation")
                timings.append(audit)
                selected[name][str(seed)] = audit["best_epoch"]
                p, f = transform(past[validation], future[validation], scaler)
                forecasts["validation"][f"{name}_seed{seed}"] = pd.DataFrame(
                    predict(model, p, f, scaler), index=days[validation], columns=y.columns)
        baseline.save_json(args.output / "selected_epochs.json", selected)
        interpretations = {}
        for name in ("LSTM", "TFT"):
            for seed in config["seeds"]:
                predictions = np.empty((test.sum(), 24), dtype=np.float32)
                test_days = days[test]
                last_fit = None
                for i, day in enumerate(test_days):
                    if last_fit is None or (day - last_fit).days >= config["retrain_days"]:
                        model, scaler, audit = fit(name, config, seed, past, future, target, days, day,
                                                  args.output, log, epochs=selected[name][str(seed)])
                        timings.append(audit)
                        last_fit = day
                    idx = days.get_loc(day)
                    p, f = transform(past[idx:idx+1], future[idx:idx+1], scaler)
                    predictions[i] = predict(model, p, f, scaler)[0]
                forecasts["test"][f"{name}_seed{seed}"] = pd.DataFrame(predictions, index=test_days, columns=y.columns)
                if name == "TFT":
                    with torch.inference_mode():
                        _, weights = model(p, f, interpret=True)
                    interpretations.update({f"seed{seed}_{k}": v.numpy() for k, v in weights.items()})
        np.savez_compressed(args.output / "tft_interpretability.npz", **interpretations)
    frames = {}
    for split, mask in (("validation", validation), ("test", test)):
        for name in ("LSTM", "TFT"):
            forecasts[split][name] = sum(forecasts[split][f"{name}_seed{s}"] for s in config["seeds"]) / len(config["seeds"])
        frames[split] = baseline.prediction_frame(y.loc[days[mask]], reference.loc[days[mask]], forecasts[split])
    source_hash = baseline.sha256(args.data)
    if args.baselines is not None:
        frames["test"] = aligned_baselines(frames["test"], args.baselines, source_hash)
    names = ["WeeklyNaive"] + (["LEAR", "LightGBM"] if args.baselines is not None else []) + ["LSTM", "TFT"]
    scores = {split: {name: baseline.metrics(frame.actual, frame[name], frame.WeeklyNaive, threshold)
                      for name in ["WeeklyNaive"] + list(forecasts[split]) +
                      (["LEAR", "LightGBM"] if split == "test" and args.baselines is not None else [])}
              for split, frame in frames.items()}
    stability = {}
    for name in ("LSTM", "TFT"):
        maes = [scores["test"][f"{name}_seed{s}"]["MAE"] for s in config["seeds"]]
        stability[name] = {"MAE_mean": float(np.mean(maes)), "MAE_std": float(np.std(maes)),
                           "MAE_min": float(min(maes)), "MAE_max": float(max(maes))}
    groups = []
    frame = frames["test"]
    dates = pd.to_datetime(frame.delivery_date)
    labels = {"hour": frame.hour_end, "month": dates.dt.strftime("%Y-%m"),
              "season": dates.dt.month.map({12:"winter",1:"winter",2:"winter",3:"spring",4:"spring",5:"spring",6:"summer",7:"summer",8:"summer",9:"autumn",10:"autumn",11:"autumn"}),
              "holiday": dates.map(calendar["是否节假日"]),
              "weekday_or_weekend": dates.dt.dayofweek.map(lambda d: "weekend" if d >= 5 else "weekday")}
    for kind, values in labels.items():
        for label, group in frame.groupby(values):
            for name in names:
                groups.append({"group": kind, "value": label, "model": name,
                               **baseline.metrics(group.actual, group[name], group.WeeklyNaive, threshold)})
    meta = {"config": config, "device": "cpu", "source_sha256": source_hash,
            "script_sha256": {p.name: baseline.sha256(p) for p in sources},
            "python": platform.python_version(), "platform": platform.platform(),
            "dependencies": {p: importlib.metadata.version(p) for p in ("torch", "numpy", "pandas", "matplotlib")},
            "selected_epochs": selected, "peak_threshold": threshold,
            "past_features": PAST_FEATURES, "future_features": FUTURE_FEATURES,
            "test_days": int(test.sum()), "validation_days": int(validation.sum()),
            "test_missing_days": [str(d.date()) for d in pd.date_range(ts, te).difference(days[test])],
            "elapsed_seconds": time.perf_counter() - started}
    for split, frame in frames.items():
        frame.to_csv(args.output / f"{split}_predictions.csv", index=False)
    pd.DataFrame(groups).to_csv(args.output / "group_metrics.csv", index=False)
    pd.read_json(args.output / "training.jsonl", lines=True).to_csv(args.output / "training.csv", index=False)
    baseline.save_json(args.output / "metrics.json", {**scores, "seed_stability": stability})
    baseline.save_json(args.output / "timing.json", timings)
    baseline.save_json(args.output / "metadata.json", meta)
    plots(args.output, frames["test"], names)
    text = report(meta, {n: scores["test"][n] for n in names}, stability)
    (args.output / "report.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/shandong/prices.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/deep_models.json")
    parser.add_argument("--baselines", type=Path, default=ROOT / "outputs/baselines_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/deep_models_v1")
    run(parser.parse_args())
