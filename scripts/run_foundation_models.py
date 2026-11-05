#!/usr/bin/env python3
"""Frozen zero-shot rolling evaluation on the exact existing baseline test rows."""
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

try:
    from . import run_baselines as b
    from .foundation_models import KNOWN_FEATURES, OfficialAdapter, make_input
except ImportError:
    import run_baselines as b
    from foundation_models import KNOWN_FEATURES, OfficialAdapter, make_input

ROOT = Path(__file__).resolve().parents[1]


def prepare_models(config, output):
    from huggingface_hub import HfApi, snapshot_download
    manifest_path = output / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name, repo in config["models"].items():
        if name in manifest:
            continue
        revision = HfApi().model_info(repo).sha
        print(f"Download {name}: {repo}@{revision}", flush=True)
        local = snapshot_download(repo, revision=revision,
                                  allow_patterns=["*.json", "*.yaml", "*.safetensors", "*.ckpt", "LICENSE*", "NOTICE*", "README.md"])
        files = {p.name: b.sha256(p) for p in Path(local).iterdir() if p.is_file()}
        manifest[name] = {"repo": repo, "revision": revision, "local_path": local, "sha256": files}
        b.save_json(manifest_path, manifest)
    return manifest


def paired_reference(prices, calendar, args, config):
    meta = json.loads((args.baselines / "metadata.json").read_text())
    if meta["source_sha256"] != b.sha256(args.data):
        raise ValueError("Baseline source hash mismatch")
    for key in ("test_start", "test_end"):
        if config[key] != meta["config"][key]:
            raise ValueError("Test dates must match saved baseline configuration")
    _, y, ref = b.build_samples(prices, calendar)
    days = y.index[(y.index >= config["test_start"]) & (y.index <= config["test_end"])]
    expected = b.prediction_frame(y.loc[days], ref.loc[days], {})
    old = pd.read_csv(args.baselines / "test_predictions.csv")
    keys = ["delivery_date", "hour_end"]
    paired = expected.merge(old, on=keys, how="outer", validate="one_to_one", indicator=True,
                            suffixes=("", "_saved"))
    if not paired._merge.eq("both").all():
        raise ValueError("Baseline row coverage mismatch")
    for column in ("actual", "WeeklyNaive"):
        np.testing.assert_allclose(paired[column], paired[column + "_saved"], rtol=0, atol=1e-9)
    if not paired.forecast_origin.eq(paired.forecast_origin_saved).all():
        raise ValueError("Baseline origin mismatch")
    frame = paired[keys + ["forecast_origin", "actual", "WeeklyNaive", "LEAR", "LightGBM"]].copy()
    # Include existing deep-model ensembles only when the same data and keys verify.
    deep = ROOT / "outputs/deep_models_v1"
    if (deep / "metadata.json").exists():
        dm = json.loads((deep / "metadata.json").read_text())
        if dm["source_sha256"] == meta["source_sha256"]:
            df = pd.read_csv(deep / "test_predictions.csv")
            check = frame.merge(df[keys + ["actual", "forecast_origin", "LSTM", "TFT"]], on=keys,
                                how="outer", validate="one_to_one", indicator=True, suffixes=("", "_deep"))
            if not check._merge.eq("both").all() or not check.forecast_origin.eq(check.forecast_origin_deep).all():
                raise ValueError("Deep model coverage/origin mismatch")
            np.testing.assert_allclose(check.actual, check.actual_deep, rtol=1e-6, atol=1e-4)
            frame = check.drop(columns=["_merge", "actual_deep", "forecast_origin_deep"])
    return frame, days, meta["peak_threshold"]


def summarize(output, frame, manifest, config, threshold, audit):
    names = [n for n in frame if n not in ("delivery_date", "hour_end", "forecast_origin", "actual")]
    scores = {name: b.metrics(frame.actual, frame[name], frame.WeeklyNaive, threshold) for name in names}
    b.save_json(output / "metrics.json", {"test": scores})
    frame.to_csv(output / "test_predictions.csv", index=False)
    groups = []
    for kind, labels in {"hour": frame.hour_end, "month": frame.delivery_date.str[:7]}.items():
        for label, group in frame.groupby(labels):
            for name in names:
                groups.append({"group": kind, "value": label, "model": name,
                               **b.metrics(group.actual, group[name], group.WeeklyNaive, threshold)})
    pd.DataFrame(groups).to_csv(output / "group_metrics.csv", index=False)
    probabilistic = {}
    for name in config["models"]:
        df = pd.read_csv(output / f"{name}_predictions.csv")
        y = frame.actual.to_numpy()
        q = df[["q10", "q50", "q90"]].to_numpy()
        delta = y[:, None] - q
        probabilistic[name] = {"mean_pinball_q10_q50_q90": float(np.maximum(delta*np.array([.1,.5,.9]), delta*(np.array([.1,.5,.9])-1)).mean()),
                               "coverage_80": float(((y >= q[:, 0]) & (y <= q[:, 2])).mean()),
                               "mean_width_80": float((q[:, 2] - q[:, 0]).mean())}
    b.save_json(output / "probabilistic_metrics.json", probabilistic)
    b.save_json(output / "verification.json", {"paired_rows": len(frame), "days": frame.delivery_date.nunique(),
                "all_finite": bool(np.isfinite(frame[names + ["actual"]]).all().all()),
                "input_audit_rows": len(audit), "max_history_delivery_before_target": all(a["history_end"] < a["forecast_day"] for a in audit),
                "pretraining_contamination_excluded": False})
    lines = ["# 山东日前电价：通用时序基础模型零样本测评", "",
             f"测试 {config['test_start']} 至 {config['test_end']}，{frame.delivery_date.nunique()} 天、{len(frame)} 个原始整点标签。所有模型逐行对齐。", "",
             "## 输入与防泄漏边界", "",
             "- 所有本轮模型固定预训练权重，不微调、不用测试集选择检查点/上下文/特征或校准；点预测统一取中位数。",
             f"- 固定过去 {config['history_days']*24} 小时日前价格，目标日一次预测 24 小时；协变量仅小时、星期、年周期和源日历节假日/休息日。未使用供需字段、实时价格、ERA5 或目标日实际价格。",
             "- 当前业务目标只有日前价格一个变量，实际实验为单目标+多维协变量；不伪造多个预测目标。适配器均使用官方协变量接口。",
             "- 历史窗口按交付日截断至 D-1，保留 01:00..24:00 的源小时末标签。输入无跨日压缩、无标签填补；不裁剪负价和尖峰。",
             "- 禁止不同预测日期之间交叉学习；每模型实测单样本与混入更晚日期历史的批量预测一致，防止批处理产生跨预测时点信息泄漏。",
             "- 与既有基线一致，假设 D-1 12:00 已发布 D-1 完整日前价格。D-1 24:00 是交付结束标记而非发布时间。原始发布时间缺失，因此不是已完全核实可交易时点的回测；日历标记也假设届时已知。",
             "- 预训练权重发布晚于测试期。公开训练语料尚不能证明完全未接触该山东数据或其衍生序列；本轮阻断本地输入、归一化、调参与评分前视，但不能宣称预训练污染已排除。",
             "- 基线为既有监督学习滚动实测，按数据 SHA-256、日期/小时、标签、周基线与预测时点逐项核对后复用；未重新调参。LSTM/TFT（若存在）为既有三种子集成。",
             f"- 尖峰阈值沿用初始训练集 95% 分位 {threshold:.6f}，非测试集估计；价格单位沿用源数据待核验。", "",
             "## 同样本结果", "", "| 模型 | MAE | RMSE | rMAE | 负价 F1 | 尖峰 MAE |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, s in scores.items():
        f1 = f"{s['negative_F1']:.4f}" if s["negative_F1"] is not None else "N/A"
        lines.append(f"| {name} | {s['MAE']:.3f} | {s['RMSE']:.3f} | {s['rMAE']:.4f} | {f1} | {s['peak_MAE']:.3f} |")
    lines += ["", "## 权重与成本", ""]
    for name, entry in manifest.items():
        timing = json.loads((output / f"{name}_timing.json").read_text())
        lines.append(f"- {name}: `{entry['repo']}@{entry['revision']}`；加载 {timing['load_seconds']:.1f}s，完整测试推理 {timing['predict_seconds']:.1f}s，CPU、{config['threads']} 线程。")
    lines += ["", "## 概率输出", "", "| 模型 | 三分位平均 Pinball | 80% 覆盖率 | 平均区间宽度 |", "| --- | ---: | ---: | ---: |"]
    for name, s in probabilistic.items():
        lines.append(f"| {name} | {s['mean_pinball_q10_q50_q90']:.3f} | {s['coverage_80']:.2%} | {s['mean_width_80']:.3f} |")
    best = min(config["models"], key=lambda n: scores[n]["MAE"])
    improvement = 100*(1-scores[best]["MAE"]/scores["LightGBM"]["MAE"])
    lines += ["", f"本轮基础模型中 {best} 的 MAE 最低，相对 LightGBM 的 MAE 改善为 {improvement:.2f}%（负数代表更差）。仅固定一个历史窗口，没有在测试结果之后重新调参；不代表各模型最佳性能或交易收益。",
              "", "复现产物：配置/代码/依赖快照、model_manifest.json 的精确权重 revision 与文件哈希、input_audit.csv 的每日历史截止点、每模型三分位预测、完整对比预测、分组指标及 verification.json。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    os.environ.setdefault("MPLCONFIGDIR", str(output / "mpl-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.bar(names, [scores[n]["MAE"] for n in names])
    ax.set(ylabel="MAE (source price unit)", title=f"Shandong day-ahead: paired test, n={len(frame)}")
    ax.tick_params(axis="x", rotation=25)
    fig.savefig(output / "test_comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps(scores, ensure_ascii=False, indent=2), flush=True)


def run(args):
    import torch
    config = json.loads(args.config.read_text())
    if config["prediction_length"] != 24 or config["history_days"] != 7 or min(config["threads"], config["batch_size"]) < 1:
        raise ValueError("This experiment requires 7-day history, 24-hour forecasts and positive batch/threads")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("Output nonempty; use a new output or --resume")
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(ROOT / "data/model_cache"))
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    prices, calendar, _ = b.load_hourly(args.data)
    frame, days, threshold = paired_reference(prices, calendar, args, config)
    inputs = [make_input(prices, calendar, day, config["history_days"]) for day in days]
    identity = {"config": config, "source_sha256": b.sha256(args.data),
                "baseline_predictions_sha256": b.sha256(args.baselines / "test_predictions.csv"),
                "scripts": {p.name: b.sha256(p) for p in (Path(__file__), Path(__file__).with_name("foundation_models.py"), Path(b.__file__))}}
    deep_prediction = ROOT / "outputs/deep_models_v1/test_predictions.csv"
    if deep_prediction.exists():
        identity["deep_predictions_sha256"] = b.sha256(deep_prediction)
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Resume identity changed; use new output")
    b.save_json(identity_path, identity)
    b.save_json(output / "config_snapshot.json", config)
    for filename in identity["scripts"]:
        (output / filename.replace(".py", "_snapshot.py")).write_bytes(Path(__file__).with_name(filename).read_bytes())
    dependencies = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    b.save_json(output / "metadata.json", {**identity, "python": platform.python_version(), "platform": platform.platform(),
                "device": "cpu", "dependencies": dependencies, "known_features": KNOWN_FEATURES,
                "pretraining_contamination_excluded": False})
    audit = [{"forecast_day": str(i.day.date()), "forecast_origin": (i.day-pd.Timedelta(days=1)).strftime("%Y-%m-%dT12:00:00+08:00"),
              "history_start": str(i.history_start.date()), "history_end": str(i.history_end.date()),
              "historical_price_points": i.target.size, "known_covariates": i.known.shape[0],
              "input_sha256": __import__("hashlib").sha256(i.target.tobytes()+i.known.tobytes()).hexdigest()} for i in inputs]
    pd.DataFrame(audit).to_csv(output / "input_audit.csv", index=False)
    manifest = prepare_models(config, output)
    if args.prepare_only:
        return
    for name in config["models"]:
        path = output / f"{name}_predictions.csv"
        if not (args.resume and path.exists()):
            for filename, digest in manifest[name]["sha256"].items():
                if b.sha256(Path(manifest[name]["local_path"]) / filename) != digest:
                    raise ValueError(f"Model hash mismatch: {name}/{filename}")
            start = time.perf_counter()
            adapter = OfficialAdapter(name, manifest[name]["local_path"])
            loaded = time.perf_counter()
            solo = adapter.predict(inputs[:1])
            mixed = adapter.predict([inputs[0], inputs[-1]])[:1]
            np.testing.assert_allclose(solo, mixed, rtol=1e-4, atol=1e-3,
                                       err_msg="Cross-origin batch leakage or numerical instability")
            b.save_json(output / f"{name}_batch_isolation.json", {
                "passed": True, "max_abs_difference": float(np.abs(solo-mixed).max()),
                "first_origin": str(inputs[0].day.date()), "mixed_later_origin": str(inputs[-1].day.date()),
                "rtol": 1e-4, "atol": 1e-3})
            chunks = []
            for offset in range(0, len(inputs), config["batch_size"]):
                chunks.append(adapter.predict(inputs[offset:offset+config["batch_size"]]))
                print(f"{name}: {min(offset+config['batch_size'], len(inputs))}/{len(inputs)} days", flush=True)
            predicted = np.concatenate(chunks).reshape(-1, 3)
            result = frame[["delivery_date", "hour_end"]].copy()
            result[["q10", "q50", "q90"]] = predicted
            result.to_csv(path.with_suffix(".tmp"), index=False)
            b.save_json(output / f"{name}_timing.json", {"load_seconds": loaded-start, "predict_seconds": time.perf_counter()-loaded})
            path.with_suffix(".tmp").replace(path)
            del adapter
            gc.collect()
        result = pd.read_csv(path)
        if not result[["delivery_date", "hour_end"]].equals(frame[["delivery_date", "hour_end"]]):
            raise ValueError("Saved model predictions misaligned")
        if not np.isfinite(result[["q10", "q50", "q90"]]).all().all():
            raise ValueError("Nonfinite saved predictions")
        frame[name] = result.q50.to_numpy()
    summarize(output, frame, manifest, config, threshold, audit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/shandong/prices.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/foundation_models.json")
    parser.add_argument("--baselines", type=Path, default=ROOT / "outputs/baselines_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/foundation_models_v1")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
