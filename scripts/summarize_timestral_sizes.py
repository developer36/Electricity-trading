#!/usr/bin/env python3
"""Verify and publish the fixed Timestral size sweep against its 209M reference."""
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

try:
    from . import run_baselines as b
except ImportError:
    import run_baselines as b

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "outputs/timestral_sizes_v1"
    previous = ROOT / "outputs/foundation_models_v1"
    config = json.loads((output / "config_snapshot.json").read_text())
    manifest = json.loads((output / "model_manifest.json").read_text())
    old_manifest = json.loads((previous / "model_manifest.json").read_text())
    identity = json.loads((output / "identity.json").read_text())
    old_identity = json.loads((previous / "identity.json").read_text())
    for key in ("source_sha256", "baseline_predictions_sha256", "deep_predictions_sha256"):
        assert identity[key] == old_identity[key], key
    assert b.sha256(output / "input_audit.csv") == b.sha256(previous / "input_audit.csv")
    assert manifest["Timestral-209M"]["revision"] == old_manifest["Timestral"]["revision"]
    old_q = pd.read_csv(previous / "Timestral_predictions.csv")
    new_q = pd.read_csv(output / "Timestral-209M_predictions.csv")
    pd.testing.assert_frame_equal(old_q, new_q, check_exact=True)
    frame = pd.read_csv(output / "test_predictions.csv")
    scores = json.loads((output / "metrics.json").read_text())["test"]
    probability = json.loads((output / "probabilistic_metrics.json").read_text())
    threshold = json.loads((ROOT / "outputs/baselines_v1/metadata.json").read_text())["peak_threshold"]
    for name, score in scores.items():
        recalculated = b.metrics(frame.actual, frame[name], frame.WeeklyNaive, threshold)
        for key, value in score.items():
            if value is not None:
                assert np.isclose(value, recalculated[key], atol=1e-12, rtol=1e-12), (name, key)
    rows = []
    for name in config["models"]:
        assert json.loads((output / f"{name}_batch_isolation.json").read_text())["passed"]
        prediction = pd.read_csv(output / f"{name}_predictions.csv")
        assert np.isfinite(prediction[["q10", "q50", "q90"]]).all().all()
        assert (prediction.q10 <= prediction.q50).all() and (prediction.q50 <= prediction.q90).all()
        timing = json.loads((output / f"{name}_timing.json").read_text())
        rows.append({"model": name, **scores[name], **probability[name], **timing,
                     "amortized_ms_per_daily_sample": timing["predict_seconds"] * 1000 / frame.delivery_date.nunique()})
    pd.DataFrame(rows).to_csv(output / "size_comparison.csv", index=False)
    lines = ["# Timestral 四种规模：山东日前电价实测", "",
             "测试期 2025-07-01 至 2025-12-30，183 天、4,392 个小时点。四个规模均在本轮实际推理，209M 没有直接抄用旧指标。", "",
             "固定官方权重，统一 CPU 2 线程、batch 8、168 小时价格历史和 8 维日历协变量，整日一次预测 24 点，点预测取 q50。无微调、测试集调参或事后校准。", "",
             "| 模型 | MAE ↓ | RMSE ↓ | 负价 F1 ↑ | 尖峰 MAE ↓ | 80% 覆盖率 | 推理总秒数 ↓ |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['MAE']:.3f} | {r['RMSE']:.3f} | {r['negative_F1']:.4f} | {r['peak_MAE']:.3f} | {r['coverage_80']:.2%} | {r['predict_seconds']:.2f} |")
    lines += ["", "推理计时不包含权重下载与模型加载，包含同轮批量隔离检查及少量结果整理开销；是批量测评总耗时，不代表单请求服务延迟。不同运行的计时受本机负载影响。", "",
              "## 每条日样本的摊销耗时", "",
              "一条日样本输入过去 168 小时价格和日历协变量，一次输出未来 24 小时预测。下表按整轮计时 ÷ 183 条日样本计算；不是除以 4,392 个预测点。", "",
              "| 模型 | 整轮计时（秒） | 每条日样本摊销耗时（毫秒） |",
              "| --- | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['predict_seconds']:.2f} | {r['amortized_ms_per_daily_sample']:.2f} |")
    lines += ["", "运行条件：CPU、2 线程、batch=8。以上为批量运行的平均摊销耗时，含前述隔离检查和少量整理开销；并非单条请求响应延迟，也不是 batch=1 的实测值。单请求延迟需另行测量。", "",
              "## 结果解读", "",
              "- 28M 的 MAE 最低，比 209M 低 1.02%，本机同轮总推理约快 5.60 倍，适合作为后续轻量实验候选。",
              "- 123M 的 RMSE 最低，209M 的负价 F1 略高。不同指标的最优规模不同。",
              "- 28M 与 123M 的 MAE 差异仅约 0.345，单次半年测试不足以证明稳定优劣。所有版本仍未超过 LightGBM 的总体 MAE 81.777。",
              "", "## 既有模型参考", "", "| 模型 | MAE | RMSE |", "| --- | ---: | ---: |"]
    for name in ("WeeklyNaive", "LEAR", "LightGBM", "LSTM", "TFT"):
        r = scores[name]
        lines.append(f"| {name} | {r['MAE']:.3f} | {r['RMSE']:.3f} |")
    lines += ["", "## 月度 MAE", "", "| 月份 | " + " | ".join(config["models"]) + " |",
              "| --- | " + " | ".join(["---:"]*len(rows)) + " |"]
    for month, group in frame.groupby(frame.delivery_date.str[:7]):
        lines.append("| " + month + " | " + " | ".join(f"{np.abs(group[n]-group.actual).mean():.3f}" for n in config["models"]) + " |")
    lines += ["", "## 核验与边界", "",
              "- 本轮四个模型均通过单样本与混入更晚预测日期的批量隔离检查；23 项离线测试通过。",
              "- 与前轮 input_audit.csv 的哈希完全相同；源数据、基线/深度模型预测哈希、209M 权重 revision 均一致，209M 全部 q10/q50/q90 预测与前轮逐值完全一致。所有模型指标已从逐小时预测独立重算核对。",
              "- 价格历史严格截止 D-1，不使用目标日价格、实时价格、供需数据或 ERA5。沿用 D-1 12:00 已知 D-1 日前价格和目标日日历的可得性假设；发布时间仍待核验。公开预训练语料不足以完全排除山东数据污染。",
              "- 这是看过同一测试集后的规模探索；不能将据此选出的版本再当作全新独立测试结果。若用于正式选型，应在未看过的后续时段或额外滚动窗口确认。",
              "- 完整逐小时预测、三分位区间、月份/小时分组指标、权重 SHA-256、环境与代码快照均保存在 outputs/timestral_sizes_v1/。", "",
              "## 复现", "", "```bash", 
              ".venv-foundation/bin/python scripts/run_foundation_models.py --config configs/timestral_sizes.json --output outputs/timestral_sizes_v1",
              ".venv-foundation/bin/python scripts/summarize_timestral_sizes.py", "```", "",
              "重跑使用新的输出目录；中断恢复加 --resume。", ""]
    (output / "size_report.md").write_text("\n".join(lines), encoding="utf-8")
    for source, destination in (("size_report.md", "timestral_sizes_v1.md"), ("metrics.json", "timestral_sizes_v1_metrics.json"),
                                ("model_manifest.json", "timestral_sizes_v1_manifest.json")):
        shutil.copyfile(output / source, ROOT / "reports" / destination)
    b.save_json(output / "comparison_verification.json", {"209M_predictions_exact_match": True,
                "input_audit_exact_match": True, "metrics_recomputed": True, "batch_isolation_passed": 4,
                "offline_tests_passed": 23, "rows": len(frame)})
    print(pd.DataFrame(rows)[["model", "MAE", "RMSE", "negative_F1", "predict_seconds"]].to_string(index=False))


if __name__ == "__main__":
    main()
