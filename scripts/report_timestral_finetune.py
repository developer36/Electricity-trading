#!/usr/bin/env python3
"""Recompute metrics and publish the Timestral adaptation experiment."""
import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

try:
    from . import run_baselines as b
except ImportError:
    import run_baselines as b

ROOT=Path(__file__).resolve().parents[1]


def publish(output):
    meta=json.loads((output/'metadata.json').read_text())
    c=meta['identity']['config']
    scores=json.loads((output/'metrics.json').read_text())['test']
    probabilities=json.loads((output/'probabilistic_metrics.json').read_text())
    timing=json.loads((output/'timing.json').read_text())
    manifest=json.loads((output/'model_manifest.json').read_text())
    selected=json.loads((output/'selected_epochs.json').read_text())
    confirmation=json.loads((output/'confirmation_metrics.json').read_text())
    verification=json.loads((output/'verification.json').read_text())
    assert all(d['passed'] for d in verification['batch_isolation_diagnostics'].values())
    meta.setdefault('finalization_scripts',{})[Path(__file__).name]=b.sha256(Path(__file__))
    shutil.copyfile(Path(__file__),output/f'{Path(__file__).stem}_snapshot.py')
    b.save_json(output/'metadata.json',meta)
    frame=pd.read_csv(output/'test_predictions.csv')
    # Export float32 predictions through float64 so pandas preserves their exact
    # numerical values in CSV instead of shortening them to float32 decimals.
    precise={}
    for n in ['ZeroShot']+[f'{mode}_seed{s}' for mode in ('Head','LoRA') for s in c['seeds']]:
        precise[n]=np.load(output/f'{n}_quantiles.npz')['quantiles']
    for mode in ('Head','LoRA'):
        precise[mode]=np.mean([precise[f'{mode}_seed{s}'] for s in c['seeds']],axis=0)
    for n,q in precise.items():
        frame[n]=q[...,10].ravel().astype(np.float64)
        pd.DataFrame({'delivery_date':frame.delivery_date,'hour_end':frame.hour_end,
                      **{label:q[...,idx].ravel().astype(np.float64) for label,idx in [('q10',2),('q50',10),('q90',18)]}}).to_csv(output/f'{n}_predictions.csv',index=False)
    frame.to_csv(output/'test_predictions.csv',index=False)
    fits=pd.DataFrame(timing['fits'])
    log_paths=[output/'training.jsonl']+sorted(output.glob('worker_*/training.jsonl'))
    all_epochs=pd.concat([pd.read_json(path,lines=True) for path in log_paths],ignore_index=True)
    # Main records win if a worker finished a deterministic duplicate in parallel.
    all_epochs=all_epochs.drop_duplicates(['fit_id','epoch'],keep='first').sort_values(['fit_id','epoch'])
    counts=all_epochs.groupby('fit_id').size()
    for audit in timing['fits']:
        assert counts[audit['fit_id']]==audit['epochs_run'],audit['fit_id']
    all_epochs.to_json(output/'training_all.jsonl',orient='records',lines=True,force_ascii=False,double_precision=15)
    worker_records=[]
    for worker in sorted(output.glob('worker_*')):
        if not worker.is_dir():
            continue
        assert json.loads((worker/'identity.json').read_text())==meta['identity']
        worker_records.append({'directory':worker.name,'orchestration_sha256':b.sha256(worker/'worker_snapshot.py'),
                               'timing':json.loads((worker/'worker_timing.json').read_text())})
    b.save_json(output/'parallel_workers.json',worker_records)
    assert len(frame)==4392 and frame[['delivery_date','hour_end']].drop_duplicates().shape[0]==4392
    assert (fits.last_training_target<fits.cutoff).all()
    expected_cutoffs=set(pd.date_range(c['test_start'],c['test_end'],freq=f"{c['retrain_days']}D").strftime('%Y-%m-%d'))
    for mode in ('Head','LoRA'):
        for seed in c['seeds']:
            seed_fits=fits[(fits.stage=='test')&(fits['mode']==mode)&(fits.seed==seed)]
            assert len(seed_fits)==len(expected_cutoffs) and set(seed_fits.cutoff)==expected_cutoffs
    for name,s in scores.items():
        rec=b.metrics(frame.actual,frame[name],frame.WeeklyNaive,meta['peak_threshold'])
        for key,value in s.items():
            if value is not None:
                assert np.isclose(value,rec[key],rtol=1e-12,atol=1e-12),(name,key)
    # The executed snapshots, datasets and base weights must still match recorded hashes.
    for name,digest in meta['identity']['scripts'].items():
        assert b.sha256(output/(Path(name).stem+'_snapshot.py'))==digest,name
    assert b.sha256(ROOT/'data/raw/shandong/prices.csv')==meta['identity']['source_sha256']
    for name,digest in manifest['sha256'].items():
        assert b.sha256(Path(manifest['local_path'])/name)==digest,name
    levels=json.loads((Path(manifest['local_path'])/'config.json').read_text())['quantile_knots']
    b.save_json(output/'quantile_levels.json',levels)
    previous=pd.read_csv(ROOT/'outputs/timestral_sizes_v1/Timestral-28M_predictions.csv')
    pd.testing.assert_frame_equal(previous[['delivery_date','hour_end']],frame[['delivery_date','hour_end']])
    zero=np.load(output/'ZeroShot_quantiles.npz')['quantiles']
    postprocess={}
    for column,level in [('q10',.1),('q50',.5),('q90',.9)]:
        difference=np.abs(previous[column].to_numpy()-zero[...,levels.index(level)].ravel())
        postprocess[column]={'max_abs_difference':float(difference.max()),'changed_above_1e-3':int((difference>1e-3).sum())}
    b.save_json(output/'postprocess_comparison.json',postprocess)
    # Independently replay the first and final daily prediction from their exact
    # historical checkpoints; future cached checkpoints must never be substituted.
    import torch
    try:
        from . import run_timestral_finetune as r
        from .foundation_models import make_input
    except ImportError:
        import run_timestral_finetune as r
        from foundation_models import make_input
    torch.set_num_threads(c['threads'])
    prices,calendar,_=b.load_hourly(ROOT/'data/raw/shandong/prices.csv')
    replay_checks=0
    for mode in ('Head','LoRA'):
        for seed in c['seeds']:
            stored=np.load(output/f'{mode}_seed{seed}_quantiles.npz')['quantiles']
            for position in (0,182):
                day=pd.Timestamp(frame.delivery_date.iloc[position*24])
                candidates=fits[(fits.stage=='test')&(fits['mode']==mode)&(fits.seed==seed)&(fits.cutoff<=str(day.date()))]
                cutoff=candidates.cutoff.max()
                saved=torch.load(output/'checkpoints'/f'test_{mode}_seed{seed}_{cutoff}.pt',map_location='cpu',weights_only=False)
                assert saved['audit']['last_training_target']<str(day.date())
                model=r.new_model(manifest['local_path'],mode,seed,c)
                model.load_trainable(saved['trainable_state'])
                item=make_input(prices,calendar,day)
                replay=r.predict(model,torch.from_numpy(item.target[None]),torch.from_numpy(item.known[None]))[0]
                np.testing.assert_allclose(replay,stored[position],rtol=1e-4,atol=1e-3)
                replay_checks+=1
                del model
    names=['WeeklyNaive','LEAR','LightGBM','LSTM','TFT','ZeroShot','Head','LoRA']
    lines=['# Timestral-28M：预测头与 LoRA 微调实测','',
           '本报告由实际训练、推理及独立指标重算生成。测试 2025-07-01 至 2025-12-30，共 183 天、4,392 个原始小时标签；基线与模型逐行对齐。','',
           '## 固定实验设置','',
           f"- 官方权重 `{manifest['repo']}@{manifest['revision']}`，完整权重 SHA-256 见 model_manifest.json。CPU、{c['threads']} 线程；训练 batch={c['train_batch_size']}，预测 batch={c['predict_batch_size']}，种子 {c['seeds']}。",
           '- 每个日样本输入过去 168 小时日前价格、8 维日历协变量，一次预测未来 24 小时；业务目标只有价格一个变量。供需、实时价格与 ERA5 均未使用。',
           '- 对齐官方可微 raw 预测路径，保留符号对称平均；统一关闭非负截断 clip_nonnegative=False 和极端值 cap。原生 21 个非交叉分位数，训练为原始价格单位上的平均 Pinball Loss；只对有效的 24 小时目标计分，不对补齐至 32 的位置制造标签。',
           '- 这是本地任务级微调，不是官方 Forge 训练流水线复刻；不额外监督历史窗口重构。',
           '- Head：冻结主干，仅更新预测头 1,880,064 个参数。LoRA：冻结原权重，在全部时间/变量注意力的 wQKV、wO 注入 rank=8、alpha=16，更新 276,480 个参数；不额外解冻预测头。',
           '- 两路线固定 AdamW，学习率 1e-4、weight_decay=0.01、梯度裁剪 1.0。没有在测试结果之后改学习率、特征、epoch、rank 或回放规则。',
           '', '## 时间划分与更新','',
           '- 初始训练：2024 年 359 个有效日样本；2025 年 1–3 月 90 天只用于验证 MAE 早停（最多 20 epoch，patience=3，min_delta=0.01），选择种子固定 42。',
           f"- 在产生任何测试预测前冻结轮数：Head={selected['Head']}，LoRA={selected['LoRA']}。这两个路线均保留，不根据测试挑选或丢弃较差种子。",
           '- 2025 年 4–6 月 91 天为确认区间：每条路线、每个种子从官方初始权重出发，用截至 3 月底全部已到达标签微调已选轮数，然后整个确认区间权重不更新。确认指标仅报告，不重新选参。',
           '- 测试开始时，三个种子各自从官方初始权重出发，用截至 6 月底全部 540 个有效日样本训练已选轮数；之后每 7 天在上一微调权重上继续训练，回放最近 56 个日历日、固定 1 epoch。每次重置 AdamW 状态，保留模型权重。',
           '- 此为固定计算预算的增量更新：与既有 LEAR/LightGBM/LSTM/TFT 的每周全历史重训方式不同，但数据可得性与评分样本相同；训练计算成本不相同。',
           '- Head/LoRA 主表为三个种子预测的等权平均；分位数也逐位置平均，属于分位数平均，不是概率混合分布。单种子结果另列。ZeroShot 仅一个固定官方模型。',
           '', '## 防止未来信息泄漏','',
           '- 构造历史输入时按交付日截断至 D-1，保留 01:00..24:00 源小时末标签。目标日实际价格不参与输入或归一化，历史缺失不填补、不压缩时间轴。',
           '- 标准化沿用模型内部历史观测统计；已知未来日历使用预测区间日历属于允许输入。训练标签只来自 cutoff 之前的交付日，更新样本选择与计分相互隔离。',
           '- 继承现有回测假设：D-1 12:00 已知 D-1 完整日前价格和目标日日历；源数据没有实际发布时间，仍需核验，不能宣称已完全验证的可交易时点回测。',
           '- 28M 和同一测试集在前轮已经被比较过，本轮是后续探索性研究；预训练权重晚于测试期，公开语料也不足以完全排除山东数据污染。正式独立确认需要新的未见时段。',
           '', '## 同样本结果','',
           '| 模型 | MAE ↓ | RMSE ↓ | rMAE ↓ | 负价 F1 ↑ | 尖峰 MAE ↓ |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for n in names:
        s=scores[n]
        f1=f"{s['negative_F1']:.4f}" if s['negative_F1'] is not None else 'N/A'
        lines.append(f"| {n} | {s['MAE']:.3f} | {s['RMSE']:.3f} | {s['rMAE']:.4f} | {f1} | {s['peak_MAE']:.3f} |")
    lines+=['','初始训练集 95% 分位尖峰阈值固定为 '+f"{meta['peak_threshold']:.6f}"+'，不在测试集重新估计；价格单位沿用源数据待核验。LSTM/TFT 是既有三个种子集成，所有既有预测经源数据哈希、真实标签、预测时点与日期/小时键核对后复用。','']
    for n in ('Head','LoRA'):
        z=100*(1-scores[n]['MAE']/scores['ZeroShot']['MAE'])
        g=100*(1-scores[n]['MAE']/scores['LightGBM']['MAE'])
        lines.append(f"- {n} 相对同设置 ZeroShot 的 MAE 改善 {z:.2f}%，相对 LightGBM 改善 {g:.2f}%（负数表示退步）。")
    lines+=['','## 负价识别','','| 模型 | 负价小时 MAE ↓ | 负价 Precision ↑ | 负价 Recall ↑ | 负价 F1 ↑ |',
            '| --- | ---: | ---: | ---: | ---: |']
    for n in ('LightGBM','ZeroShot','Head','LoRA'):
        s=scores[n]
        values=[f"{s[k]:.4f}" if s[k] is not None else 'N/A' for k in ('negative_precision','negative_recall','negative_F1')]
        lines.append(f"| {n} | {s['negative_MAE']:.3f} | "+' | '.join(values)+' |')
    lines+=['','早停目标为总体 MAE，并未按负价分类指标选参；总体误差与事件识别需分别判断。']
    lines+=['','## 单种子与确认结果','','| 路线 | 种子 | 确认 MAE | 测试 MAE | 测试 RMSE |','| --- | ---: | ---: | ---: | ---: |']
    stability={}
    for mode in ('Head','LoRA'):
        values=[]
        for seed in c['seeds']:
            n=f'{mode}_seed{seed}';s=scores[n];values.append(s['MAE'])
            lines.append(f"| {mode} | {seed} | {confirmation[n]['MAE']:.3f} | {s['MAE']:.3f} | {s['RMSE']:.3f} |")
        stability[mode]={'mean_MAE':float(np.mean(values)),'std_MAE':float(np.std(values))}
    for mode,s in stability.items():
        lines += ['',f"{mode} 单种子测试 MAE 均值±总体标准差：{s['mean_MAE']:.3f} ± {s['std_MAE']:.3f}。"]
    lines+=['','## 概率预测','','| 模型 | 21 分位平均 Pinball ↓ | 80% 覆盖率 | 平均区间宽度 |','| --- | ---: | ---: | ---: |']
    for n in ('ZeroShot','Head','LoRA'):
        p=probabilities[n]
        lines.append(f"| {n} | {p['mean_pinball_21']:.3f} | {p['coverage_80']:.2%} | {p['width_80']:.3f} |")
    lines+=['','## 计算成本','','| 路线 | 选轮数训练秒数 | 确认训练秒数（三种子） | 测试训练秒数（三种子） | 测试预测秒数（三种子） |','| --- | ---: | ---: | ---: | ---: |']
    for mode in ('Head','LoRA'):
        f=fits[fits['mode']==mode]
        lines.append(f"| {mode} | {f.loc[f.stage=='selection','seconds'].sum():.2f} | {f.loc[f.stage=='confirmation','seconds'].sum():.2f} | {f.loc[f.stage=='test','seconds'].sum():.2f} | {f.loc[f.stage=='test','predict_seconds'].sum():.2f} |")
    lines += ['',f"- ZeroShot 完整测试预测 {timing['zero_predict_seconds']:.2f} 秒；除以 183 为 {timing['zero_predict_seconds']/183*1000:.2f} ms/日样本的批量摊销值，不代表单请求延迟。",
              '- LoRA 推理尚未合并低秩权重，因此耗时包含额外矩阵计算；微调模型主结果采用三个种子集成，不能把集成总耗时当单模型延迟。',
              '- 独立进程预计算 Head 三个种子的测试拟合、LoRA 种子 43/44 的确认与测试拟合，每进程 2 线程，主进程按完全相同的训练身份复用已完成检查点。最多同时五个进程，训练拟合秒数之和包含并行、不能与墙钟相加。各进程日志保留在 worker_*/。最终推理计时在训练进程全部退出后，单进程、2 线程重放时测量。',
              f"- 从身份记录落盘到最终汇总的墙钟 {timing['elapsed_seconds']/60:.2f} 分钟，包含末尾核验与恢复，不含本报告生成。详细每次训练、样本截止和推理成本见 fit_audit.csv、timing.json。",'',
              '## 历史报告口径更正','',
              '- 前轮 Timestral 规模比较实际使用了官方默认 clip_nonnegative=True：历史全非负时将负预测截为 0。之前“不裁剪负价”的概括不够准确；原始真实标签未被裁剪，但模型预测存在此条件后处理。',
              '- 本轮明确关闭该开关并重跑 ZeroShot，微调收益只相对此同设置参照计算；没有把后处理差异算成微调收益。旧表中 28M 的 MAE 85.594 不能直接作为本轮微调收益的分母。','',
              f"- 实测本轮与旧版 q50 最大绝对差为 {postprocess['q50']['max_abs_difference']:.8f}（舍入量级），因此两轮 MAE 显示到三位小数相同；q10 有 {postprocess['q10']['changed_above_1e-3']} 点变化超过 0.001，最大 {postprocess['q10']['max_abs_difference']:.3f}。开关对本轮中位数几乎无影响，但对少量下分位区间有影响。",'',
              '## 验证与产物','',
              '- 可微路径与官方未截断预测逐值一致，LoRA 初始输出与原模型一致；优化器仅修改授权参数；修改 cutoff 起及之后的标签不改变本次训练结果；检查点恢复能逐值复现预测。',
              '- 每个模型的批量任务均按日隔离；完整 fit_audit 的训练标签严格早于 cutoff。生成报告时独立重算全部指标，并校验权重和执行快照 SHA-256。',
              '- 原始核验在 LoRA seed43 上发现 B=1 与 B=2 的 float32 预测有少量差异超出 0.001 容差。诊断表明固定 B=2、只替换另一日输入时首样本预测不变；float64 跨批大小计算也通过严格检查，因此是跨批大小的 float32 舍入效应，不能据此认定信息泄漏。没有修改训练、选参或原有预测来消除差异。',
              f"- 七个模型/种子的核验结果：float32 跨批大小最大差 {max(d['fp32_cross_batch_max_abs'] for d in verification['batch_isolation_diagnostics'].values()):.8f}；等批大小输入扰动最大差 {max(d['fp32_fixed_batch_intervention_max_abs'] for d in verification['batch_isolation_diagnostics'].values()):.8g}；float64 跨批大小最大差 {max(d['fp64_cross_batch_max_abs'] for d in verification['batch_isolation_diagnostics'].values()):.8g}。详见 verification.json。",
              '- 另从每个路线、每个种子的首日与末日历史检查点独立回放，共 12 个日预测，与已保存原生分位数逐项核对，防止并行缓存误用未来检查点。',
              '- tests.log 记录 30 项离线测试通过，包含拟合后 LoRA 的舍入差异回归检查与真实跨样本泄漏反例；training_all.jsonl 合并主进程与独立种子的逐 epoch 日志（按 fit_id、epoch 去重），原始日志保留在 training.jsonl 与 worker_*/training.jsonl；checkpoints/ 保存可训练权重（配合固定官方权重复原）。',
              '- test_predictions.csv 保存全部单种子/集成预测；*_quantiles.npz 保存原生 21 分位结果，group_metrics.csv 为小时/月分组；selection/confirmation 产物保留全部路线和种子。','',
              '## 复现','', '```bash',
              '.venv-foundation/bin/python scripts/run_timestral_finetune.py --output outputs/timestral_finetune_repeat',
              '# 中断后继续，代码、源数据和配置必须与身份记录一致',
              '.venv-foundation/bin/python scripts/run_timestral_finetune.py --output outputs/timestral_finetune_repeat --resume',
              '.venv-foundation/bin/python scripts/report_timestral_finetune.py --output outputs/timestral_finetune_repeat',
              '# 从已完成的固定历史检查点重新核验并汇总，不重新训练',
              '.venv-foundation/bin/python scripts/finalize_timestral_finetune.py',
              '.venv-foundation/bin/python scripts/report_timestral_finetune.py','```','']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    b.save_json(output/'seed_stability.json',stability)
    verification['metrics_independently_recomputed']=True
    verification['weights_and_source_snapshots_verified']=True
    verification['historical_checkpoint_prediction_replays']=replay_checks
    verification['fit_count']=len(fits)
    verification['epoch_records_reconciled']=len(all_epochs)
    b.save_json(output/'verification.json',verification)
    for source,destination in [('report.md','timestral_finetune_v1.md'),('metrics.json','timestral_finetune_v1_metrics.json'),('model_manifest.json','timestral_finetune_v1_manifest.json')]:
        shutil.copyfile(output/source,ROOT/'reports'/destination)
    os.environ.setdefault('MPLCONFIGDIR',str(output/'mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
    ax.bar(names,[scores[n]['MAE'] for n in names])
    ax.set(ylabel='MAE (source price unit)',title='Timestral-28M adaptation: paired test, n=4392')
    ax.tick_params(axis='x',rotation=25)
    fig.savefig(output/'test_comparison.png',dpi=160)
    plt.close(fig)
    history=all_epochs
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for mode,ax in zip(('Head','LoRA'),axes):
        h=history[(history.stage=='selection')&(history['mode']==mode)]
        ax.plot(h.epoch,h.validation_MAE,marker='o')
        ax.axvline(selected[mode],linestyle='--',color='gray')
        ax.set(title=mode,xlabel='Epoch',ylabel='Jan-Mar validation MAE')
    fig.savefig(output/'selection_curves.png',dpi=160)
    plt.close(fig)
    readme=ROOT/'README.md'
    text=readme.read_text()
    marker='### 4.6 Timestral-28M：预测头与 LoRA 微调'
    section=[marker,'',
             '实现见 [timestral_finetune.py](scripts/timestral_finetune.py)，入口见 [run_timestral_finetune.py](scripts/run_timestral_finetune.py)，固定配置见 [timestral_finetune.json](configs/timestral_finetune.json)。复用独立 `.venv-foundation` 和已下载的 28M 权重。','',
             f"预测头训练 1,880,064 个参数；LoRA 在时间/变量注意力的 wQKV、wO 注入 rank=8、alpha=16，训练 276,480 个参数，其余冻结。统一输入 168 小时价格历史和 8 维日历协变量，以原生 21 分位 Pinball Loss 监督未来 24 点，关闭非负截断与极端值 cap，保留符号对称平均。",
             '',f"2024 年 359 日初始训练，2025 年 1–3 月 90 日选轮数（Head {selected['Head']}、LoRA {selected['LoRA']}），4–6 月 91 日确认但不改方案。测试起点使用全部已到达的 540 日历史；之后每 7 日继续训练，最近 56 个日历日回放 1 epoch。每次训练严格使用截止日之前的标签；增量回放与既有基线全历史重训的方式不同。",'',
             '| 模型 | MAE ↓ | RMSE ↓ | 负价 F1 ↑ |','| --- | ---: | ---: | ---: |']
    for n in names:
        s=scores[n]
        f1=f"{s['negative_F1']:.4f}" if s['negative_F1'] is not None else 'N/A'
        section.append(f"| {n} | {s['MAE']:.3f} | {s['RMSE']:.3f} | {f1} |")
    section+=['','Head/LoRA 为 42、43、44 三个种子的等权平均；单种子、确认指标、概率区间与完整训练/推理成本见[微调实测报告](reports/timestral_finetune_v1.md)。30 项离线测试通过；报告生成时重新计算全部指标，并从首末日的 12 个历史检查点独立回放预测。','',
              '```bash','.venv-foundation/bin/python scripts/run_timestral_finetune.py --output outputs/timestral_finetune_repeat',
              '# 中断恢复：配置、源数据、执行代码必须一致','.venv-foundation/bin/python scripts/run_timestral_finetune.py --output outputs/timestral_finetune_repeat --resume',
              '.venv-foundation/bin/python scripts/report_timestral_finetune.py --output outputs/timestral_finetune_repeat','```','',
              '完整产物保存于 `outputs/timestral_finetune_v1/`，包括逐小时预测、21 分位数组、170 个训练截止点检查点、训练日志及审计信息。源价格发布时间与日历提前可得性仍依赖前述假设，预训练污染未完全排除；同一测试集已参与先前规模探索，正式独立确认需要新的未见区间。','',
              '**前轮口径更正：**Timestral 先前规模测评沿用了默认条件非负截断（历史全非负时将负预测截为 0），此前“不裁剪负价”的概括仅适用于原始标签。当前零样本和两条微调路线均关闭该后处理，收益相对此同设置零样本计算。','']
    if marker in text:
        before,after=text.split(marker,1)
        _,tail=after.split('## 5. 评估与复现',1)
        text=before+'\n'.join(section)+'\n## 5. 评估与复现'+tail
    else:
        text=text.replace('## 5. 评估与复现','\n'.join(section)+'\n## 5. 评估与复现')
    readme.write_text(text,encoding='utf-8')
    print(pd.DataFrame({n:scores[n] for n in names}).T[['MAE','RMSE','negative_F1']].to_string())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/timestral_finetune_v1')
    publish(parser.parse_args().output.resolve())
