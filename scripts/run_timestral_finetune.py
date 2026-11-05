#!/usr/bin/env python3
"""Auditable head/LoRA adaptation; all selection precedes the test interval."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import pandas as pd
import torch

try:
    from . import run_baselines as b
    from .run_foundation_models import paired_reference
    from .timestral_finetune import AdaptedTimestral, build_dataset, pinball, predict
    from .timestral_isolation import check_isolation
except ImportError:
    import run_baselines as b
    from run_foundation_models import paired_reference
    from timestral_finetune import AdaptedTimestral, build_dataset, pinball, predict
    from timestral_isolation import check_isolation

ROOT = Path(__file__).resolve().parents[1]


def training_indices(days, cutoff, replay_days=None):
    mask = days < cutoff
    if replay_days is not None:
        mask &= days >= cutoff - pd.Timedelta(days=replay_days)
    idx = np.flatnonzero(mask)
    if not len(idx) or not (days[idx] < cutoff).all():
        raise ValueError('Empty or leaking training set')
    return idx


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def fit(model, mode, seed, days, data, cutoff, config, output, log,
        stage, epochs=None, validation=None, warm=False):
    context, known, target = data
    idx = training_indices(days, cutoff, config['replay_days'] if warm else None)
    ident = f'{stage}_{mode}_seed{seed}_{cutoff.date()}'
    path = output / 'checkpoints' / f'{ident}.pt'
    if path.exists():
        saved = torch.load(path, weights_only=False, map_location='cpu')
        if saved['audit']['training_days'] != len(idx) or saved['audit']['last_training_target'] != str(days[idx[-1]].date()):
            raise ValueError('Checkpoint cutoff mismatch')
        model.load_trainable(saved['trainable_state'])
        return model, saved['audit']
    seed_all(seed)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config['learning_rates'][mode], weight_decay=config['weight_decay'])
    generator = torch.Generator().manual_seed(seed)
    best, best_epoch, stale, best_state = float('inf'), 0, 0, None
    started = time.perf_counter()
    for epoch in range(1, (epochs if epochs is not None else config['max_epochs']) + 1):
        model.train()
        shuffled = torch.as_tensor(idx)[torch.randperm(len(idx), generator=generator)]
        total = 0.
        for batch in shuffled.split(config['train_batch_size']):
            optimizer.zero_grad(set_to_none=True)
            q = model(context[batch], known[batch])
            loss = pinball(q, target[batch], model.levels)
            if not torch.isfinite(loss):
                raise ValueError(f'Nonfinite loss: {ident}')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, config['gradient_clip'])
            if not torch.isfinite(norm):
                raise ValueError(f'Nonfinite gradients: {ident}')
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        record = {'fit_id': ident, 'stage': stage, 'mode': mode, 'seed': seed, 'epoch': epoch,
                  'cutoff': str(cutoff.date()), 'last_training_target': str(days[idx[-1]].date()),
                  'first_training_target': str(days[idx[0]].date()), 'training_days': len(idx),
                  'warm_update': warm, 'train_pinball': total / len(idx), 'elapsed_seconds': time.perf_counter()-started}
        if validation is not None:
            q = predict(model, context[validation], known[validation], config['predict_batch_size'])
            score = float(np.abs(q[..., model.levels.index(.5)] - target[validation].numpy()).mean())
            record['validation_MAE'] = score
            if score < best - config['min_delta']:
                best, best_epoch, best_state, stale = score, epoch, model.trainable_state(), 0
            else:
                stale += 1
        log.write(json.dumps(record, allow_nan=False) + '\n')
        log.flush()
        print(f"{ident} epoch {epoch}: pinball={record['train_pinball']:.3f}" +
              (f' val_MAE={score:.3f}' if validation is not None else ''), flush=True)
        if validation is not None and stale >= config['patience']:
            break
    if validation is not None:
        model.load_trainable(best_state)
    else:
        best_epoch = epochs
    audit = {**{k: record[k] for k in ('fit_id','stage','mode','seed','cutoff','first_training_target','last_training_target','training_days','warm_update')},
             'epochs_run': epoch, 'selected_epochs': best_epoch, 'seconds': time.perf_counter()-started,
             'trainable_parameters': sum(p.numel() for p in trainable),
             'validation_MAE': best if validation is not None else None}
    saved = {'trainable_state': model.trainable_state(), 'audit': audit, 'config': config,
             'base_weights': 'model_manifest.json; immutable pinned official weights',
             'optimizer_reset_each_fit': True}
    tmp = path.with_suffix('.tmp')
    torch.save(saved, tmp)
    tmp.replace(path)
    return model, audit


def new_model(checkpoint, mode, seed, config):
    seed_all(seed)
    return AdaptedTimestral(checkpoint, mode, config['lora_rank'], config['lora_alpha'])


def run(args):
    started = time.perf_counter()
    config = json.loads(args.config.read_text())
    vs, cs, ts, te = [pd.Timestamp(config[k]) for k in ('selection_start','confirmation_start','test_start','test_end')]
    if not vs < cs < ts <= te or config['history_days'] != 7 or config['clip_nonnegative'] or not config['symmetric_averaging'] or config['quantile_real_cap_k'] != 0:
        raise ValueError('Invalid split or unsupported inference configuration')
    torch.set_num_threads(config['threads'])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError('Output nonempty; choose new output or --resume')
    output.mkdir(parents=True, exist_ok=True)
    (output/'checkpoints').mkdir(exist_ok=True)
    manifest = json.loads((ROOT/'outputs/timestral_sizes_v1/model_manifest.json').read_text())['Timestral-28M']
    checkpoint = manifest['local_path']
    for name, digest in manifest['sha256'].items():
        if b.sha256(Path(checkpoint)/name) != digest:
            raise ValueError('Base checkpoint hash mismatch')
    prices, calendar, _ = b.load_hourly(args.data)
    frame, test_days, threshold = paired_reference(prices, calendar, args, config)
    _, y, _ = b.build_samples(prices, calendar)
    days = y.loc[:te].index
    data = build_dataset(prices, calendar, days)
    context, known, target = data
    selection = np.flatnonzero((days >= vs) & (days < cs))
    confirmation = np.flatnonzero((days >= cs) & (days < ts))
    test = np.flatnonzero((days >= ts) & (days <= te))
    if not days[test].equals(test_days) or (len(selection),len(confirmation),len(test)) != (90,91,183):
        raise ValueError('Unexpected evaluation coverage')
    source_files = [Path(__file__), Path(__file__).with_name('timestral_finetune.py'), Path(__file__).with_name('timestral_isolation.py'), Path(__file__).with_name('foundation_models.py'), Path(b.__file__), Path(__file__).with_name('run_foundation_models.py')]
    identity = {'config': config, 'source_sha256': b.sha256(args.data),
                'baselines_sha256': b.sha256(args.baselines/'test_predictions.csv'),
                'deep_predictions_sha256': b.sha256(ROOT/'outputs/deep_models_v1/test_predictions.csv'),
                'model_revision': manifest['revision'], 'model_weights_sha256': manifest['sha256']['model.safetensors'],
                'scripts': {p.name: b.sha256(p) for p in source_files}}
    if (output/'identity.json').exists() and json.loads((output/'identity.json').read_text()) != identity:
        raise ValueError('Resume identity changed')
    b.save_json(output/'identity.json', identity)
    b.save_json(output/'config_snapshot.json', config)
    b.save_json(output/'model_manifest.json', manifest)
    for path in source_files:
        shutil.copyfile(path, output/f'{path.stem}_snapshot.py')
    shutil.copyfile(ROOT/'requirements-foundation-lock.txt', output/'requirements_snapshot.txt')
    # Input fingerprints contain no target-day labels.
    audit_inputs = []
    for i, day in enumerate(days):
        audit_inputs.append({'forecast_day': str(day.date()), 'history_end': str((day-pd.Timedelta(days=1)).date()),
                             'input_sha256': hashlib.sha256(context[i].numpy().tobytes()+known[i].numpy().tobytes()).hexdigest()})
    pd.DataFrame(audit_inputs).to_csv(output/'input_audit.csv', index=False)
    audits, confirmation_q = [], {}
    selections = {}
    with (output/'training.jsonl').open('a') as log:
        # No test predictions are produced until both recipes/epoch counts are frozen.
        for mode in ('Head','LoRA'):
            model = new_model(checkpoint, mode, config['selection_seed'], config)
            model, audit = fit(model, mode, config['selection_seed'], days, data, vs, config, output, log,
                               stage='selection', validation=selection)
            selections[mode] = audit['selected_epochs']
            audits.append(audit)
            del model
            gc.collect()
        b.save_json(output/'selected_epochs.json', selections)
        for mode in ('Head','LoRA'):
            for seed in config['seeds']:
                path = output/f'confirmation_{mode}_seed{seed}.npz'
                model = new_model(checkpoint, mode, seed, config)
                model, audit = fit(model, mode, seed, days, data, cs, config, output, log,
                                   stage='confirmation', epochs=selections[mode])
                audits.append(audit)
                if path.exists():
                    q = np.load(path)['quantiles']
                else:
                    q = predict(model, context[confirmation], known[confirmation], config['predict_batch_size'])
                    np.savez_compressed(path, quantiles=q)
                confirmation_q[f'{mode}_seed{seed}'] = q
                del model
                gc.collect()
        # Confirmation is reported without changing either selected recipe.
        b.save_json(output/'confirmation_metrics.json', {n: {'MAE': float(np.abs(q[...,10]-target[confirmation].numpy()).mean())} for n,q in confirmation_q.items()})
        zero = new_model(checkpoint, 'ZeroShot', 42, config)
        single = predict(zero, context[test[:1]], known[test[:1]])
        mixed = predict(zero, context[test[[0,-1]]], known[test[[0,-1]]])[:1]
        diagnostics={'ZeroShot':check_isolation(zero,context[test[[0,-1]]],known[test[[0,-1]]])}
        zpath = output/'ZeroShot_quantiles.npz'
        t0 = time.perf_counter()
        fresh_zero = predict(zero, context[test], known[test], config['predict_batch_size'])
        zero_seconds = time.perf_counter()-t0
        if zpath.exists():
            zero_q = np.load(zpath)['quantiles']
            np.testing.assert_allclose(zero_q,fresh_zero,rtol=1e-4,atol=1e-3)
        else:
            zero_q = fresh_zero
            np.savez_compressed(zpath, quantiles=zero_q)
        levels = zero.levels
        del zero
        quantiles = {'ZeroShot': zero_q}
        isolation = {'ZeroShot': float(np.abs(single-mixed).max())}
        for mode in ('Head','LoRA'):
            for seed in config['seeds']:
                path = output/f'{mode}_seed{seed}_quantiles.npz'
                model = new_model(checkpoint, mode, seed, config)
                chunks, cursor, last_fit = [], 0, None
                while cursor < len(test):
                    cutoff = days[test[cursor]]
                    warm = last_fit is not None
                    model, audit = fit(model, mode, seed, days, data, cutoff, config, output, log,
                                       stage='test', epochs=config['update_epochs'] if warm else selections[mode], warm=warm)
                    audits.append(audit)
                    stop = cursor
                    while stop < len(test) and days[test[stop]] < cutoff+pd.Timedelta(days=config['retrain_days']):
                        stop += 1
                    t0 = time.perf_counter()
                    q = predict(model, context[test[cursor:stop]], known[test[cursor:stop]], config['predict_batch_size'])
                    audit['predict_seconds'] = time.perf_counter()-t0
                    chunks.append(q)
                    last_fit, cursor = cutoff, stop
                q = np.concatenate(chunks)
                np.savez_compressed(path, quantiles=q)
                quantiles[f'{mode}_seed{seed}'] = q
                single = predict(model, context[test[:1]], known[test[:1]])
                mixed = predict(model, context[test[[0,-1]]], known[test[[0,-1]]])[:1]
                diagnostics[f'{mode}_seed{seed}']=check_isolation(model,context[test[[0,-1]]],known[test[[0,-1]]])
                isolation[f'{mode}_seed{seed}'] = float(np.abs(single-mixed).max())
                del model
                gc.collect()
    for mode in ('Head','LoRA'):
        quantiles[mode] = np.mean([quantiles[f'{mode}_seed{s}'] for s in config['seeds']],axis=0)
    for name,q in quantiles.items():
        if q.shape != (183,24,21) or not np.isfinite(q).all() or (np.diff(q,axis=-1)<-1e-4).any():
            raise ValueError('Invalid saved quantiles')
        frame[name] = q[...,levels.index(.5)].ravel().astype(np.float64)
        pd.DataFrame({'delivery_date': frame.delivery_date, 'hour_end': frame.hour_end,
                      'q10': q[...,levels.index(.1)].ravel().astype(np.float64), 'q50': q[...,levels.index(.5)].ravel().astype(np.float64),
                      'q90': q[...,levels.index(.9)].ravel().astype(np.float64)}).to_csv(output/f'{name}_predictions.csv',index=False)
    frame.to_csv(output/'test_predictions.csv',index=False)
    scores = {name:b.metrics(frame.actual,frame[name],frame.WeeklyNaive,threshold)
              for name in frame if name not in ('delivery_date','hour_end','forecast_origin','actual')}
    b.save_json(output/'metrics.json',{'test':scores})
    probabilities = {}
    actual = frame.actual.to_numpy().reshape(183,24)
    for name,q in quantiles.items():
        error = actual[...,None]-q
        probabilities[name] = {'mean_pinball_21':float(np.maximum(error*np.array(levels),error*(np.array(levels)-1)).mean()),
                               'coverage_80':float(((actual>=q[...,levels.index(.1)])&(actual<=q[...,levels.index(.9)])).mean()),
                               'width_80':float((q[...,levels.index(.9)]-q[...,levels.index(.1)]).mean())}
    b.save_json(output/'probabilistic_metrics.json',probabilities)
    groups=[]
    for kind,labels in {'hour':frame.hour_end,'month':frame.delivery_date.str[:7]}.items():
        for label,group in frame.groupby(labels):
            for name in ('WeeklyNaive','LEAR','LightGBM','LSTM','TFT','ZeroShot','Head','LoRA'):
                groups.append({'group':kind,'value':label,'model':name,**b.metrics(group.actual,group[name],group.WeeklyNaive,threshold)})
    pd.DataFrame(groups).to_csv(output/'group_metrics.csv',index=False)
    b.save_json(output/'timing.json',{'fits':audits,'zero_predict_seconds':zero_seconds,'elapsed_seconds':time.perf_counter()-started})
    pd.DataFrame(audits).to_csv(output/'fit_audit.csv',index=False)
    assert all(a['last_training_target']<a['cutoff'] for a in audits)
    b.save_json(output/'verification.json',{'rows':len(frame),'days':len(test),'all_training_labels_before_cutoff':True,
                'batch_isolation_max_abs_difference':isolation,'batch_isolation_diagnostics':diagnostics,'selected_before_test':True,
                'pretraining_contamination_excluded':False})
    b.save_json(output/'metadata.json',{'identity':identity,'peak_threshold':threshold,'python':platform.python_version(),
                'dependencies':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
                'device':'cpu','split_samples':{'initial_train':len(training_indices(days,vs)), 'selection':len(selection),'confirmation':len(confirmation),'test':len(test)}})
    print(json.dumps({n:scores[n] for n in ('ZeroShot','Head','LoRA')},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/timestral_finetune.json')
    parser.add_argument('--data',type=Path,default=ROOT/'data/raw/shandong/prices.csv')
    parser.add_argument('--baselines',type=Path,default=ROOT/'outputs/baselines_v1')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/timestral_finetune_v1')
    parser.add_argument('--resume',action='store_true')
    run(parser.parse_args())
