#!/usr/bin/env python3
"""Finalize immutable historical checkpoints, with shape-controlled isolation tests.

No optimizer or training is invoked. Existing saved forecasts are independently
replayed and verified before reuse. All fit recipes remain unchanged.
"""
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
    from . import run_timestral_finetune as r
    from .timestral_isolation import check_isolation
except ImportError:
    import run_timestral_finetune as r
    from timestral_isolation import check_isolation

ROOT=r.ROOT


def main(output):
    identity=json.loads((output/'identity.json').read_text());c=identity['config']
    manifest=json.loads((output/'model_manifest.json').read_text())
    for name,digest in identity['scripts'].items():
        assert r.b.sha256(output/(Path(name).stem+'_snapshot.py'))==digest,name
    assert r.b.sha256(ROOT/'data/raw/shandong/prices.csv')==identity['source_sha256']
    for name,digest in manifest['sha256'].items():
        assert r.b.sha256(Path(manifest['local_path'])/name)==digest,name
    torch.set_num_threads(c['threads']);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args=type('Args',(),{'data':ROOT/'data/raw/shandong/prices.csv','baselines':ROOT/'outputs/baselines_v1'})()
    prices,calendar,_=r.b.load_hourly(args.data)
    frame,days,threshold=r.paired_reference(prices,calendar,args,c)
    context,known,target=r.build_dataset(prices,calendar,days)
    levels=json.loads((Path(manifest['local_path'])/'config.json').read_text())['quantile_knots']
    checkpoints={}
    for path in sorted((output/'checkpoints').glob('*.pt')):
        saved=torch.load(path,weights_only=False,map_location='cpu')
        audit=saved['audit']
        assert saved['config']==c and audit['last_training_target']<audit['cutoff']
        checkpoints[audit['fit_id']]=audit
    assert len(checkpoints)==170
    quantiles,diagnostics,cache_difference={},{},{}
    zero=r.new_model(manifest['local_path'],'ZeroShot',42,c)
    start=time.perf_counter()
    quantiles['ZeroShot']=r.predict(zero,context,known,c['predict_batch_size'])
    zero_seconds=time.perf_counter()-start
    diagnostics['ZeroShot']=check_isolation(zero,context[[0,-1]],known[[0,-1]])
    del zero
    for mode in ('Head','LoRA'):
        for seed in c['seeds']:
            model=r.new_model(manifest['local_path'],mode,seed,c)
            chunks=[]
            for cutoff in pd.date_range(c['test_start'],c['test_end'],freq=f"{c['retrain_days']}D"):
                ident=f'test_{mode}_seed{seed}_{cutoff.date()}'
                saved=torch.load(output/'checkpoints'/f'{ident}.pt',weights_only=False,map_location='cpu')
                model.load_trainable(saved['trainable_state'])
                take=np.flatnonzero((days>=cutoff)&(days<cutoff+pd.Timedelta(days=c['retrain_days'])))
                assert len(take)>0 and saved['audit']['last_training_target']<str(days[take[0]].date())
                start=time.perf_counter()
                chunks.append(r.predict(model,context[take],known[take],c['predict_batch_size']))
                checkpoints[ident]['predict_seconds']=time.perf_counter()-start
            name=f'{mode}_seed{seed}'
            quantiles[name]=np.concatenate(chunks)
            diagnostics[name]=check_isolation(model,context[[0,-1]],known[[0,-1]])
            print(f'{name}: historical predictions and isolation verified',flush=True)
            del model
    for name,q in quantiles.items():
        assert q.shape==(183,24,21) and np.isfinite(q).all() and (np.diff(q,axis=-1)>=-1e-4).all()
        path=output/f'{name}_quantiles.npz'
        if path.exists():
            old=np.load(path)['quantiles']
            np.testing.assert_allclose(old,q,rtol=1e-4,atol=1e-3)
            cache_difference[name]=float(np.abs(old-q).max())
            quantiles[name]=old  # Preserve the originally measured forecasts exactly.
        else:
            np.savez_compressed(path,quantiles=q)
    for mode in ('Head','LoRA'):
        quantiles[mode]=np.mean([quantiles[f'{mode}_seed{s}'] for s in c['seeds']],axis=0)
    for name,q in quantiles.items():
        frame[name]=q[...,levels.index(.5)].ravel().astype(np.float64)
        pd.DataFrame({'delivery_date':frame.delivery_date,'hour_end':frame.hour_end,
                      **{f'q{int(v*100)}':q[...,levels.index(v)].ravel().astype(np.float64) for v in (.1,.5,.9)}}).to_csv(output/f'{name}_predictions.csv',index=False)
    frame.to_csv(output/'test_predictions.csv',index=False)
    scores={n:r.b.metrics(frame.actual,frame[n],frame.WeeklyNaive,threshold) for n in frame if n not in ('delivery_date','hour_end','forecast_origin','actual')}
    r.b.save_json(output/'metrics.json',{'test':scores})
    probabilities={};actual=frame.actual.to_numpy().reshape(183,24)
    for name,q in quantiles.items():
        error=actual[...,None]-q
        probabilities[name]={'mean_pinball_21':float(np.maximum(error*np.array(levels),error*(np.array(levels)-1)).mean()),
                             'coverage_80':float(((actual>=q[...,levels.index(.1)])&(actual<=q[...,levels.index(.9)])).mean()),
                             'width_80':float((q[...,levels.index(.9)]-q[...,levels.index(.1)]).mean())}
    r.b.save_json(output/'probabilistic_metrics.json',probabilities)
    groups=[]
    for kind,labels in {'hour':frame.hour_end,'month':frame.delivery_date.str[:7]}.items():
        for label,group in frame.groupby(labels):
            for name in ('WeeklyNaive','LEAR','LightGBM','LSTM','TFT','ZeroShot','Head','LoRA'):
                groups.append({'group':kind,'value':label,'model':name,**r.b.metrics(group.actual,group[name],group.WeeklyNaive,threshold)})
    pd.DataFrame(groups).to_csv(output/'group_metrics.csv',index=False)
    audits=list(checkpoints.values())
    pd.DataFrame(audits).to_csv(output/'fit_audit.csv',index=False)
    r.b.save_json(output/'timing.json',{'fits':audits,'zero_predict_seconds':zero_seconds,
                 'elapsed_seconds':time.time()-(output/'identity.json').stat().st_mtime,
                 'wall_clock_origin':'identity.json creation; includes final verification/recovery',
                 'prediction_timing':'single-process historical replay after all training workers exited'})
    extra={}
    for name in ('finalize_timestral_finetune.py','report_timestral_finetune.py'):
        source=ROOT/'scripts'/name;extra[name]=r.b.sha256(source)
        shutil.copyfile(source,output/f'{source.stem}_snapshot.py')
    r.b.save_json(output/'metadata.json',{'identity':identity,'peak_threshold':threshold,'python':platform.python_version(),
                 'dependencies':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
                 'device':'cpu','finalization_scripts':extra,
                 'split_samples':{'initial_train':359,'selection':90,'confirmation':91,'test':183}})
    r.b.save_json(output/'verification.json',{'rows':len(frame),'days':len(days),'all_training_labels_before_cutoff':True,
                 'batch_isolation_max_abs_difference':{n:d['fp32_cross_batch_max_abs'] for n,d in diagnostics.items()},
                 'batch_isolation_diagnostics':diagnostics,'independent_replay_existing_predictions_max_abs':cache_difference,
                 'selected_before_test':True,'pretraining_contamination_excluded':False})
    print(pd.DataFrame({n:scores[n] for n in ('ZeroShot','Head','LoRA')}).T[['MAE','RMSE','negative_F1']].to_string())


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/timestral_finetune_v1')
    main(parser.parse_args().output.resolve())
