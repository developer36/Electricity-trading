#!/usr/bin/env python3
"""Compute an independent seed in a separate process, publishing atomic fit caches.

The main experiment remains the only writer of aggregate results. A worker never
changes selection, configuration or another seed's weights. Existing main outputs
win; optional duplicate deterministic work is discarded.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch

try:
    from . import run_timestral_finetune as r
except ImportError:
    import run_timestral_finetune as r


def main(seed, mode='LoRA', skip_confirmation=False):
    root=r.ROOT
    output=root/'outputs/timestral_finetune_v1'
    config=json.loads((output/'config_snapshot.json').read_text())
    if seed not in config['seeds'] or (seed==config['selection_seed'] and not (mode=='Head' and skip_confirmation)):
        raise ValueError('Selection seed may only precompute Head test fits after choices freeze')
    selected_path=output/'selected_epochs.json'
    if not selected_path.exists():
        raise ValueError('All epoch choices must be frozen before dispatch')
    selected=json.loads(selected_path.read_text())[mode]
    torch.set_num_threads(config['threads'])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    worker=output/(f'worker_seed{seed}' if mode=='LoRA' else f'worker_Head_seed{seed}')
    worker.mkdir(exist_ok=True)
    (worker/'checkpoints').mkdir(exist_ok=True)
    shutil.copyfile(Path(__file__),worker/'worker_snapshot.py')
    prices,calendar,_=r.b.load_hourly(root/'data/raw/shandong/prices.csv')
    _,y,_=r.b.build_samples(prices,calendar)
    days=y.loc[:config['test_end']].index
    data=r.build_dataset(prices,calendar,days)
    context,known,_=data
    cs,ts,te=[pd.Timestamp(config[k]) for k in ('confirmation_start','test_start','test_end')]
    confirmation=np.flatnonzero((days>=cs)&(days<ts))
    test=np.flatnonzero((days>=ts)&(days<=te))
    manifest=json.loads((output/'model_manifest.json').read_text())
    checkpoint=manifest['local_path']
    for filename,digest in manifest['sha256'].items():
        assert r.b.sha256(Path(checkpoint)/filename)==digest
    source_identity=json.loads((output/'identity.json').read_text())
    assert r.b.sha256(root/'data/raw/shandong/prices.csv')==source_identity['source_sha256']
    for filename,digest in source_identity['scripts'].items():
        assert r.b.sha256(root/'scripts'/filename)==digest
    r.b.save_json(worker/'identity.json',source_identity)

    def publish(source,destination):
        if destination.exists():
            return
        temporary=destination.with_name(destination.name+f'.worker{seed}.tmp')
        shutil.copyfile(source,temporary)
        if destination.exists():
            temporary.unlink()
        else:
            temporary.replace(destination)

    def fit_publish(model,cutoff,stage,epochs,warm=False):
        ident=f'{stage}_{mode}_seed{seed}_{cutoff.date()}.pt'
        # Reuse an already completed main fit, if any, in the isolated worker cache.
        destination=output/'checkpoints'/ident
        if destination.exists() and not (worker/'checkpoints'/ident).exists():
            shutil.copyfile(destination,worker/'checkpoints'/ident)
        model,audit=r.fit(model,mode,seed,days,data,cutoff,config,worker,log,
                          stage=stage,epochs=epochs,warm=warm)
        publish(worker/'checkpoints'/ident,destination)
        return model,audit

    started=time.perf_counter()
    with (worker/'training.jsonl').open('a') as log:
        if not skip_confirmation:
            model=r.new_model(checkpoint,mode,seed,config)
            model,_=fit_publish(model,cs,'confirmation',selected)
            q=r.predict(model,context[confirmation],known[confirmation],config['predict_batch_size'])
            path=worker/f'confirmation_{mode}_seed{seed}.npz'
            np.savez_compressed(path,quantiles=q)
            publish(path,output/path.name)
            del model
        model=r.new_model(checkpoint,mode,seed,config)
        cursor=0
        while cursor<len(test):
            cutoff=days[test[cursor]]
            model,_=fit_publish(model,cutoff,'test',selected if cursor==0 else config['update_epochs'],warm=cursor>0)
            stop=cursor
            while stop<len(test) and days[test[stop]]<cutoff+pd.Timedelta(days=config['retrain_days']):
                stop+=1
            cursor=stop
    r.b.save_json(worker/'worker_timing.json',{'elapsed_seconds':time.perf_counter()-started,'seed':seed,'threads':config['threads']})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--mode',choices=('LoRA','Head'),default='LoRA')
    parser.add_argument('--skip-confirmation',action='store_true')
    args=parser.parse_args()
    main(args.seed,args.mode,args.skip_confirmation)
