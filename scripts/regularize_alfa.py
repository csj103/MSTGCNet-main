"""Create a separate regular-time ALFA version using causal previous-value hold."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def regularize_split(data, meta, hz, max_gap):
    if len(data) != len(meta) or hz <= 0 or max_gap <= 0:
        raise ValueError('Aligned metadata and positive hz/max_gap required')
    outputs, mappings = [], []
    segment = 0
    # Never group disconnected runs of the same segment identifier together.
    boundary = meta.segment_id.ne(meta.segment_id.shift())
    if 'source_file' in meta: boundary |= meta.source_file.ne(meta.source_file.shift())
    if 'original_index' in meta: boundary |= meta.original_index.diff().fillna(1).ne(1)
    times=data.iloc[:,0].to_numpy(float)
    dt=np.diff(times,prepend=times[0])
    boundary |= (dt <= 0) | (dt > max_gap)
    for _, group in meta.groupby(boundary.cumsum(), sort=False):
        rows=group.index.to_numpy()
        t=times[rows]
        if len(t)<2: continue
        grid=t[0]+np.arange(int(np.floor((t[-1]-t[0])*hz+1e-8))+1)/hz
        selected=np.searchsorted(t,grid+1e-10,side='right')-1
        out=data.iloc[rows[selected]].copy().reset_index(drop=True)
        out.iloc[:,0]=grid
        mapping=meta.iloc[rows[selected]].copy().reset_index(drop=True)
        mapping['original_time_sec']=t[selected]
        mapping['input_split_row']=rows[selected]
        mapping['time_sec']=grid
        mapping['segment_id']=segment
        outputs.append(out); mappings.append(mapping); segment+=1
    if not outputs: raise ValueError('No continuous segments to regularize')
    return pd.concat(outputs,ignore_index=True),pd.concat(mappings,ignore_index=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('dataset/ALFA10vars'))
    parser.add_argument('--output',type=Path,default=Path('dataset/ALFA10vars_20hz'))
    parser.add_argument('--hz',type=float,default=20.)
    parser.add_argument('--max-gap',type=float,default=.25)
    a=parser.parse_args()
    if a.source.resolve()==a.output.resolve(): raise ValueError('Output must differ from original dataset')
    if a.output.exists(): raise FileExistsError('Output exists; choose a new version directory')
    metadata=json.loads((a.source/'metadata.json').read_text(encoding='utf-8'))
    a.output.mkdir(parents=True)
    summary={}
    audits=[]
    output_frames={}; output_meta={}
    for split in ('train','val','test'):
        path=a.source/f'{split}.csv'
        if not path.exists(): continue
        data=pd.read_csv(path); meta=pd.read_csv(a.source/f'{split}_meta.csv')
        for segment,g in meta.groupby('segment_id',sort=False):
            dt=np.diff(g.time_sec.to_numpy(float))
            if len(dt): audits.append(dict(split=split,segment=segment,rows=len(g),
                source_file=str(g.source_file.iloc[0]) if 'source_file' in g else '',
                dt_p05=float(np.quantile(dt,.05)),dt_median=float(np.median(dt)),dt_p95=float(np.quantile(dt,.95)),
                dt_cv=float(dt.std()/dt.mean()),duplicates=int((dt<=0).sum()),large_gaps=int((dt>a.max_gap).sum())))
        out,mapping=regularize_split(data,meta,a.hz,a.max_gap)
        output_frames[split]=out; output_meta[split]=mapping
        out.to_csv(a.output/path.name,index=False)
        mapping.to_csv(a.output/f'{split}_meta.csv',index=False)
        summary[split]=dict(input_rows=len(data),output_rows=len(out),
            input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            output_sha256=hashlib.sha256((a.output/path.name).read_bytes()).hexdigest())
        metadata[f'{split}_rows']=len(out)
        if split=='test':
            metadata['test_anomaly_rows']=int(out.label.sum())
            metadata['test_anomaly_ratio']=float(out.label.mean())
        print(split,summary[split])
    # Replace counts that refer to the original row grid; retain source metadata
    # separately so a regular-time experiment cannot be mistaken for raw V12.
    metadata['original_metadata']=json.loads((a.source/'metadata.json').read_text(encoding='utf-8'))
    metadata['output']=str(a.output)
    train_count=len(output_frames['train']); val_count=len(output_frames.get('val',[]))
    metadata['train_val_rows']=train_count+val_count
    metadata['loader_train_rows']=train_count if val_count else int(train_count*.9)
    metadata['loader_val_rows']=val_count if val_count else train_count-int(train_count*.9)
    metadata['matched_paper_train_val_rows']=metadata['train_val_rows']==metadata.get('paper_train_val_target_rows')
    test_mapping=output_meta.get('test')
    if test_mapping is not None:
        for col,name in [('fault_type','test_anomaly_rows_by_fault'),('fine_anomaly_type','test_anomaly_rows_by_fine_type')]:
            if col in test_mapping:
                metadata[name]={str(k):int(v) for k,v in test_mapping.groupby(col).label.sum().items() if v>0}
        label=output_frames['test'].label
        metadata['test_anomaly_intervals']=int(((label!=label.shift())|test_mapping.segment_id.ne(test_mapping.segment_id.shift()))[label==1].sum())
    metadata.update(sampling_mode='regular',sampling_hz=a.hz,resampling='previous_value_hold',
        max_gap_seconds=a.max_gap,resampling_source=str(a.source.resolve()),resampling_summary=summary,
        label_mapping='previous observed label; onset quantization < 1/hz; original row/time saved')
    (a.output/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    pd.DataFrame(audits).to_csv(a.output/'original_sampling_audit.csv',index=False)


if __name__=='__main__': main()
