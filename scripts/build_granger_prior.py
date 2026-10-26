"""Conditional linear VAR Granger on normal TRAIN flights; no test/val reads."""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as f_distribution

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.preprocess_alfa import FEATURE_COLUMNS


def estimate_window(values,max_lag):
    """Nested OLS conditional on ALL variables' history, G[target,source].

    Rank differences supply effective numerator degrees of freedom, including
    windows with constant/collinear channels. No ridge F-test approximation.
    Weights are relative restricted-model SSR improvement (not correlation).
    """
    values=np.asarray(values,dtype=np.float64)
    if max_lag<1 or values.ndim!=2 or not np.isfinite(values).all():
        raise ValueError('Finite 2-D data and positive lag required')
    t,v=values.shape
    if t-max_lag<=v*max_lag+2: raise ValueError('Window too short for conditional VAR')
    std=values.std(0)
    z=(values-values.mean(0))/np.where(std>1e-10,std,1.)
    y=z[max_lag:]
    design=np.column_stack([np.ones(t-max_lag)]+[z[max_lag-l:t-l] for l in range(1,max_lag+1)])
    coef,_,rank,_=np.linalg.lstsq(design,y,rcond=None)
    ssr=((y-design@coef)**2).sum(0)
    score=np.zeros((v,v)); pvalue=np.ones((v,v))
    df2=len(y)-rank
    for source in range(v):
        if std[source]<=1e-10: continue
        keep=np.ones(design.shape[1],dtype=bool)
        keep[1+np.arange(max_lag)*v+source]=False
        restricted=design[:,keep]
        cr,_,rr,_=np.linalg.lstsq(restricted,y,rcond=None)
        df1=rank-rr
        if df1<=0 or df2<=0: continue
        ssrr=((y-restricted@cr)**2).sum(0)
        improvement=np.maximum(ssrr-ssr,0)
        # Exact prediction is a strong nested-model limit, not an invalid edge.
        # Constant targets and redundant sources are handled by std/rank above.
        valid=(std>1e-10)&(ssrr>1e-12)
        statistic=(improvement/df1)/(np.maximum(ssr,1e-30)/df2)
        pvalue[valid,source]=f_distribution.sf(statistic[valid],df1,df2)
        score[valid,source]=np.clip(improvement[valid]/ssrr[valid],0,1)
    np.fill_diagonal(score,0); np.fill_diagonal(pvalue,1)
    return score,pvalue


def bh_mask(pvalues,alpha):
    v=len(pvalues); off=~np.eye(v,dtype=bool)
    p=pvalues[off]; ordered=np.sort(p)
    passed=ordered<=alpha*np.arange(1,len(p)+1)/len(p)
    cutoff=ordered[np.flatnonzero(passed)[-1]] if passed.any() else -1
    return (pvalues<=cutoff)&off


def training_segments(root,normal_flights_only=True):
    """Read training files only and respect the loader's fallback 90% split."""
    data=pd.read_csv(root/'train.csv'); meta=pd.read_csv(root/'train_meta.csv')
    info=json.loads((root/'metadata.json').read_text(encoding='utf-8'))
    if len(data)!=len(meta) or list(data.columns[1:-1])!=FEATURE_COLUMNS:
        raise ValueError('Training data/metadata alignment or variable order mismatch')
    if not data.label.eq(0).all() or not meta.label.eq(0).all():
        raise ValueError('Training prior requires exclusively normal rows')
    if not np.allclose(data.iloc[:,0],meta.time_sec): raise ValueError('Time metadata misalignment')
    if info.get('split_policy') not in {'fine_grained','fault_balanced'} or not (root/'val.csv').exists():
        stop=int(len(data)*.9); data=data.iloc[:stop]; meta=meta.iloc[:stop]
    if normal_flights_only:
        if 'fault_type' not in meta: raise ValueError('fault_type required for normal flight selection')
        mask=meta.fault_type.eq('normal')
    else: mask=pd.Series(True,index=meta.index)
    boundary=meta.segment_id.ne(meta.segment_id.shift())|meta.source_file.ne(meta.source_file.shift())
    if 'original_index' in meta and info.get('sampling_mode')!='regular':
        boundary |= meta.original_index.diff().fillna(1).ne(1)
    dt=meta.time_sec.diff()
    boundary |= dt.le(0)|dt.gt(info.get('max_gap_seconds',.25))|mask.ne(mask.shift())
    for run,indices in meta.groupby(boundary.cumsum(),sort=False).groups.items():
        rows=np.asarray(list(indices))
        if not mask.loc[rows].all() or len(rows)<2: continue
        yield str(run),data.loc[rows,FEATURE_COLUMNS].to_numpy(float),meta.loc[rows],info


def build(args):
    if args.max_lag<1 or args.stride<1 or args.window<=10*args.max_lag+args.max_lag+3:
        raise ValueError('Invalid lag/window/stride')
    if not 0<args.alpha<1 or not 0<=args.stability<=1 or args.min_flights<1:
        raise ValueError('Invalid alpha/stability/min_flights')
    source=args.root
    meta_info=json.loads((source/'metadata.json').read_text(encoding='utf-8'))
    if meta_info.get('sampling_mode')!='regular':
        raise ValueError('Regular-time dataset required; run scripts/regularize_alfa.py first')
    rows=[]; audits=[]; flight_scores={}; flight_hits={}; flight_counts={}
    for segment,values,meta,info in training_segments(source,not args.include_prefault):
        dt=np.diff(meta.time_sec.to_numpy(float))
        audits.append(dict(segment=segment,source_file=str(meta.source_file.iloc[0]),rows=len(meta),
            dt_median=float(np.median(dt)),dt_p05=float(np.quantile(dt,.05)),
            dt_p95=float(np.quantile(dt,.95)),dt_cv=float(dt.std()/dt.mean()),
            duplicates=int((dt<=0).sum()),large_gaps=int((dt>info.get('max_gap_seconds',.25)).sum())))
        if not np.allclose(dt,1/info['sampling_hz'],atol=1e-7,rtol=1e-6):
            raise ValueError('Nonuniform intervals remain inside a prior estimation segment')
        file=str(meta.source_file.iloc[0])
        for start in range(0,len(values)-args.window+1,args.stride):
            window=values[start:start+args.window]
            if args.transform=='difference': window=np.diff(window,axis=0)
            weight,p=estimate_window(window,args.max_lag)
            hit=bh_mask(p,args.alpha)&(weight>=args.min_effect)
            flight_scores[file]=flight_scores.get(file,0)+weight*hit
            flight_hits[file]=flight_hits.get(file,0)+hit
            flight_counts[file]=flight_counts.get(file,0)+1
            for target,source_id in zip(*np.where(hit)):
                rows.append(dict(source_file=file,segment=segment,start=start,target=FEATURE_COLUMNS[target],
                    source=FEATURE_COLUMNS[source_id],weight=float(weight[target,source_id]),pvalue=float(p[target,source_id])))
    if not flight_counts: raise ValueError('No normal training windows large enough for VAR')
    files=sorted(flight_counts)
    rates=np.stack([flight_hits[f]/flight_counts[f] for f in files])
    effects=np.stack([flight_scores[f]/flight_counts[f] for f in files])
    stability=rates.mean(0)
    supporting=(rates>0).sum(0)
    matrix=effects.mean(0)*((stability>=args.stability)&(supporting>=args.min_flights))
    np.fill_diagonal(matrix,0)
    metadata=dict(features=FEATURE_COLUMNS,orientation='target_source',method='linear_var',
        training_only=True,normal_flights_only=not args.include_prefault,max_lag=args.max_lag,
        sampling_mode='regular',sampling_hz=meta_info['sampling_hz'],resampling=meta_info['resampling'],
        transform=args.transform,window=args.window,stride=args.stride,fdr_alpha=args.alpha,
        min_effect=args.min_effect,stability_threshold=args.stability,min_flights=args.min_flights,
        stability_aggregation='mean within flight then equal mean across flights',
        weight='mean relative restricted SSR improvement times significant-edge indicator across windows and flights',
        source_root=str(args.root.resolve()),source_files=files,flight_window_counts=flight_counts,
        train_sha256=hashlib.sha256((args.root/'train.csv').read_bytes()).hexdigest(),
        train_meta_sha256=hashlib.sha256((args.root/'train_meta.csv').read_bytes()).hexdigest(),
        python=platform.python_version(),numpy=np.__version__,total_windows=sum(flight_counts.values()),
        caveat='Linear predictive dependence, not intervention causality; nonstationarity and hidden confounders remain possible')
    out=args.output; out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'granger_prior.npz',matrix=matrix.astype(np.float32),metadata=json.dumps(metadata))
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    pd.DataFrame(matrix,index=FEATURE_COLUMNS,columns=FEATURE_COLUMNS).to_csv(out/'matrix.csv')
    pd.DataFrame(stability,index=FEATURE_COLUMNS,columns=FEATURE_COLUMNS).to_csv(out/'stability.csv')
    pd.DataFrame(audits).to_csv(out/'sampling_audit.csv',index=False)
    pd.DataFrame(rows,columns=['source_file','segment','start','target','source','weight','pvalue']).to_csv(out/'window_edges.csv',index=False)
    edge_rows=[dict(source=FEATURE_COLUMNS[i],target=FEATURE_COLUMNS[j],weight=float(matrix[j,i]),
        stability=float(stability[j,i]),supporting_flights=int(supporting[j,i])) for j,i in zip(*np.where(matrix>0))]
    pd.DataFrame(edge_rows,columns=['source','target','weight','stability','supporting_flights']).to_csv(out/'stable_edges.csv',index=False)
    per_flight=[]
    for f in files:
        for j in range(10):
            for i in range(10):
                if i!=j: per_flight.append(dict(source_file=f,target=FEATURE_COLUMNS[j],source=FEATURE_COLUMNS[i],
                    window_support=float(flight_hits[f][j,i]/flight_counts[f]),windows=flight_counts[f]))
    pd.DataFrame(per_flight).to_csv(out/'flight_stability.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,8)); im=ax.imshow(matrix,cmap='magma',vmin=0)
    ax.set(xticks=range(10),yticks=range(10),xticklabels=FEATURE_COLUMNS,yticklabels=FEATURE_COLUMNS,
           xlabel='Source variable i',ylabel='Target variable j',title='Normal TRAIN VAR-Granger G[j,i]')
    plt.setp(ax.get_xticklabels(),rotation=60,ha='right'); fig.colorbar(im,ax=ax,label='Stable conditional SSR improvement')
    fig.tight_layout(); fig.savefig(out/'heatmap.png',dpi=150); plt.close(fig)
    print(json.dumps(dict(flights=len(files),windows=sum(flight_counts.values()),stable_edges=len(edge_rows),output=str(out))))
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('dataset/ALFA10vars_20hz'))
    p.add_argument('--output',type=Path,default=Path('analysis/variable_granger_v1/prior'))
    p.add_argument('--method',choices=['linear_var'],default='linear_var')
    p.add_argument('--max-lag',type=int,default=5)
    p.add_argument('--window',type=int,default=512)
    p.add_argument('--stride',type=int,default=512)
    p.add_argument('--transform',choices=['difference','level'],default='difference')
    p.add_argument('--alpha',type=float,default=.05)
    p.add_argument('--min-effect',type=float,default=.01)
    p.add_argument('--stability',type=float,default=.3)
    p.add_argument('--min-flights',type=int,default=2)
    p.add_argument('--include-prefault',action='store_true',help='Also use normal pre-fault training segments; never test/val')
    build(p.parse_args())


if __name__=='__main__': main()
