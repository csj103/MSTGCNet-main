from pathlib import Path
import sys, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.alfa_anomaly_types import canonical_anomaly_type
OUT = Path(__file__).parent
features = json.loads((ROOT/'dataset/ALFA10vars/metadata.json').read_text())['features']
train = pd.read_csv(ROOT/'dataset/ALFA10vars/train.csv')
scale = train[features].std(ddof=0).replace(0, np.nan)
test = pd.read_csv(ROOT/'dataset/ALFA10vars/test.csv')
meta = pd.read_csv(ROOT/'dataset/ALFA10vars/test_meta.csv')
test_names = set(meta.source_file)
rows, quality = [], []
for path in sorted((ROOT/'alfa_10vars').glob('*.csv')):
    if 'no_ground_truth' in path.name:
        continue
    raw = pd.read_csv(path)
    kind = canonical_anomaly_type(path.name)
    y = raw.label_anomaly.fillna(0).ne(0).to_numpy()
    t = raw.time_sec.to_numpy()
    dt = np.diff(t)
    quality.append(dict(file=path.name, type=kind, rows=len(raw), missing=int(raw[features].isna().sum().sum()),
        nonfinite=int(np.isinf(raw[features].to_numpy()).sum()), time_nonpositive=int((dt<=0).sum()),
        dt_median=float(np.median(dt)), dt_p95=float(np.quantile(dt,.95)), anomaly_rows=int(y.sum())))
    starts = np.flatnonzero(y & ~np.r_[False,y[:-1]])
    ends = np.flatnonzero(y & ~np.r_[y[1:],False])+1
    data = raw[features].bfill().ffill()
    for event, (start,end) in enumerate(zip(starts,ends)):
        prev_end = ends[event-1] if event else 0
        n = min(end-start, start-prev_end, 600)
        if n < 30:
            continue
        pre = data.iloc[start-n:start]
        post = data.iloc[start:start+n]
        shift = (post.median()-pre.median())/scale
        spread = (post.quantile(.75)-post.quantile(.25))/(pre.quantile(.75)-pre.quantile(.25)).replace(0,np.nan)
        c0,c1=pre.corr(method='spearman'),post.corr(method='spearman')
        delta=(c1-c0).to_numpy()
        pairs=[(abs(delta[i,j]),features[i]+' / '+features[j],float(c0.iloc[i,j]),float(c1.iloc[i,j])) for i in range(10) for j in range(i+1,10) if np.isfinite(delta[i,j])]
        pairs.sort(reverse=True)
        order=shift.abs().sort_values(ascending=False).index.tolist()
        rows.append(dict(file=path.name,type=kind,test=path.name in test_names,event=event,start=int(start),end=int(end),n=int(n),
            onset_time=float(t[start]),duration=float(t[end-1]-t[start]),
            pre_span=float(t[start-1]-t[start-n]),post_span=float(t[start+n-1]-t[start]),
            shifts=shift.to_dict(),iqr_ratio=spread.to_dict(),top=order[:3],
            corr_change=float(np.mean([x[0] for x in pairs])) if pairs else None,top_pairs=pairs[:3]))
result=[]
for r in rows:
    if not r['test']: continue
    ids=meta.index[meta.source_file.eq(r['file'])].to_numpy()
    r['scores']={}
    for directory in (ROOT/'label_results').glob('*v12*'):
        p=directory/'causal_last'
        if not (p/'test_energy.npy').exists(): continue
        idx=np.load(p/'test_indices.npy'); lab=np.load(p/'test_labels.npy'); energy=np.load(p/'test_energy.npy'); threshold=np.load(p/'threshold.npy'); pred=np.load(p/'raw_pred.npy')
        assert np.array_equal(lab,test.label.to_numpy()[idx])
        local=np.isin(idx,ids); anomaly=local & (lab==1); normal=local & (lab==0)
        hits=np.flatnonzero(anomaly & (pred==1))
        name='no_graph' if 'no_graph' in directory.name else 'baseline'
        r['scores'][name]=dict(normal_score_median=float(np.median(energy[normal])),anomaly_score_median=float(np.median(energy[anomaly])),
            anomaly_threshold_median=float(np.median(threshold[anomaly])),recall=float(pred[anomaly].mean()),
            first_delay_seconds=float(test.time_sec.iloc[idx[hits[0]]]-r['onset_time']) if len(hits) else None)
    result.append(r)
summary=[]
for kind in sorted(set(r['type'] for r in rows)):
    group=[r for r in rows if r['type']==kind]
    shifts=pd.DataFrame([r['shifts'] for r in group])
    summary.append(dict(type=kind,flights=len(set(r['file'] for r in group)),median_abs_shift=shifts.abs().median().to_dict(),
        signed_median_shift=shifts.median().to_dict(),positive_fraction=(shifts>0).mean().to_dict()))
payload=dict(method='Equal number of adjacent pre/post onset rows, n=min(anomaly length, preceding normal length,600), n>=30; shift=(post median-pre median)/training normal std; raw features filled exactly as preprocessing. Spearman correlations descriptive, not causal.',test=result,all_types=summary,events=rows,quality=quality)
(OUT/'analysis.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=True),encoding='utf-8')
for r in result:
    print(json.dumps({k:r[k] for k in ['type','n','duration','pre_span','post_span','top','corr_change','top_pairs','scores']}|{'shifts':{k:round(r['shifts'][k],3) for k in r['top']}},ensure_ascii=True))
print('ALL_TYPES',json.dumps(summary))
print('QUALITY',json.dumps(dict(files=len(quality),missing=sum(r['missing'] for r in quality),nonfinite=sum(r['nonfinite'] for r in quality),nonpositive=sum(r['time_nonpositive'] for r in quality),dt_range=[min(r['dt_median'] for r in quality),max(r['dt_median'] for r in quality)])))


