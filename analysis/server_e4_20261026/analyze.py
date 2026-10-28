"""Read-only audit of exported E4 server results; writes only this analysis folder."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.atssd import atssd
from utils.experiment_reporting import event_details
from utils.alfa_anomaly_types import canonical_anomaly_type_from_row
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from run import build_setting
from types import SimpleNamespace


def metrics(y, pred, scores):
    precision, recall, f1, _ = precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    return dict(precision=float(precision),recall=float(recall),f1=float(f1),
                roc_auc=float(roc_auc_score(y,scores)),ap=float(average_precision_score(y,scores)),
                confusion_matrix=confusion_matrix(y,pred).tolist(),points=len(y),anomalies=int(y.sum()))


def runs(mask):
    bounds=np.diff(np.r_[0,np.asarray(mask,dtype=int),0])
    return list(zip(np.flatnonzero(bounds==1),np.flatnonzero(bounds==-1)))


def mdtable(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+
                     ['|'+'|'.join(map(str,r))+'|' for r in rows])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=ROOT/'服务器训练结果')
    parser.add_argument('--data',type=Path,default=ROOT/'dataset/ALFA10vars_20hz')
    args=parser.parse_args()
    out=Path(__file__).resolve().parent
    candidates=list((args.results/'label_results').glob('*/causal_last'))
    if len(candidates)!=1: raise ValueError('Select exactly one E4 result directory')
    source=candidates[0]; setting=source.parent.name
    ckpt_dir=args.results/'checkpoints'/setting
    config=json.loads((ckpt_dir/'experiment_config.json').read_text(encoding='utf-8'))
    exported=json.loads((source/'metrics.json').read_text(encoding='utf-8'))
    names=['test_energy','test_labels','test_indices','raw_pred','adjusted_pred','threshold']
    arrays={name:np.load(source/(name+'.npy'),allow_pickle=False) for name in names}
    s=arrays['test_energy']; y=arrays['test_labels'].astype(int); ix=arrays['test_indices'].astype(int)
    pred=arrays['raw_pred'].astype(int); adjusted=arrays['adjusted_pred'].astype(int); threshold=arrays['threshold']
    assert all(a.shape==s.shape and np.isfinite(a).all() for a in arrays.values())
    assert np.all(np.diff(ix)>0) and set(np.unique(y))=={0,1}
    test=pd.read_csv(args.data/'test.csv'); meta=pd.read_csv(args.data/'test_meta.csv')
    train_meta=pd.read_csv(args.data/'train_meta.csv'); val_meta=pd.read_csv(args.data/'val_meta.csv')
    split_files={name:set(frame.source_file) for name,frame in [('train',train_meta),('val',val_meta),('test',meta)]}
    split_disjoint=not any(split_files[a]&split_files[b] for a,b in [('train','val'),('train','test'),('val','test')])
    prior_files=set(exported['prior_metadata']['source_files'])
    prior_training_only=prior_files.issubset(split_files['train']) and not prior_files&(split_files['val']|split_files['test'])
    assert split_disjoint and prior_training_only and train_meta.label.eq(0).all() and val_meta.label.eq(0).all()
    aligned=meta.iloc[ix].reset_index(drop=True); segments=aligned.segment_id.to_numpy()
    assert np.array_equal(test.label.to_numpy()[ix],y)
    assert np.array_equal(meta.label.to_numpy()[ix],y)
    assert np.allclose(test.time_sec,meta.time_sec)
    for _,g in meta.groupby('segment_id'):
        assert np.allclose(np.diff(g.time_sec),.05,atol=1e-8)
    recomputed_pred,recomputed_threshold=atssd(s,window_size=config['args']['winsize'],alpha=config['args']['alpha'],segment_ids=segments)
    assert np.array_equal(recomputed_pred,pred)
    assert np.array_equal(recomputed_threshold,threshold)
    assert np.array_equal(s>threshold,pred)
    _,recomputed_adjusted=Exp_Anomaly_Detection._adjust_by_segment(y,pred,segments)
    assert np.array_equal(adjusted,recomputed_adjusted)
    result=metrics(y,pred,s); pa=metrics(y,adjusted,s)
    for field,key in [('roc_auc','roc_auc'),('ap','pr_auc_average_precision'),('precision','raw_precision'),('recall','raw_recall'),('f1','raw_f1')]:
        assert abs(result[field]-exported[key])<1e-12
    assert abs(pa['f1']-exported['point_adjusted_f1'])<1e-12
    result['prevalence']=float(y.mean())
    result['false_positive_rate']=float(pred[y==0].mean())
    result['point_adjusted']=pa
    result['event_hits']=exported['event_hits']; result['events']=exported['events']
    # Decompose ATSSD without selecting/tuning a replacement threshold.
    base=np.zeros(len(s)); trend_factor=np.ones(len(s))
    for segment in np.unique(segments):
        rows=np.flatnonzero(segments==segment); values=s[rows].astype(float); pm=ps=None
        for t,row in enumerate(rows):
            window=values[max(0,t-95):t+1]; mean=window.mean(); std=window.std()
            base[row]=mean+NormalDist().inv_cdf(.99)*std
            if pm is not None and pm!=0 and ps!=0:
                trend_factor[row]=1+max(0,(mean-pm)/abs(pm)+(std-ps)/abs(ps))
            pm,ps=mean,std
    assert np.array_equal(base*trend_factor,threshold)
    decomposition=dict(base_only_offline_diagnostic=metrics(y,(s>base).astype(int),s),
        additional_anomaly_points_suppressed_by_trend=int(((y==1)&(s>base)&(pred==0)).sum()),
        normal_points_suppressed_by_trend=int(((y==0)&(s>base)&(pred==0)).sum()),
        anomaly_trend_factor_median=float(np.median(trend_factor[y==1])),
        scope='Same recorded scores; omit only trend multiplier for explanation, not a validated replacement threshold')
    types=aligned.apply(canonical_anomaly_type_from_row,axis=1).to_numpy()
    raw_csv=pd.read_csv(source/'raw_metrics_by_anomaly_type.csv')
    events=event_details(test.label.to_numpy(),pred,ix,meta)
    supplied_events=pd.read_csv(source/'event_details.csv')
    event_rows=[]; fault_rows=[]; fp_runs=[]; tp_runs=[]
    for event in events:
        label=event['start_fine_anomaly_type']; mask=types==label
        row=metrics(y[mask],pred[mask],s[mask]); row['fault_type']=label
        old=raw_csv[raw_csv['Anomaly Type']==label].iloc[0]
        for a,b in [('precision','Precision'),('recall','Recall'),('f1','F-score'),('roc_auc','ROC-AUC'),('ap','PR-AUC (AP)')]:
            assert abs(row[a]-old[b])<1e-12
        normal=mask&(y==0); abnormal=mask&(y==1)
        row.update(normal_points=int(normal.sum()),false_positives=int(pred[normal].sum()),
            true_positives=int(pred[abnormal].sum()),false_positive_rate=float(pred[normal].mean()),
            prevalence=float(y[mask].mean()),normal_score_median=float(np.median(s[normal])),
            anomaly_score_median=float(np.median(s[abnormal])),anomaly_threshold_median=float(np.median(threshold[abnormal])),
            normal_threshold_median=float(np.median(threshold[normal])),delay_seconds=event['delay_seconds'])
        fault_rows.append(row)
        scored=(ix>=event['start_index'])&(ix<event['end_index_exclusive'])
        assert int(scored.sum())==event['total_points']
        event_y=pred[scored]; alarms=np.flatnonzero(event_y)
        misses=runs(event_y==0)
        event.update(true_positive_points=int(event_y.sum()),raw_coverage=float(event_y.mean()),
            duration_seconds=event['total_points']/20,
            longest_missed_points=max((b-a for a,b in misses),default=0),
            last_alarm_after_onset_seconds=float(alarms[-1]/20) if len(alarms) else None,
            alarm_runs=len(runs(event_y>0)))
        expected=supplied_events[supplied_events.segment_id==event['segment_id']].iloc[0]
        assert event['first_detection_index']==expected.first_detection_index
        assert abs(event['delay_seconds']-expected.delay_seconds)<1e-10
        event_rows.append(event)
    for segment in np.unique(segments):
        mask=segments==segment
        fp_runs.extend(b-a for a,b in runs((pred[mask]==1)&(y[mask]==0)))
        tp_runs.extend(b-a for a,b in runs((pred[mask]==1)&(y[mask]==1)))
    result['normal_alarm_runs']=len(fp_runs); result['anomaly_alarm_runs']=len(tp_runs)
    result['normal_single_point_runs']=int(np.sum(np.array(fp_runs)==1))
    result['anomaly_single_point_runs']=int(np.sum(np.array(tp_runs)==1))
    result['normal_alarm_runs_per_minute']=len(fp_runs)/(int((y==0).sum())/20/60)
    result['delay_mean_seconds']=float(np.mean([r['delay_seconds'] for r in event_rows]))
    result['delay_median_seconds']=float(np.median([r['delay_seconds'] for r in event_rows]))
    # Auditable source, data and checkpoint identities.
    provenance=dict(setting=setting,configuration_fingerprint_matches=build_setting(SimpleNamespace(**config['args']),0)==setting,
        server_commit=config['implementation']['git_commit'],server_branch=config['implementation']['git_branch'],
        source_checks={},data_checks={},checkpoint_sha256=hashlib.sha256((ckpt_dir/'checkpoint.pth').read_bytes()).hexdigest(),
        prior_sha256=config['args']['granger_prior_sha256'],
        test_scope='Local labels/indices/times match, but server config did not hash test feature files')
    for name,expected in config['implementation']['source_sha256'].items():
        raw=(ROOT/name).read_bytes(); lf=raw.replace(b'\r\n',b'\n'); crlf=lf.replace(b'\n',b'\r\n')
        provenance['source_checks'][name]=dict(exact_bytes=hashlib.sha256(raw).hexdigest()==expected,
            lf_match=hashlib.sha256(lf).hexdigest()==expected,crlf_match=hashlib.sha256(crlf).hexdigest()==expected)
        assert any(provenance['source_checks'][name].values())
    for name,expected in config['implementation']['data_sha256'].items():
        provenance['data_checks'][name]=hashlib.sha256((args.data/name).read_bytes()).hexdigest()==expected
        assert provenance['data_checks'][name]
    state=torch.load(ckpt_dir/'checkpoint.pth',map_location='cpu',weights_only=True)
    assert all(torch.isfinite(t).all() for t in state.values())
    routing=[]
    for epoch in range(1,11):
        for r in json.loads((ckpt_dir/f'routing_epoch_{epoch}.json').read_text()):
            assert abs(sum(r['selection_counts'])-3*r['samples'])<1e-8
            for expert,rate in enumerate(r['per_window_selection_rate']):
                routing.append(dict(epoch=epoch,layer=r['block'],expert=expert,rate=rate,entropy=r['entropy'],samples=r['samples']))
    graphs=[]
    for item in json.loads((source/'graphs/manifest.json').read_text()):
        adj=np.load(source/'graphs'/item['file'],allow_pickle=False)
        n=item['num_nodes']; patch=np.arange(n)//10
        illegal=patch[:,None]<patch[None,:]
        prefix=f"blocks.{item['layer']}.experts.{item['expert']}."
        prior=state[prefix+'expanded_prior'].numpy()
        assert adj.shape==(n,n) and np.isfinite(adj).all() and (adj>=0).all()
        assert np.count_nonzero(adj[illegal])==0
        assert np.max(np.abs(adj.sum(1)-1))<1e-6
        graphs.append(dict(layer=item['layer'],expert=item['expert'],patch_size=item['patch_size'],nodes=n,
            future_edges=int(np.count_nonzero(adj[illegal])),nonzero_edges=int(np.count_nonzero(adj)),
            self_weight_mean=float(np.diag(adj).mean()),prior_edge_mass_mean=float((adj*(prior>0)).sum(1).mean())))
    pd.DataFrame(fault_rows).drop(columns='confusion_matrix').to_csv(out/'fault_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(event_rows).to_csv(out/'event_audit.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(routing).to_csv(out/'routing.csv',index=False)
    pd.DataFrame(graphs).to_csv(out/'graph_audit.csv',index=False)
    inventory=[dict(path=str(p.relative_to(args.results)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(args.results.rglob('*')) if p.is_file()]
    pd.DataFrame(inventory).to_csv(out/'input_inventory.csv',index=False,encoding='utf-8-sig')
    summary=dict(metrics=result,threshold_diagnostic=decomposition,provenance=provenance,
        checks=dict(all_arrays_finite=True,labels_indices_times_aligned=True,atssd_predictions_exact=True,
            atssd_thresholds_exact=True,point_adjustment_exact=True,per_fault_metrics_match=True,
            all_anomalies_scored=True,warmup_normal_points=len(test)-len(ix),checkpoint_tensors_finite=True,
            train_val_test_source_files_disjoint=split_disjoint,prior_sources_training_only=prior_training_only,
            train_val_labels_all_normal=True),
        missing_evidence=['train/validation loss curve','best checkpoint epoch','GPU timing and memory','matched E0/E1/E2/E3/E5 full training','additional random seeds'])
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    make_plots(out,meta,aligned,ix,s,threshold,y,pred,events,fault_rows,routing)
    print(json.dumps(dict(setting=setting,metrics=result,threshold_diagnostic=decomposition,output=str(out)),ensure_ascii=False,indent=2))


def make_plots(out,meta,aligned,ix,s,threshold,y,pred,events,fault_rows,routing):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['Engine','Elevator','Rudder-R','Rudder-L','Both ailerons','Right aileron','Left aileron']
    fig,axes=plt.subplots(7,2,figsize=(16,18))
    for i,event in enumerate(events):
        mask=aligned.segment_id.to_numpy()==event['segment_id']; t=aligned.loc[mask,'time_sec'].to_numpy()
        start=meta.time_sec.iloc[event['start_index']]; end=meta.time_sec.iloc[event['end_index_exclusive']-1]+.05
        ax=axes[i,0]; ax.plot(t,np.maximum(s[mask],1e-12),lw=.8,label='Reconstruction score')
        ax.plot(t,np.maximum(threshold[mask],1e-12),lw=.8,label='ATSSD threshold',color='orange')
        hits=pred[mask]>0; ax.scatter(t[hits],s[mask][hits],s=9,color='red',label='Raw alarm',zorder=3)
        ax.axvspan(start,end,color='red',alpha=.10,label='Ground-truth anomaly'); ax.set_yscale('log')
        ax.set(title=f'{labels[i]}: score / threshold',xlabel='Flight time (s)',ylabel='Score (log scale)')
        ax=axes[i,1]; ax.plot(t,s[mask]/np.maximum(threshold[mask],1e-12),lw=.7)
        ax.axhline(1,color='orange',ls='--'); ax.axvspan(start,end,color='red',alpha=.10)
        ax.set(title=f'{labels[i]}: raw recall {pred[mask][y[mask]==1].mean():.2%}',xlabel='Flight time (s)',ylabel='Score / threshold')
    axes[0,0].legend(fontsize=8,loc='upper left'); fig.suptitle('E4 audit: event hits do not imply sustained detection',fontsize=16)
    fig.tight_layout(rect=[0,0,1,.98]); fig.savefig(out/'scores_and_thresholds.png',dpi=130); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,6))
    ordered=fault_rows
    for ax,key,title in zip(axes,['roc_auc','ap','recall'],['ROC-AUC','PR-AUC (Average Precision)','Raw recall / anomaly coverage']):
        values=[r[key] for r in ordered]; ax.barh(labels,values,color='#3478b8'); ax.invert_yaxis()
        ax.set(xlim=(0,1),title=title)
        if key=='roc_auc': ax.axvline(.5,ls='--',color='gray')
        if key=='ap': ax.scatter([r['prevalence'] for r in ordered],range(7),marker='|',color='red',s=150,label='Local anomaly prevalence'); ax.legend(fontsize=8)
        for i,v in enumerate(values): ax.text(min(v+.015,.94),i,f'{v:.3f}',va='center',fontsize=9)
    fig.tight_layout(); fig.savefig(out/'fault_performance.png',dpi=150); plt.close(fig)
    frame=pd.DataFrame(routing); fig,axes=plt.subplots(2,3,figsize=(15,8))
    patches=[[8,12,16,32],[6,8,12,16],[2,6,8,12]]
    for layer in range(3):
        for expert in range(4):
            g=frame[(frame.layer==layer)&(frame.expert==expert)]
            axes[0,layer].plot(g.epoch,g.rate,marker='.',label=f'Expert {expert}, p{patches[layer][expert]}')
        g=frame[(frame.layer==layer)&(frame.expert==0)]
        axes[1,layer].plot(g.epoch,g.entropy,marker='o'); axes[1,layer].axhline(math.log(3),ls='--',color='gray',label='Uniform Top-3 entropy')
        axes[0,layer].set(title=f'Layer {layer}: selection per window',ylim=(0,1.05),xlabel='Epoch'); axes[0,layer].legend(fontsize=8)
        axes[1,layer].set(title=f'Layer {layer}: gate weight entropy',ylim=(0,1.15),xlabel='Epoch'); axes[1,layer].legend(fontsize=8)
    fig.suptitle('Training routing (last epoch is not necessarily best-checkpoint epoch)'); fig.tight_layout(); fig.savefig(out/'routing_training.png',dpi=150); plt.close(fig)


if __name__=='__main__': main()
