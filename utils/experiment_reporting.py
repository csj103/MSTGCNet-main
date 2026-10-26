"""Read-only experiment diagnostics; never changes scores, losses or thresholds."""
import copy
import json
import platform
import hashlib
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def experiment_provenance(args):
    root=Path(__file__).resolve().parents[1]
    def git(*command):
        try:
            return subprocess.check_output(['git',*command],cwd=root,stderr=subprocess.DEVNULL).decode('utf-8').strip()
        except (OSError,subprocess.CalledProcessError):
            return None
    sources=['run.py','models/MSTGCNet.py','models/variable_granger.py','layers/GrangerPrior.py',
             'exp/exp_anomaly_detection.py','data_provider/data_loader.py','utils/atssd.py']
    data_root=Path(args.root_path)
    return dict(workspace=str(root),entrypoint='run.py',git_commit=git('rev-parse','HEAD'),
        git_branch=git('branch','--show-current'),git_status=git('status','--porcelain'),
        source_sha256={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in sources if (root/f).exists()},
        data_sha256={f:hashlib.sha256((data_root/f).read_bytes()).hexdigest()
                     for f in ['train.csv','train_meta.csv','val.csv','val_meta.csv','metadata.json'] if (data_root/f).exists()},
        torch=torch.__version__,numpy=np.__version__,python=platform.python_version())


def parameter_counts(model):
    dormant=[]
    mode=getattr(model,'experiment_mode','E0')
    for name,p in model.named_parameters():
        if name.endswith('node_embeddings') or (mode in {'E1','E3'} and ('.query.' in name or '.key.' in name)):
            dormant.append((name,p.numel()))
    total=sum(p.numel() for p in model.parameters())
    return dict(total=total,effective_differentiable=total-sum(n for _,n in dormant),
        dormant_or_nondifferentiable=dict(dormant))


def event_details(full_labels,pred,indices,meta):
    labels=np.asarray(full_labels).astype(int)
    indices=np.asarray(indices,dtype=int); pred=np.asarray(pred,dtype=int)
    segment=np.zeros(len(labels),dtype=int) if meta is None else meta.segment_id.to_numpy()
    bounds=np.flatnonzero(np.r_[True,segment[1:]!=segment[:-1],True])
    result=[]
    for a,b in zip(bounds[:-1],bounds[1:]):
        changes=np.diff(np.r_[0,labels[a:b]!=0,0].astype(int))
        for start,end in zip(np.flatnonzero(changes==1)+a,np.flatnonzero(changes==-1)+a):
            evaluated=(indices>=start)&(indices<end)
            detections=indices[evaluated&(pred==1)]
            first=int(detections[0]) if len(detections) else None
            row=dict(segment_id=int(segment[start]),start_index=int(start),end_index_exclusive=int(end),
                evaluated_points=int(evaluated.sum()),total_points=int(end-start),hit=first is not None,
                first_detection_index=first,delay_points=first-int(start) if first is not None else None,
                delay_seconds=None)
            if meta is not None:
                if first is not None: row['delay_seconds']=float(meta.time_sec.iloc[first]-meta.time_sec.iloc[start])
                for col in ('source_file','fine_anomaly_type','original_index','original_time_sec'):
                    if col in meta: row['start_'+col]=meta.iloc[start][col]
            result.append(row)
    return result


def benchmark_cpu(model,inputs,repeats=5):
    # Standalone inference timing; reports threads, batch and environment.
    model=copy.deepcopy(model).cpu().eval()
    inputs=inputs.detach().cpu()
    with torch.inference_mode():
        for _ in range(2): model(inputs)
        durations=[]
        for _ in range(repeats):
            start=time.perf_counter(); model(inputs); durations.append(time.perf_counter()-start)
    return dict(batch_size=len(inputs),repeats=repeats,threads=torch.get_num_threads(),
        median_ms_per_batch=float(np.median(durations)*1000),
        median_ms_per_window=float(np.median(durations)*1000/len(inputs)),
        torch=torch.__version__,processor=platform.processor(),device='cpu',
        parameters=sum(p.numel() for p in model.parameters()),parameter_counts=parameter_counts(model))


def graph_snapshot(model,inputs,output):
    """Save a labelled representative sparse-routed batch, never used for scoring."""
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    model.eval()
    for block in model.blocks:
        for expert in block.experts:
            if hasattr(expert,'capture_graph'):
                expert.capture_graph=True; expert.last_adjacency=None
    with torch.no_grad(): model(inputs)
    from scripts.preprocess_alfa import FEATURE_COLUMNS
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest=[]
    for bi,block in enumerate(model.blocks):
        for ei,expert in enumerate(block.experts):
            expert.capture_graph=False
            adj=getattr(expert,'last_adjacency',None)
            if adj is None: continue
            matrix=adj[0].numpy()
            name=f'layer{bi}_expert{ei}_p{expert.patch_size}'
            np.save(output/(name+'.npy'),matrix)
            # Sum across source patches, average across target patches: variable view.
            p=expert.num_patches; v=expert.num_vars
            var=matrix.reshape(p,v,p,v).sum(2).mean(0)
            fig,axes=plt.subplots(1,2,figsize=(14,5))
            axes[0].imshow(matrix,aspect='auto',vmin=0,cmap='viridis')
            axes[0].set(title=name+' patch-major nodes',xlabel='Source node',ylabel='Target node')
            im=axes[1].imshow(var,vmin=0,cmap='viridis')
            axes[1].set(xticks=range(v),yticks=range(v),xticklabels=FEATURE_COLUMNS[:v],
                yticklabels=FEATURE_COLUMNS[:v],title='Aggregated variable propagation',xlabel='Source',ylabel='Target')
            plt.setp(axes[1].get_xticklabels(),rotation=60,ha='right')
            fig.colorbar(im,ax=axes[1]); fig.tight_layout(); fig.savefig(output/(name+'.png'),dpi=100); plt.close(fig)
            selected=getattr(block,'last_selected_indices',{}).get(ei,[])
            manifest.append(dict(layer=bi,expert=ei,patch_size=expert.patch_size,num_nodes=expert.num_nodes,
                representative_batch_index=selected[0] if selected else None,
                node_order='patch*V+variable',matrix_orientation='target_source',file=name+'.npy'))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


def export_evaluation(exp,path,test_data,gt,raw_pred,adjusted_pred,scores,indices,raw,adjusted,roc,pr):
    path=Path(path)
    model=exp.model.module if hasattr(exp.model,'module') else exp.model
    events=event_details(test_data.test_labels,raw_pred,indices,test_data.test_meta)
    pd.DataFrame(events).to_csv(path/'event_details.csv',index=False)
    payload=dict(roc_auc=float(roc),pr_auc_average_precision=float(pr),raw_precision=float(raw[1]),
        raw_recall=float(raw[2]),raw_f1=float(raw[3]),point_adjusted_f1=float(adjusted[3]),
        scored_points=len(gt),event_hits=sum(r['hit'] for r in events),events=len(events),
        parameters=sum(p.numel() for p in model.parameters()),score_mode=exp.args.score_mode,
        parameter_counts=parameter_counts(model),
        threshold_method=exp.args.threshold_method,experiment_mode=exp.args.experiment_mode,
        seed=exp.args.seed,point_adjusted_gt_assisted=True,
        prior_permutation=getattr(model,'prior_permutation',torch.arange(exp.args.enc_in)).cpu().tolist(),
        prior_metadata=getattr(model,'prior_metadata',None))
    if exp.args.report_diagnostics:
        starts=test_data.test_windows[:min(16,len(test_data.test_windows))]
        x=torch.tensor(np.stack([test_data.test[s:s+exp.args.seq_len] for s in starts]),dtype=torch.float32)
        payload['cpu_inference']=benchmark_cpu(model,x[:1])
        payload['graph_snapshots']=graph_snapshot(model,x.to(exp.device),path/'graphs')
        payload['representative_routing_loads']=[b.router.last_expert_load.cpu().tolist() for b in model.blocks]
    (path/'metrics.json').write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8')
    # Extend type reports with score ranking metrics; scores unchanged.
    types=exp._score_anomaly_types(test_data.test_meta,indices)
    if types is not None:
        from sklearn.metrics import roc_auc_score,average_precision_score
        rows=exp._metrics_by_anomaly_type(gt,raw_pred,test_data.test_meta,indices)
        for row in rows:
            mask=types==row['Anomaly Type']
            both=len(np.unique(gt[mask]))==2
            row['ROC-AUC']=float(roc_auc_score(gt[mask],scores[mask])) if both else None
            row['PR-AUC (AP)']=float(average_precision_score(gt[mask],scores[mask])) if both else None
        pd.DataFrame(rows).to_csv(path/'raw_metrics_by_anomaly_type.csv',index=False)
