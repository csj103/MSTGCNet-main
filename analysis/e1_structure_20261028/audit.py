"""Audit existing E1 without changing production code, parameters, or results."""
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_fscore_support

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from models.MSTGCNet import Model
from utils.atssd import atssd
from utils.alfa_anomaly_types import canonical_anomaly_type_from_row

OUT=Path(__file__).resolve().parent
torch.set_num_threads(2)


def config_for(mode):
    matches=[]
    for path in (ROOT/'服务器训练结果/checkpoints').glob('*/experiment_config.json'):
        c=json.loads(path.read_text(encoding='utf-8'))
        if c['args']['experiment_mode']==mode: matches.append((path,c))
    assert len(matches)==1
    return matches[0]


def make_model(mode,load=True,d_var=None):
    path,c=config_for(mode); args=SimpleNamespace(**c['args'])
    args.use_gpu=False; args.device=torch.device('cpu')
    args.granger_prior=str(ROOT/args.granger_prior)
    if d_var is not None: args.d_var=d_var
    model=Model(args).float().eval()
    if load:
        model.load_state_dict(torch.load(path.parent/'checkpoint.pth',map_location='cpu',weights_only=True),strict=True)
    return model


def param_group(name):
    if 'node_embeddings' in name: return 'fixed_topology_embeddings'
    if '.query.' in name or '.key.' in name: return 'unused_QK_in_E1'
    if 'graph_bias' in name: return 'graph_bias'
    if 'graph_weight' in name: return 'graph_temporal_weight'
    if 'temporal_attn' in name or 'attention_in' in name or 'attention_out' in name: return 'patch_attention'
    if '.router.' in name: return 'router'
    if '.ffn.' in name: return 'FFN'
    if '.norm_' in name: return 'normalization'
    if 'to_vars' in name or 'to_model' in name: return 'cross_variable_projections'
    if name.startswith('reconstruction'): return 'reconstruction_head'
    if name.startswith('encoder') or name.startswith('conv_'): return 'input_encoder'
    return 'other'


def counts(model):
    grouped={}
    for name,p in model.named_parameters():
        k=param_group(name); grouped[k]=grouped.get(k,0)+p.numel()
    total=sum(grouped.values())
    inactive=grouped.get('fixed_topology_embeddings',0)+grouped.get('unused_QK_in_E1',0)
    return dict(total=total,structural_differentiable=total-inactive,groups=grouped,
        graph_bias_share_total=grouped['graph_bias']/total,
        graph_bias_share_differentiable=grouped['graph_bias']/(total-inactive))


def main():
    rows=[]; fault_rows=[]; configurations={}; reference_indices=reference_labels=None
    test_meta=pd.read_csv(ROOT/'dataset/ALFA10vars_20hz/test_meta.csv')
    for mode in ['E0','E1','E2','E3','E4','E5']:
        path,c=config_for(mode); configurations[mode]=c
        folder=ROOT/'服务器训练结果/label_results'/path.parent.name/'causal_last'
        y=np.load(folder/'test_labels.npy'); s=np.load(folder/'test_energy.npy'); pred=np.load(folder/'raw_pred.npy'); ix=np.load(folder/'test_indices.npy')
        exported=json.loads((folder/'metrics.json').read_text())
        if reference_labels is None: reference_labels=y; reference_indices=ix
        else: assert np.array_equal(y,reference_labels) and np.array_equal(ix,reference_indices)
        assert np.array_equal(pred,s>np.load(folder/'threshold.npy'))
        p,r,f,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
        row=dict(mode=mode,roc_auc=float(roc_auc_score(y,s)),ap=float(average_precision_score(y,s)),
            precision=float(p),recall=float(r),f1=float(f),parameters=exported['parameters'],
            tp=int(((y==1)&(pred==1)).sum()),fp=int(((y==0)&(pred==1)).sum()))
        for a,b in [('ap','pr_auc_average_precision'),('f1','raw_f1')]: assert abs(row[a]-exported[b])<1e-12
        rows.append(row)
        if mode in ['E0','E1']:
            types=test_meta.iloc[ix].apply(canonical_anomaly_type_from_row,axis=1).to_numpy()
            supplied=pd.read_csv(folder/'raw_metrics_by_anomaly_type.csv')
            for name in sorted(set(types)):
                mask=types==name
                if not np.any(y[mask]): continue
                pp,rr,ff,_=precision_recall_fscore_support(y[mask],pred[mask],average='binary',zero_division=0)
                fr=dict(mode=mode,fault=name,precision=float(pp),recall=float(rr),f1=float(ff),
                    roc_auc=float(roc_auc_score(y[mask],s[mask])),ap=float(average_precision_score(y[mask],s[mask])),
                    tp=int(((y[mask]==1)&(pred[mask]==1)).sum()),fp=int(((y[mask]==0)&(pred[mask]==1)).sum()))
                sr=supplied[supplied['Anomaly Type']==name].iloc[0]
                assert abs(fr['ap']-sr['PR-AUC (AP)'])<1e-12 and abs(fr['f1']-sr['F-score'])<1e-12
                fault_rows.append(fr)
    for mode,c in configurations.items():
        assert c['implementation']==configurations['E0']['implementation']
        for k in ['root_path','seed','batch_size','seq_len','train_epochs','learning_rate','score_mode','threshold_method','patch_size_list','num_experts_list']:
            assert c['args'][k]==configurations['E0']['args'][k]
    source_checks={}
    for name,expected in configurations['E0']['implementation']['source_sha256'].items():
        raw=(ROOT/name).read_bytes(); lf=raw.replace(b'\r\n',b'\n'); crlf=lf.replace(b'\n',b'\r\n')
        source_checks[name]=dict(raw_match=hashlib.sha256(raw).hexdigest()==expected,
            lf_match=hashlib.sha256(lf).hexdigest()==expected,crlf_match=hashlib.sha256(crlf).hexdigest()==expected)
        assert any(source_checks[name].values())
    models={m:make_model(m) for m in ['E0','E1']}
    topology=[]
    with torch.no_grad():
        for layer,(b0,b1) in enumerate(zip(models['E0'].blocks,models['E1'].blocks)):
            for i,(e0,e1) in enumerate(zip(b0.experts,b1.experts)):
                a=e0._adjacency()>0; b=e1._adjacency(None)>0
                a.fill_diagonal_(False); b.fill_diagonal_(False)
                topology.append(dict(layer=layer,expert=i,patch_size=e1.patch_size,
                    fraction_rows_with_changed_neighbors=float((a!=b).any(-1).float().mean()),
                    mean_neighbor_overlap=float((a&b).sum(-1).float().mean()),neighbors_per_node=e1.knn_k))
    parameter_counts={m:counts(model) for m,model in models.items()}
    parameter_counts['E1_dvar32_untrained_architecture_only']=counts(make_model('E1',load=False,d_var=32))
    kernel=models['E1'].encoder.temporal.weight.detach().double().squeeze(1)
    singular=torch.linalg.svdvals(kernel)
    target=torch.tensor([[1/3,1/3,1/3],[-1,0,1],[-1,1,0]],dtype=torch.float64)
    # Linear information span after subtracting known offsets; not a learned
    # trend detector and not an argument that long-term modeling is sufficient.
    coefficients=torch.linalg.lstsq(kernel.T,target.T).solution
    linear_recovery=(kernel.T@coefficients-target.T).abs().max().item()
    frequencies=[0,.5,1.,2.,5.,10.]
    omega=np.asarray(frequencies)*2*np.pi/20
    response=(kernel.numpy()[:,None,:]*np.exp(-1j*omega[None,:,None]*np.arange(-1,2)[None,None,:])).sum(-1)
    encoder=dict(weight_shape=list(kernel.shape),singular_values=singular.tolist(),
        rank=int(torch.linalg.matrix_rank(kernel)),condition_number=float(singular[0]/singular[-1]),
        linear_three_point_average_difference_recovery_max_error=linear_recovery,
        frequency_hz=frequencies,mean_filter_gain=np.abs(response).mean(0).tolist(),
        scope='Shared 3-point affine encoder; full local rank does not establish long-trend or anomaly-detection capacity')
    torch.manual_seed(2025)
    x=torch.randn(2,96,10)
    perturb=x.clone(); perturb[:,:,3]+=torch.sin(torch.arange(96)*.3)[None,:]
    model=models['E1']; expert=model.blocks[0].experts[0]
    with torch.no_grad():
        h=model.encoder(x); hp=model.encoder(perturb)
        pe=model.position_embedding(h)[:,:,None,:]
        nodes=expert._temporal_encode(expert._patch(h+pe))
        changed_nodes=expert._temporal_encode(expert._patch(hp+pe))
        others=[i for i in range(10) if i!=3]
        other_nodes=torch.arange(expert.num_nodes)%10!=3
        encoder_other_delta=float((h[:,:,others]-hp[:,:,others]).abs().max())
        temporal_other_delta=float((nodes[:,other_nodes]-changed_nodes[:,other_nodes]).abs().max())
        before_graph_delta=float((nodes-changed_nodes).abs().max())
        y1=expert(h+pe)[0]; y2=expert(hp+pe)[0]
        after_graph_other_delta=float((y1[:,:,others]-y2[:,:,others]).abs().max())
        graph_fixed=torch.equal(expert._adjacency(nodes),expert._adjacency(changed_nodes))
    assert encoder_other_delta==0 and temporal_other_delta==0 and before_graph_delta>0
    assert after_graph_other_delta>0 and graph_fixed
    # No optimizer step. Check actual autograd connectivity of the static graph.
    model.zero_grad(set_to_none=True)
    output,aux=model(x); ((output-x).square().mean()+aux).backward()
    node_grad=[p.grad for name,p in model.named_parameters() if 'node_embeddings' in name]
    qk_grad=[p.grad for name,p in model.named_parameters() if '.query.' in name or '.key.' in name]
    assert all(g is None for g in node_grad+qk_grad)
    assert model.encoder.temporal.weight.grad is not None and torch.isfinite(model.encoder.temporal.weight.grad).all()
    interactions=dict(perturbed_variable=3,encoder_other_variable_max_delta=encoder_other_delta,
        pre_graph_temporal_other_variable_max_delta=temporal_other_delta,
        post_graph_expert_other_variable_max_delta=after_graph_other_delta,
        graph_identical_for_changed_input=graph_fixed,
        static_graph_adjacency_requires_grad=expert._adjacency(nodes).requires_grad,
        node_embeddings_without_gradient=len(node_grad),unused_QK_without_gradient=len(qk_grad),
        encoder_gradient_norm=float(model.encoder.temporal.weight.grad.norm()),
        scope='Synthetic local feature-path perturbation, not performance attribution; router couples expert selection separately')
    e0_path,_=config_for('E0')
    ep=ROOT/'服务器训练结果/label_results'/e0_path.parent.name/'causal_last'
    energy=np.load(ep/'test_energy.npy').astype(float); indices=np.load(ep/'test_indices.npy')
    segment=pd.read_csv(ROOT/'dataset/ALFA10vars_20hz/test_meta.csv').segment_id.to_numpy()[indices]
    original_pred,original_threshold=atssd(energy,96,.01,segment)
    scaled_pred,scaled_threshold=atssd(energy*.125,96,.01,segment)
    assert np.array_equal(original_pred,scaled_pred)
    scale_check=dict(scale=.125,predictions_identical=True,
        threshold_scaling_max_error=float(np.max(np.abs(scaled_threshold-original_threshold*.125))),
        note='Lower absolute reconstruction scores alone do not explain lower ROC/AP or ATSSD performance')
    routing=[]
    for mode in ['E0','E1']:
        path,_=config_for(mode)
        for epoch in range(1,11):
            for block in json.loads((path.parent/f'routing_epoch_{epoch}.json').read_text()):
                routing.append(dict(mode=mode,epoch=epoch,layer=block['block'],entropy=block['entropy'],
                    samples=block['samples'],**{f'expert_{i}_rate':r for i,r in enumerate(block['per_window_selection_rate'])}))
    pd.DataFrame(rows).to_csv(OUT/'experiment_metrics.csv',index=False)
    pd.DataFrame(fault_rows).to_csv(OUT/'fault_metrics_E0_E1.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(routing).to_csv(OUT/'training_routing.csv',index=False)
    payload=dict(scope='Existing trained E1 structural diagnosis only; no parameter updates or production-code changes',
        experiment_metrics=rows,fault_metrics_E0_E1=fault_rows,source_checks=source_checks,parameter_counts=parameter_counts,
        encoder=encoder,feature_path_interactions=interactions,
        E0_E1_topology_comparison=topology,atssd_positive_scale_check=scale_check,
        warnings=['No trained d_var=32 comparison','One seed per experiment','Missing train/val loss logs and best epoch'])
    (OUT/'audit_summary.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(payload,indent=2,ensure_ascii=False))


if __name__=='__main__': main()
