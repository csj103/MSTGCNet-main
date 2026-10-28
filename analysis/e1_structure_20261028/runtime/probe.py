"""Bounded CPU, train/normal-val only structural probe. Never opens test.csv.
Run from repository root: python -B -X utf8 analysis/e1_structure_20261028/runtime/probe.py
"""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import sys, json, time, hashlib, math
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from data_provider.data_loader import fill_features_by_segment, build_window_index
from models.MSTGCNet import Model, TrendSeasonalRouter
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
torch.manual_seed(2025)
np.random.seed(2025)
START=time.monotonic()
OUT=Path(__file__).resolve().parent
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rms(t): return float(t.double().square().mean().sqrt())
def stat(t):
    t=t.double()
    return {'rms':rms(t),'global_variance':float(t.var(unbiased=False)),
      'time_variance_mean':float(t.var(dim=1,unbiased=False).mean()),
      'window_variance_mean':float(t.var(dim=0,unbiased=False).mean())}
def spectral(t):
    a=t.double().reshape(-1,t.shape[-1]); a=a-a.mean(0)
    s=torch.linalg.svdvals(a); p=s.square()/s.square().sum().clamp_min(1e-30)
    return {'singular_values':s.tolist(),'numerical_rank_relative_1e-6':int((s>s.max()*1e-6).sum()),
     'effective_rank_covariance':float((-(p*p.clamp_min(1e-30).log()).sum()).exp()),
     'note':'Rows flatten window/time/(variable for E1); width and row axes differ; not a capacity or performance comparison.'}
def sample_windows(values,meta,count=64):
    starts=build_window_index(meta,len(values),96,1)
    groups={}
    for s in starts: groups.setdefault(str(meta.iloc[s].segment_id),[]).append(s)
    rng=np.random.default_rng(2025)
    choices={k:list(rng.permutation(v)) for k,v in groups.items()}
    selected=[]; ids=[]
    while len(selected)<count:
        progress=False
        for k in choices:
            if choices[k] and len(selected)<count:
                selected.append(int(choices[k].pop()));ids.append(k);progress=True
        if not progress:break
    return torch.tensor(np.stack([values[s:s+96] for s in selected]),dtype=torch.float32),{
      'starts':selected,'segment_ids':ids,'eligible_segments':len(groups),'covered_segments':len(set(ids)),
      'sampling':'seed2025 equal round-robin across eligible segments; random no-replacement within segment; windows can overlap'}
def prepare_data(folder):
    frames={};metas={}
    for split in ['train','val']:
        frame=pd.read_csv(folder/f'{split}.csv');meta=pd.read_csv(folder/f'{split}_meta.csv')
        normal=frame.label==0
        frame=frame.loc[normal].reset_index(drop=True);meta=meta.loc[normal].reset_index(drop=True)
        frames[split]=fill_features_by_segment(frame,meta);metas[split]=meta
    scaler=StandardScaler().fit(frames['train'].iloc[:,1:-1].to_numpy(dtype=np.float32))
    result={};info={}
    for split in frames:
        vals=scaler.transform(frames[split].iloc[:,1:-1].to_numpy(dtype=np.float32))
        result[split],info[split]=sample_windows(vals,metas[split]);info[split]['normal_rows']=len(vals)
    return result,info
def encoding(model,x,mode):
    z=model.revin_layer(x,'norm') if model.revin else x
    if mode=='E0':
        data=model.conv_scale*model.conv_embedding(z.transpose(1,2)).transpose(1,2)
        pe=model.position_embedding(data).expand_as(data)
        parts={'data':data,'position':pe};full=data+pe
    else:
        b,l,v=z.shape
        # Conv bias is constant, so expose it instead of silently calling it data.
        signal=torch.nn.functional.conv1d(torch.nn.functional.pad(z.transpose(1,2).reshape(b*v,1,l),(1,1),mode='replicate'),model.encoder.temporal.weight,None)
        data=signal.reshape(b,v,-1,l).permute(0,3,1,2)
        bias=model.encoder.temporal.bias[None,None,None,:].expand_as(data)
        identity=model.encoder.identity[None,None].expand_as(data)
        pe=model.position_embedding(data)[:,:,None,:].expand_as(data)
        parts={'data':data,'conv_bias':bias,'identity':identity,'position':pe};full=sum(parts.values())
    result={'parts':{k:stat(v) for k,v in parts.items()},'sum':stat(full),
      'data_spectrum':spectral(data),'sum_spectrum':spectral(full),'cross_mean_products':{}}
    for i,(k,a) in enumerate(parts.items()):
        for j,(q,b) in enumerate(parts.items()):
            if j>i:result['cross_mean_products'][k+'__'+q]=float((a.double()*b.double()).mean())
    # E1 identity relative to window-variable signal; only a scale diagnostic.
    return result,full
def run(model,x,mode):
    gates=[[] for _ in model.blocks];updates=[[] for _ in model.blocks]
    previous=[None];first=[];handles=[]
    for i,block in enumerate(model.blocks):
        def bh(module,inputs,output,i=i):
            a=inputs[0].detach();b=output[0].detach();delta=b-a
            row={'shape_in':list(a.shape),'shape_out':list(b.shape),'input_rms':rms(a),'output_rms':rms(b),
              'delta_rms':rms(delta),'delta_relative_rms':rms(delta)/max(rms(a),1e-30),
              'min_sample_delta_relative_rms':float(delta.flatten(1).norm(dim=1).div(a.flatten(1).norm(dim=1).clamp_min(1e-30)).min()),
              'next_input_exactly_previous_output':None if i==0 else bool(torch.equal(a,previous[0]))}
            previous[0]=b.clone();updates[i].append(row)
            if i==0:first.append(a.clone())
        handles.append(block.register_forward_hook(bh))
        def rh(module,inputs,output,i=i): gates[i].append(output[0].detach().clone())
        handles.append(block.router.register_forward_hook(rh))
    preds=[]
    for c in x.split(8):
        previous[0]=None;preds.append(model(c)[0])
        if time.monotonic()-START>55:raise RuntimeError('55 second bound exceeded; abort rather than unbounded CPU')
    for h in handles:h.remove()
    layers=[]
    for i,gg in enumerate(gates):
        g=torch.cat(gg); entropy=-(g*g.clamp_min(1e-12).log()).sum(1)
        u=updates[i]
        layers.append({'layer':i+1,'patch_sizes':[e.patch_size for e in model.blocks[i].experts],
          'mean_gate':g.mean(0).tolist(),'selection_rate':(g>0).float().mean(0).tolist(),
          'std_gate_across_windows':g.std(0,unbiased=False).tolist(),
          'argmax_rate':torch.bincount(g.argmax(1),minlength=g.shape[1]).float().div(len(g)).tolist(),
          'mean_max_gate':float(g.max(1).values.mean()),'mean_entropy_nats':float(entropy.mean()),
          'entropy_over_log_topk':float(entropy.mean()/math.log(model.blocks[i].top_k)),
          'chain_all_exact':all(z['next_input_exactly_previous_output'] is not False for z in u),
          'mean_update_relative_rms':float(np.mean([z['delta_relative_rms'] for z in u])),
          'min_sample_update_relative_rms':min(z['min_sample_delta_relative_rms'] for z in u),
          'chunk_update_details':u})
    pred=torch.cat(preds)
    error=(pred-x).square()
    sse=error.flatten(1).sum(1)
    last=error[:,-1].mean(1)
    result={'layers':layers,'normal_reconstruction_mse':float((pred-x).square().mean()),
      'normal_reconstruction_mse_last_position':float((pred[:,-1]-x[:,-1]).square().mean()),
      'training_reconstruction_window_SSE_mean':float(sse.mean()),
      'window_SSE_per_window':sse.tolist(),
      'last_point_MSE_mean':float(last.mean()),
      'last_point_MSE_median':float(last.quantile(0.5)),
      'last_point_MSE_per_window':last.tolist(),
      'loss_definition':'exp_anomaly_detection._reconstruction_loss: squared error summed over 96 time points and 10 variables, then averaged over windows; excludes balance loss.',
      'warning':'Normal reconstruction and gating descriptive only; no test fault labels or detection conclusions.'}
    if mode=='E1':
        h=torch.cat(first);b,l,v,d=h.shape;r=model.blocks[0].router
        scores,_=TrendSeasonalRouter._routing_features(r,h.permute(0,2,1,3).reshape(b*v,l,d))
        scores=scores.reshape(b,v,l)
        centered=scores-scores.mean(-1,keepdim=True)
        ratio=centered.mean(1).square().mean(-1).sqrt()/centered.square().mean(-1).sqrt().mean(1).clamp_min(1e-30)
        result['first_router_variable_time_centered_cancellation']={'mean_ratio_rms_mean_over_mean_rms':float(ratio.mean()),
          'min':float(ratio.min()),'max':float(ratio.max()),
          'definition':'Per window: RMS_t(mean_v(score_vt - mean_t score_vt)) / mean_v RMS_t(score_vt - mean_t score_vt). Smaller means stronger cancellation/averaging of time variation; not causal evidence.'}
        # Remove shared PE/bias/identity counterfactual to avoid mistaking a
        # common deterministic positional waveform for informative agreement.
        static=model.encoder(torch.zeros_like(x[:1]))
        static=static+model.position_embedding(static)[:,:,None,:]
        static_scores,_=TrendSeasonalRouter._routing_features(r,static.permute(0,2,1,3).reshape(v,l,d))
        difference=scores-static_scores[None]
        difference=difference-difference.mean(-1,keepdim=True)
        dratio=difference.mean(1).square().mean(-1).sqrt()/difference.square().mean(-1).sqrt().mean(1).clamp_min(1e-30)
        result['first_router_zero_normalized_signal_counterfactual']={
          'mean_variable_cancellation_ratio':float(dratio.mean()),'min_ratio':float(dratio.min()),'max_ratio':float(dratio.max()),
          'full_per_variable_score_rms':rms(scores),'data_response_per_variable_score_rms':rms(difference),
          'data_response_mean_variable_score_rms':rms(difference.mean(1)),
          'note':'Subtract routing scores at normalized-signal=0 while retaining learned conv bias, identity and PE, then time-center. Nonlinear counterfactual response, not an additive attribution.'}
        e=model.blocks[0].experts[0];small=h[:1];nodes=e._temporal_encode(e._patch(small));adj=e._adjacency(nodes)
        isolated=small.clone();isolated[:,:,1:]=0
        n2=e._temporal_encode(e._patch(isolated));adj2=e._adjacency(n2)
        result['E1_fixed_graph_isolation']={'expert':0,'patch_size':e.patch_size,'adjacency_shape':list(adj.shape),
          'zero_variables_1_to_9_max_adjacency_change':float((adj-adj2).abs().max()),
          'note':'Input-isolation diagnostic of fixed adjacency only; not graph-propagation ablation; original tensors and checkpoint untouched.'}
    return result
def main():
    checkpoints=ROOT/'服务器训练结果'/'checkpoints';configs={}
    for p in checkpoints.glob('*/experiment_config.json'):
        c=json.loads(p.read_text(encoding='utf-8'));m=c['args']['experiment_mode']
        if m in ['E0','E1']:configs[m]=(p,c)
    inputs,sampling=prepare_data(ROOT/configs['E1'][1]['args']['root_path'])
    result={'seed':2025,'torch':torch.__version__,'device':'cpu','threads':2,'batch_size':8,
      'test_csv_opened':False,'sampling':sampling,'modes':{},'data_compatibility':{}}
    folder=ROOT/configs['E1'][1]['args']['root_path']
    for mode,(path,c) in configs.items():
        args=SimpleNamespace(**c['args']);model=Model(args).cpu().eval()
        ck=path.parent/'checkpoint.pth';state=torch.load(ck,map_location='cpu',weights_only=True)
        loaded=model.load_state_dict(state,strict=True)
        datahash={name:{'sha256':sha(folder/name),'matches_saved_config':sha(folder/name)==expected} for name,expected in c['implementation']['data_sha256'].items()}
        sourcehash={}
        for name,expected in c['implementation']['source_sha256'].items():
            content=(ROOT/name).read_bytes()
            crlf=hashlib.sha256(content.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')).hexdigest()
            sourcehash[name]={'sha256':sha(ROOT/name),'raw_matches_saved_config':sha(ROOT/name)==expected,
              'CRLF_sha256':crlf,'CRLF_matches_saved_config':crlf==expected,
              'note':'Canonicalize source line endings to CRLF to compare with saved server-source hash.'}
        result['modes'][mode]={'checkpoint':str(ck.relative_to(ROOT)),'checkpoint_sha256':sha(ck),
          'config':str(path.relative_to(ROOT)),'strict_state_dict_loaded':not loaded.missing_keys and not loaded.unexpected_keys,
          'parameters':sum(p.numel() for p in model.parameters()),'source_compatibility':sourcehash,
          'data_compatibility':datahash,'splits':{}}
        with torch.inference_mode():
            for split,x in inputs.items():
                enc,_=encoding(model,x,mode)
                result['modes'][mode]['splits'][split]={'encoding':enc,**run(model,x,mode)}
    result['elapsed_seconds']=time.monotonic()-START
    result['pooled_normal_128_windows']={}
    for mode,row in result['modes'].items():
        sse=np.concatenate([q['window_SSE_per_window'] for q in row['splits'].values()])
        last=np.concatenate([q['last_point_MSE_per_window'] for q in row['splits'].values()])
        result['pooled_normal_128_windows'][mode]={'training_reconstruction_window_SSE_mean':float(sse.mean()),
          'last_point_MSE_mean':float(last.mean()),'last_point_MSE_median':float(np.median(last))}
    (OUT/'probe.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'output':str(OUT/'probe.json'),'elapsed_seconds':result['elapsed_seconds'],
      'data_matches':{m:all(z['matches_saved_config'] for z in r['data_compatibility'].values()) for m,r in result['modes'].items()}},ensure_ascii=False))
if __name__=='__main__':main()
