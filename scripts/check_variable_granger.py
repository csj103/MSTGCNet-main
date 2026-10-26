"""CPU structural smoke, then bounded real normal-window training (not a benchmark)."""
import argparse
import importlib.util
import json
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from data_provider.data_loader import Dataset_ALFA
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from models.MSTGCNet import Model
from run import build_parser,build_setting,normalize_args
from utils.experiment_reporting import benchmark_cpu,graph_snapshot,experiment_provenance

BASE='5382d528ff93d03f5a30e3b34563c4e70803f5e0'


def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)


def verify_baseline(args):
    source=subprocess.check_output(['git','show',BASE+':models/MSTGCNet.py'],cwd=ROOT).decode('utf-8')
    run_source=subprocess.check_output(['git','show',BASE+':run.py'],cwd=ROOT).decode('utf-8')
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'original.py'; path.write_text(source,encoding='utf-8')
        spec=importlib.util.spec_from_file_location('original_v12',path)
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        seed(args.seed); old=module.Model(args).eval()
        seed(args.seed); new=Model(args).eval()
        old_state,new_state=old.state_dict(),new.state_dict()
        assert old_state.keys()==new_state.keys()
        assert all(torch.equal(v,new_state[k]) for k,v in old_state.items())
        x=torch.randn(2,96,10)
        with torch.no_grad():
            a=old(x); b=new(x)
        assert all(torch.equal(x,y) for x,y in zip(a,b))
        path2=Path(tmp)/'original_run.py'; path2.write_text(run_source,encoding='utf-8')
        spec2=importlib.util.spec_from_file_location('original_run',path2)
        module2=importlib.util.module_from_spec(spec2); spec2.loader.exec_module(module2)
        # Verify original defaults naming with original root_path as well.
        old_args=module2.normalize_args(module2.build_parser().parse_args(['--use_gpu','false']))
        new_args=normalize_args(build_parser().parse_args(['--use_gpu','false']))
        assert module2.build_setting(old_args,0)==build_setting(new_args,0)
    return dict(commit=BASE,state_dict_bit_exact=True,output_and_aux_bit_exact=True,
        default_setting_bit_exact=True,parameters=sum(p.numel() for p in new.parameters()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='dataset/ALFA10vars_20hz')
    p.add_argument('--prior',default='analysis/variable_granger_v1/prior/granger_prior.npz')
    p.add_argument('--output',type=Path,default=Path('analysis/variable_granger_v1/verification'))
    p.add_argument('--steps',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=2025)
    p.add_argument('--evaluate',action='store_true',help='Also score entire test split after tiny training; not full-training results')
    a=p.parse_args(); torch.set_num_threads(a.threads)
    if a.steps<1 or a.batch_size<1: raise ValueError('Positive steps/batch-size required')
    a.output.mkdir(parents=True,exist_ok=True)
    result=dict(scope='CPU smoke + bounded normal-training windows; no full training or performance claim',
        seed=a.seed,threads=a.threads,torch=torch.__version__,modes={})
    base_args=normalize_args(build_parser().parse_args(['--use_gpu','false']))
    result['baseline']=verify_baseline(base_args)
    # First complete structural forward/backward for ALL modes; only then train.
    for mode in ['E0','E1','E2','E3','E4','E5']:
        seed(a.seed)
        args=normalize_args(build_parser().parse_args(['--use_gpu','false','--experiment_mode',mode,
            '--root_path',a.root,'--granger_prior',a.prior,'--paper_strict','false','--batch_size',str(a.batch_size)]))
        model=Model(args).train(); x=torch.randn(2,96,10)
        y,aux=model(x); loss=(y-x).square().flatten(1).sum(1).mean()+aux; loss.backward()
        assert y.shape==x.shape and torch.isfinite(loss)
        grads={name:float(v.grad.norm()) for name,v in model.named_parameters() if v.grad is not None}
        assert all(np.isfinite(list(grads.values())))
        dynamic=[v for k,v in grads.items() if '.query.' in k or '.key.' in k]
        if mode in {'E2','E4','E5'}: assert dynamic and min(dynamic)>0
        model.eval()
        with torch.no_grad(): assert torch.equal(model(x)[0],model(x)[0])
        stats=dict(shape=list(y.shape),forward_backward_finite=True,eval_repeat_bit_exact=True,
            dynamic_grad_min=min(dynamic) if dynamic else None,gradient_norms=grads,
            cpu_inference=benchmark_cpu(model,x[:1]),
            nodes=[[e.num_nodes for e in b.experts] for b in model.blocks])
        result['modes'][mode]=stats
        print(mode,'smoke',dict(parameters=stats['cpu_inference']['parameters'],
            cpu_ms_per_window=stats['cpu_inference']['median_ms_per_window'],
            effective_parameters=stats['cpu_inference']['parameter_counts']['effective_differentiable']),flush=True)
    for mode in ['E1','E2','E3','E4']:
        seed(a.seed)
        args=normalize_args(build_parser().parse_args(['--use_gpu','false','--experiment_mode',mode,
            '--root_path',a.root,'--granger_prior',a.prior,'--paper_strict','false',
            '--batch_size',str(a.batch_size),'--report_diagnostics','true']))
        ds=Dataset_ALFA(args,args.root_path,96,flag='train')
        if not len(ds.train_windows) or not len(ds.val_windows): raise ValueError('Missing real training/validation windows')
        # Fixed distributed normal windows, same order for E1--E4. No test values.
        rng=np.random.default_rng(a.seed)
        chosen=rng.choice(len(ds.train_windows),size=a.steps*a.batch_size,replace=False)
        validation=np.linspace(0,len(ds.val_windows)-1,num=min(32,len(ds.val_windows)),dtype=int)
        model=Model(args).train(); optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
        history=[]; start=time.perf_counter()
        for step in range(a.steps):
            ids=chosen[step*a.batch_size:(step+1)*a.batch_size]
            batch=torch.tensor(np.stack([ds[int(i)][0] for i in ids]))
            optimizer.zero_grad(); y,aux=model(batch)
            rec=Exp_Anomaly_Detection._reconstruction_loss(y,batch)
            loss=rec+aux; assert torch.isfinite(loss); loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            optimizer.step()
            history.append(dict(step=step+1,reconstruction=float(rec.detach()),balance=float(aux.detach())))
        model.eval(); losses=[]
        with torch.no_grad():
            for off in range(0,len(validation),a.batch_size):
                batch=torch.tensor(np.stack([ds.val[ds.val_windows[i]:ds.val_windows[i]+96] for i in validation[off:off+a.batch_size]]))
                y,_=model(batch); losses.append(float(Exp_Anomaly_Detection._reconstruction_loss(y,batch)))
        folder=a.output/mode; folder.mkdir(exist_ok=True)
        snapshot=torch.tensor(np.stack([ds[int(i)][0] for i in chosen[:min(16,len(chosen))]]))
        graph_snapshot(model,snapshot,folder/'graphs')
        routing=[dict(load=b.router.last_expert_load.cpu().tolist(),entropy=float(b.router.last_entropy)) for b in model.blocks]
        train_report=dict(steps=a.steps,batch_size=a.batch_size,training_window_ids=chosen.tolist(),
            training_window_starts=[ds.train_windows[int(i)] for i in chosen],validation_window_ids=validation.tolist(),
            validation_reconstruction=float(np.mean(losses)),history=history,
            elapsed_seconds=time.perf_counter()-start,routing_representative_batch=routing,
            parameters=sum(p.numel() for p in model.parameters()),
            prior_metadata=getattr(model,'prior_metadata',{}),args=vars(args))
        train_report['implementation']=experiment_provenance(args)
        assert np.isfinite(train_report['validation_reconstruction'])
        (folder/'small_training.json').write_text(json.dumps(train_report,indent=2,default=str),encoding='utf-8')
        torch.save(model.state_dict(),folder/'smoke_checkpoint.pth')
        result['modes'][mode]['small_training']=dict(steps=a.steps,validation_reconstruction=train_report['validation_reconstruction'],
            report=str(folder/'small_training.json'))
        if a.evaluate:
            exp=Exp_Anomaly_Detection(args); exp.model=model
            exp.test('variable_granger_cpu_smoke_'+mode)
        print(mode,'tiny normal training validation reconstruction',train_report['validation_reconstruction'],flush=True)
    (a.output/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__': main()
