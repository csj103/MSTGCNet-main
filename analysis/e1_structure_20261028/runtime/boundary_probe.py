"""Frozen checkpoint padding-operator sensitivity on the same 128 normal windows.
Run: python -B -X utf8 analysis/e1_structure_20261028/runtime/boundary_probe.py
This is NOT a retrained ablation; switched operators can be out of distribution.
"""
import sys,json,time
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
import probe
import torch
import numpy as np
START=time.monotonic()
ROOT=probe.ROOT
OUT=Path(__file__).resolve().parent
def measure(model,x):
    errors=[]
    with torch.inference_mode():
        for chunk in x.split(8):
            error=(model(chunk)[0]-chunk).square()
            errors.append(error)
            if time.monotonic()-START>25:
                raise RuntimeError('25 second CPU budget exceeded')
    error=torch.cat(errors).double()
    per_position=error.mean((0,2))
    window=error.flatten(1).sum(1)
    last=error[:,-1].mean(1)
    middle=per_position[1:-1].mean()
    return {
      'position_MSE_0_to_95':per_position.tolist(),
      'last_point_MSE_mean':float(last.mean()),
      'last_point_MSE_median':float(last.quantile(.5)),
      'first_point_MSE_mean':float(per_position[0]),
      'middle_positions_1_to_94_MSE_mean':float(middle),
      'last_over_middle_MSE_ratio':float(per_position[-1]/middle),
      'last_over_whole_window_per_element_MSE_ratio':float(per_position[-1]/per_position.mean()),
      'window_SSE_mean':float(window.mean()),
      'last_point_share_of_total_window_SSE':float(error[:,-1].sum()/error.sum()),
      'mean_per_window_last_point_SSE_share':float((error[:,-1].sum(1)/window).mean()),
      'all_finite':bool(torch.isfinite(error).all())}
def main():
    checkpoints=ROOT/'服务器训练结果'/'checkpoints';configs={}
    for p in checkpoints.glob('*/experiment_config.json'):
        c=json.loads(p.read_text(encoding='utf-8'))
        if c['args']['experiment_mode'] in ['E0','E1']:
            configs[c['args']['experiment_mode']]=(p,c)
    inputs,sampling=probe.prepare_data(ROOT/configs['E1'][1]['args']['root_path'])
    all_x=torch.cat(list(inputs.values()))
    result={'seed':2025,'torch':torch.__version__,'device':'cpu','threads':2,'batch_size':8,
      'test_csv_opened':False,'sampling':sampling,'modes':{},
      'interpretation_limit':'Frozen-checkpoint operator sensitivity only, not a retrained ablation. Switching padding can create OOD representations. Results do not establish the cause of detection AP differences.',
      'last_share_definition':'Total squared error over last time point and all windows/variables divided by total window SSE.'}
    for mode,(path,config) in configs.items():
        model=probe.Model(SimpleNamespace(**config['args'])).cpu().eval()
        checkpoint=path.parent/'checkpoint.pth'
        loaded=model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
        for parameter in model.parameters():parameter.requires_grad_(False)
        conv=model.conv_embedding if mode=='E0' else model.encoder.temporal
        original=conv.padding_mode
        switched='replicate' if original=='circular' else 'circular'
        row={'checkpoint':str(checkpoint.relative_to(ROOT)),'checkpoint_sha256':probe.sha(checkpoint),
          'strict_state_dict_loaded':not loaded.missing_keys and not loaded.unexpected_keys,
          'all_weights_frozen':all(not p.requires_grad for p in model.parameters()),
          'operator':'conv_embedding' if mode=='E0' else 'encoder.temporal',
          'original_padding':original,'temporary_padding':switched,'splits':{}}
        try:
            for split,x in list(inputs.items())+[('pooled_normal_128_windows',all_x)]:
                conv.padding_mode=original;before=measure(model,x)
                conv.padding_mode=switched;after=measure(model,x)
                row['splits'][split]={'original':before,'temporary_switch':after,
                  'switch_over_original_last_point_MSE':after['last_point_MSE_mean']/before['last_point_MSE_mean'],
                  'switch_over_original_window_SSE':after['window_SSE_mean']/before['window_SSE_mean']}
        finally:
            conv.padding_mode=original
        row['restored_original_padding']=conv.padding_mode==original
        with torch.inference_mode():
            x=all_x[:8];original_prediction=model(x)[0].clone()
            conv.padding_mode=switched
            try: model(x)
            finally:conv.padding_mode=original
            restored_prediction=model(x)[0]
            row['restore_forward_exact']=bool(torch.equal(original_prediction,restored_prediction))
        result['modes'][mode]=row
    result['elapsed_seconds']=time.monotonic()-START
    (OUT/'boundary_probe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(OUT/'boundary_probe.json'),'elapsed_seconds':result['elapsed_seconds'],
      'results':{m:r['splits']['pooled_normal_128_windows'] for m,r in result['modes'].items()}},ensure_ascii=False))
if __name__=='__main__':main()
