import copy
import json
import math

import numpy as np
import pytest
import torch

from models.MSTGCNet import Model
from run import build_parser, normalize_args

torch.set_num_threads(2)


def args_for(mode, prior=None, **overrides):
    args = normalize_args(build_parser().parse_args([
        '--use_gpu', 'false', '--experiment_mode', 'E0' if mode=='E0' else 'E2',
        '--granger_prior', str(prior or ''), '--paper_strict', 'false',
    ]))
    args.experiment_mode=mode
    args._unit_test_synthetic_prior=True  # constructor-level tests; unavailable in formal CLI
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


@pytest.fixture
def prior(tmp_path):
    from scripts.preprocess_alfa import FEATURE_COLUMNS
    matrix = np.zeros((10, 10)); matrix[1, 0] = .9; matrix[2, 1] = .6
    path = tmp_path / 'prior.npz'
    np.savez(path, matrix=matrix, metadata=json.dumps(dict(
        features=FEATURE_COLUMNS, orientation='target_source', max_lag=5,
        method='linear_var', sampling_mode='regular', sampling_hz=20,
        training_only=True, synthetic_test=True)))
    return path


@pytest.mark.parametrize('mode', ['E0', 'E1', 'E2', 'E3', 'E4', 'E5'])
def test_all_modes_shape_and_backward(mode, prior):
    model = Model(args_for(mode, prior)).train()
    x = torch.randn(2, 96, 10)
    seen = []
    hooks = [b.register_forward_hook(lambda m, a, o: seen.append((a[0].shape, o[0].shape))) for b in model.blocks]
    y, aux = model(x)
    loss = (y-x).square().flatten(1).sum(1).mean()+aux
    loss.backward()
    assert y.shape == x.shape and torch.isfinite(y).all()
    assert torch.isfinite(loss)
    if mode != 'E0':
        assert all(a == b == (2, 96, 10, 16) for a,b in seen)
        assert model.encoder.temporal.weight.grad.norm() > 0
        assert model.blocks[0].router.router.weight.grad.norm() > 0
        for block in model.blocks:
            active = [e for e in block.experts if e.graph_weight.grad is not None]
            assert active
            assert all(torch.isfinite(e.graph_weight.grad).all() for e in active)
            if mode in {'E2','E4','E5'}:
                assert all(e.query.weight.grad.norm() > 0 and e.key.weight.grad.norm() > 0 for e in active)
    for h in hooks: h.remove()


def test_encoder_preserves_variables(prior):
    model = Model(args_for('E2', prior)).eval()
    x = torch.randn(2,96,10); z=x.clone(); z[:,:,3] += torch.randn(2,96)
    a=model.encoder(x); b=model.encoder(z)
    others = [i for i in range(10) if i != 3]
    assert torch.equal(a[:,:,others], b[:,:,others])
    assert not torch.equal(a[:,:,3], b[:,:,3])


@pytest.mark.parametrize('p', [2,6,8,12,16,32,7,128])
def test_patch_roundtrip_node_count(prior,p):
    from models.variable_granger import VariableExpert
    e=VariableExpert(args_for('E2',prior), p)
    x=torch.randn(2,96,10,16)
    patch=e._patch(x)
    assert patch.shape==(2,10*math.ceil(96/p),p,16)
    assert torch.equal(e._unpatch(patch),x)
    assert torch.equal(patch[:,12,0],x[:,p,2]) if e.num_patches>1 else True


def test_prior_direction_and_historical_support(prior):
    e=Model(args_for('E4',prior)).blocks[0].experts[0]
    assert e.expanded_prior[11,0] == pytest.approx(.9)
    assert e.expanded_prior[10,1] == 0
    assert e.expanded_prior[1,0] == 0  # no same-patch causal claim
    assert e.expanded_prior[0,11] == 0
    assert e.expanded_prior[21,0] == 0  # outside max_lag=5, patch=8


@pytest.mark.parametrize('mode',['E1','E2','E3','E4','E5'])
def test_future_mask_and_patch_encoding_isolation(prior,mode):
    e=Model(args_for(mode,prior)).blocks[0].experts[0].eval()
    nodes=torch.randn(2,e.num_nodes,e.patch_size,16)
    a=e._adjacency(nodes)
    assert torch.count_nonzero(a[...,~e.causal_mask])==0
    z=nodes.clone(); z[:,10:] += 5*torch.randn_like(z[:,10:])
    h=e._temporal_encode(nodes); hz=e._temporal_encode(z)
    assert torch.equal(h[:,:10],hz[:,:10])
    az=e._adjacency(z)
    if a.ndim == 3:
        assert torch.equal(a[:,:10],az[:,:10])
    y,_=e(e._unpatch(nodes)); yz,_=e(e._unpatch(z))
    assert torch.equal(y[:,:e.patch_size], yz[:,:e.patch_size])


def test_dynamic_depends_on_input_and_zero_lambda(prior):
    a=Model(args_for('E2',prior)).eval()
    b=Model(args_for('E4',prior,lambda_prior=0.)).eval()
    b.load_state_dict(a.state_dict(),strict=False)
    x=torch.randn(2,96,10)
    assert torch.equal(a(x)[0],b(x)[0])
    e=a.blocks[0].experts[0]
    p=e._temporal_encode(e._patch(a.encoder(x)))
    assert not torch.allclose(e._adjacency(p),e._adjacency(torch.randn_like(p)))
    for block in b.blocks:
        for expert in block.experts: expert.expanded_prior.fill_(999)
    assert torch.equal(a(x)[0],b(x)[0])


def test_eval_reproducible_and_dvar32(prior):
    torch.manual_seed(5); a=Model(args_for('E5',prior,d_var=32)).eval()
    torch.manual_seed(5); b=Model(args_for('E5',prior,d_var=32)).eval()
    x=torch.randn(1,96,10)
    assert torch.equal(a(x)[0],a(x)[0])
    assert torch.equal(a(x)[0],b(x)[0])
    assert not torch.equal(a.prior_matrix,b.prior_matrix.roll(1,0))


def test_prior_missing_and_order_rejected(prior):
    with pytest.raises((ValueError,FileNotFoundError)):
        Model(args_for('E4'))
    from layers.GrangerPrior import load_granger_prior
    with pytest.raises(ValueError,match='order'):
        load_granger_prior(prior,['wrong']*10)
    with pytest.raises(ValueError,match='Synthetic'):
        normalize_args(build_parser().parse_args(['--experiment_mode','E4','--granger_prior',str(prior)]))


def test_prior_stays_effective_and_shuffle_is_control(prior):
    a=Model(args_for('E4',prior)).eval()
    b=Model(args_for('E5',prior)).eval()
    idx=b.prior_permutation
    assert torch.equal(b.prior_matrix,a.prior_matrix[idx][:,idx])
    assert not torch.equal(a.prior_matrix,b.prior_matrix)
    e=a.blocks[0].experts[0]
    nodes=torch.randn(2,e.num_nodes,e.patch_size,16)
    e.lambda_prior=100
    weighted=e._adjacency(nodes)
    e.lambda_prior=0
    assert not torch.equal(weighted,e._adjacency(nodes))


def test_checkpoint_roundtrip_preserves_prior(prior,tmp_path):
    a=Model(args_for('E4',prior)).eval(); x=torch.randn(2,96,10)
    checkpoint=tmp_path/'weights.pth'; torch.save(a.state_dict(),checkpoint)
    b=Model(args_for('E4',prior)).eval()
    b.load_state_dict(torch.load(checkpoint,weights_only=True))
    assert torch.equal(a(x)[0],b(x)[0])


def test_layers_consume_updated_variable_features(prior):
    model=Model(args_for('E2',prior)).eval()
    seen=[]
    handles=[block.register_forward_hook(lambda m,a,o: seen.append((a[0].detach().clone(),o[0].detach().clone()))) for block in model.blocks]
    model(torch.randn(2,96,10))
    assert torch.equal(seen[1][0],seen[0][1])
    assert torch.equal(seen[2][0],seen[1][1])
    assert not torch.equal(seen[0][0],seen[1][0])
    for h in handles: h.remove()


def test_reporting_does_not_change_checkpoint_name():
    from run import build_setting
    a=args_for('E2'); b=copy.deepcopy(a); b.report_diagnostics=True
    assert build_setting(a,0)==build_setting(b,0)


def test_existing_train_and_test_pipeline(tmp_path,monkeypatch):
    import pandas as pd
    from scripts.preprocess_alfa import FEATURE_COLUMNS
    from exp.exp_anomaly_detection import Exp_Anomaly_Detection
    rng=np.random.default_rng(3); data_root=tmp_path/'data'; data_root.mkdir()
    for split in ['train','val','test']:
        n=116
        df=pd.DataFrame(rng.normal(size=(n,10)),columns=FEATURE_COLUMNS)
        df.insert(0,'time_sec',np.arange(n)/20)
        df['label']=0
        if split=='test': df.loc[105:,'label']=1; df.loc[105:,FEATURE_COLUMNS] += 4
        df.to_csv(data_root/f'{split}.csv',index=False)
        pd.DataFrame(dict(segment_id=np.zeros(n,int),source_file=[split+'.csv']*n,
            time_sec=df.time_sec,label=df.label,fault_type=['engine']*n,
            original_index=np.arange(n))).to_csv(data_root/f'{split}_meta.csv',index=False)
    (data_root/'metadata.json').write_text(json.dumps(dict(split_policy='fine_grained',features=FEATURE_COLUMNS)),encoding='utf-8')
    args=normalize_args(build_parser().parse_args(['--experiment_mode','E2','--root_path',str(data_root),
        '--use_gpu','false','--paper_strict','false','--train_epochs','1','--batch_size','4',
        '--checkpoints',str(tmp_path/'custom_checkpoints')]))
    monkeypatch.chdir(tmp_path)
    exp=Exp_Anomaly_Detection(args); exp.train('integration'); exp.test('integration',test=1)
    root=tmp_path/'label_results/integration/causal_last'
    metrics=json.loads((root/'metrics.json').read_text())
    assert metrics['scored_points']==21
    assert np.isfinite(metrics['roc_auc']) and np.isfinite(metrics['raw_f1'])
    assert (root/'event_details.csv').exists()
    assert 'PR-AUC (AP)' in pd.read_csv(root/'raw_metrics_by_anomaly_type.csv')
    assert (tmp_path/'custom_checkpoints/integration/routing_epoch_1.json').exists()
