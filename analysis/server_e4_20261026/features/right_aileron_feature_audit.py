"""Bounded CPU reconstruction audit; no training, calibration, or model changes."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import hashlib
import json
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from data_provider.data_loader import Dataset_ALFA
from models.MSTGCNet import Model

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(2025)
    np.random.seed(2025)
    output = Path(__file__).resolve().parent
    checkpoint_dir = next((ROOT / '服务器训练结果/checkpoints').glob('*E4*'))
    results = ROOT / '服务器训练结果/label_results' / checkpoint_dir.name / 'causal_last'
    config = json.loads((checkpoint_dir / 'experiment_config.json').read_text(encoding='utf8'))
    args = SimpleNamespace(**config['args'])
    args.use_gpu = False
    args.use_multi_gpu = False
    args.device = 'cpu'
    args.batch_size = 16
    args.root_path = str(ROOT / 'dataset/ALFA10vars_20hz')
    args.granger_prior = str(ROOT / args.granger_prior)
    dataset = Dataset_ALFA(args, args.root_path, args.seq_len, step=1, flag='test')
    saved_indices = np.load(results / 'test_indices.npy')
    saved_labels = np.load(results / 'test_labels.npy')
    saved_energy = np.load(results / 'test_energy.npy')
    selected = dataset.test_meta.iloc[saved_indices].fine_anomaly_type.to_numpy() == 'Right aileron stuck at zero'
    indices = saved_indices[selected]
    labels = saved_labels[selected]
    starts = indices - args.seq_len + 1
    assert len(indices) == 2566
    assert np.array_equal(labels, dataset.test_labels[indices])
    assert np.isin(starts, dataset.test_windows).all()
    assert dataset.test_meta.iloc[indices].segment_id.nunique() == 1
    model = Model(args).float().cpu()
    state = torch.load(checkpoint_dir / 'checkpoint.pth', map_location='cpu', weights_only=True)
    loaded = model.load_state_dict(state, strict=True)
    model.eval()
    errors = []
    with torch.no_grad():
        for offset in range(0, len(starts), 16):
            batch_starts = starts[offset:offset + 16]
            values = torch.from_numpy(np.stack([dataset.test[s:s+args.seq_len] for s in batch_starts])).float()
            marks = torch.from_numpy(np.stack([dataset.test_mark[s:s+args.seq_len] for s in batch_starts])).float()
            predicted, _ = model(values, marks)
            errors.append((values[:, -1, :] - predicted[:, -1, :]).pow(2).cpu().numpy())
            if offset % 256 == 0:
                print(f'right_aileron_windows={min(offset+16,len(starts))}/{len(starts)} elapsed_seconds={time.perf_counter()-started:.2f}', flush=True)
    errors = np.concatenate(errors)
    local_energy = errors.mean(axis=1)
    reference = saved_energy[selected]
    delta = np.abs(local_energy - reference)
    saved_threshold = np.load(results / 'threshold.npy')[selected]
    saved_pred = np.load(results / 'raw_pred.npy')[selected]
    local_pred_same_threshold = (local_energy > saved_threshold).astype(np.int8)
    normal = labels == 0
    anomaly = labels == 1
    totals = errors.sum(axis=0)
    normal_totals = errors[normal].sum(axis=0)
    anomaly_totals = errors[anomaly].sum(axis=0)
    rows = []
    for column, name in enumerate(args.variable_names):
        feature_errors = errors[:, column]
        rows.append(dict(variable=name, normal_points=int(normal.sum()), anomaly_points=int(anomaly.sum()),
            normal_mean=float(feature_errors[normal].mean()), anomaly_mean=float(feature_errors[anomaly].mean()),
            normal_median=float(np.median(feature_errors[normal])), anomaly_median=float(np.median(feature_errors[anomaly])),
            roc_auc=float(roc_auc_score(labels, feature_errors)), ap=float(average_precision_score(labels, feature_errors)),
            normal_contribution_share=float(normal_totals[column]/normal_totals.sum()),
            anomaly_contribution_share=float(anomaly_totals[column]/anomaly_totals.sum()),
            total_contribution_share=float(totals[column]/totals.sum())))
    table = pd.DataFrame(rows)
    table.to_csv(output / 'right_aileron_feature_metrics.csv', index=False)
    detail = pd.DataFrame(errors, columns=[name + '_squared_error' for name in args.variable_names])
    detail.insert(0, 'label', labels)
    detail.insert(0, 'test_index', indices)
    detail['local_mean_energy'] = local_energy
    detail['saved_mean_energy'] = reference
    detail.to_csv(output / 'right_aileron_point_errors.csv', index=False)
    source_hashes = {name:dict(expected=digest, actual=sha256(ROOT/name),
        local_lf_as_crlf_matches_server=hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')).hexdigest()==digest)
        for name,digest in config['implementation']['source_sha256'].items()}
    data_hashes = {name:dict(expected=digest, actual=sha256(Path(args.root_path)/name)) for name,digest in config['implementation']['data_sha256'].items()}
    report = dict(scope='Only all 2566 previously scored right-aileron-flight windows; no training, no test-label tuning',
        root=str(ROOT), root_resolves_relative_to_script=True,
        checkpoint=str(checkpoint_dir/'checkpoint.pth'), checkpoint_sha256=sha256(checkpoint_dir/'checkpoint.pth'),
        experiment_config_sha256=sha256(checkpoint_dir/'experiment_config.json'), torch_version=torch.__version__,
        numpy_version=np.__version__, device='cpu', threads=torch.get_num_threads(), batch_size=16,
        mode='eval with torch.no_grad', strict_checkpoint_load=str(loaded), scored_windows=len(indices),
        normal_points=int(normal.sum()), anomaly_points=int(anomaly.sum()), segment_id=int(dataset.test_meta.iloc[indices[0]].segment_id),
        source_file=str(dataset.test_meta.iloc[indices[0]].source_file), indices_min=int(indices.min()), indices_max=int(indices.max()),
        labels_equal=True, windows_present_in_Dataset_ALFA=True,
        score_comparison=dict(np_allclose_default=bool(np.allclose(local_energy,reference)),
            allclose_rtol_1e_4_atol_1e_7=bool(np.allclose(local_energy,reference,rtol=1e-4,atol=1e-7)),
            max_abs_error=float(delta.max()), mean_abs_error=float(delta.mean()),
            max_relative_error=float((delta/np.maximum(np.abs(reference),1e-12)).max()),
            abs_error_exceeds_1e_7=int((delta>1e-7).sum()), abs_error_exceeds_1e_6=int((delta>1e-6).sum()),
            abs_error_exceeds_1e_5=int((delta>1e-5).sum()), default_isclose_failures=int((~np.isclose(local_energy,reference)).sum()),
            max_difference_endpoint=int(indices[delta.argmax()]),max_difference_label=int(labels[delta.argmax()]),
            max_difference_local_energy=float(local_energy[delta.argmax()]),max_difference_saved_energy=float(reference[delta.argmax()]),
            decisions_changed_using_identical_saved_threshold=int((local_pred_same_threshold!=saved_pred).sum()),
            decision_change_endpoints=indices[local_pred_same_threshold!=saved_pred].tolist(),
            limitation='Strict allclose fails. CPU/GPU arithmetic and hard TopK boundary sensitivity are possible causes, not established causes.'),
        aggregate_local=dict(roc_auc=float(roc_auc_score(labels,local_energy)), ap=float(average_precision_score(labels,local_energy)),
            normal_mean=float(local_energy[normal].mean()),anomaly_mean=float(local_energy[anomaly].mean()),
            normal_median=float(np.median(local_energy[normal])),anomaly_median=float(np.median(local_energy[anomaly]))),
        aggregate_saved=dict(roc_auc=float(roc_auc_score(labels,reference)),ap=float(average_precision_score(labels,reference))),
        source_hashes=source_hashes,data_hashes=data_hashes,
        feature_metrics=rows, elapsed_seconds=time.perf_counter()-started,
        caveat='Per-variable test-label metrics are post-hoc diagnosis only, not a trained or validated replacement scoring method.')
    (output/'right_aileron_feature_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(table.to_string(index=False),flush=True)
    print(json.dumps({k:report[k] for k in ['score_comparison','aggregate_local','aggregate_saved','elapsed_seconds']},indent=2),flush=True)

if __name__ == '__main__':
    main()
