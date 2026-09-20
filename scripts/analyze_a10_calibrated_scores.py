import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.exp_anomaly_detection import Exp_Anomaly_Detection  # noqa: E402
from run import normalize_args  # noqa: E402
from utils.alfa_anomaly_types import anomaly_type_sort_key  # noqa: E402
from utils.score_calibration import (  # noqa: E402
    empirical_tail_calibrate,
    robust_zscore_calibrate,
    topk_mean,
    zscore_calibrate,
)
from utils.score_diagnostics import load_feature_names  # noqa: E402


def _safe_auc(labels, scores, metric):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    if metric == "roc":
        return float(roc_auc_score(labels, scores))
    return float(average_precision_score(labels, scores))


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "P95": float("nan"),
            "P99": float("nan"),
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "P95": float(np.percentile(values, 95)),
        "P99": float(np.percentile(values, 99)),
    }


def _load_experiment_args(checkpoint_dir, use_gpu=None, batch_size=None, num_workers=None):
    config_path = Path(checkpoint_dir) / "experiment_config.json"
    with config_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)["args"]
    payload.pop("device", None)
    if use_gpu is not None:
        payload["use_gpu"] = bool(use_gpu)
    if batch_size is not None:
        payload["batch_size"] = int(batch_size)
    if num_workers is not None:
        payload["num_workers"] = int(num_workers)
    payload["is_training"] = 0
    payload["threshold_diagnostics"] = False
    payload["router_diagnostics"] = False
    payload["single_expert_diagnostics"] = False
    return normalize_args(Namespace(**payload))


def _load_checkpoint(exp, checkpoint_dir):
    checkpoint_path = Path(checkpoint_dir) / "checkpoint.pth"
    state = torch.load(checkpoint_path, map_location=exp.device)
    exp.model.load_state_dict(state)
    exp.model.eval()


def _compute_train_observation_scores(exp, train_data):
    _, _, _, _, components = exp._causal_last_scores(
        train_data.train,
        train_data.train_mark,
        np.zeros(len(train_data.train), dtype=int),
        train_data.train_windows,
        train_data.train_meta,
        return_components=True,
    )
    if "s_obs_feature" not in components:
        raise KeyError("Model output did not provide s_obs_feature for train data")
    return components["s_obs_feature"]


def _load_test_scores(result_dir):
    result_dir = Path(result_dir)
    component_dir = result_dir / "score_components"
    scores = np.load(component_dir / "s_obs_feature.npy")
    labels = np.load(result_dir / "test_labels.npy").astype(int)
    labels[labels != 0] = 1
    indices = np.load(result_dir / "test_indices.npy").astype(int)
    return scores, labels, indices


def _align_meta(data_root, indices):
    meta_path = Path(data_root) / "test_meta.csv"
    if not meta_path.exists():
        return None
    meta = pd.read_csv(meta_path)
    return meta.iloc[np.asarray(indices, dtype=int)].reset_index(drop=True)


def _score_tables(calibrated_scores, labels, feature_names, aligned_meta, topk):
    score_rows = []
    feature_rows = []
    dist_rows = []
    type_rows = []

    for method, values in calibrated_scores.items():
        total = topk_mean(values, topk)
        score_rows.append(
            {
                "score_name": method,
                "topk": topk,
                "ROC-AUC": _safe_auc(labels, total, "roc"),
                "PR-AUC": _safe_auc(labels, total, "pr"),
            }
        )
        for group_name, mask in [("normal", labels == 0), ("anomaly", labels == 1)]:
            dist_rows.append(
                {
                    "score_name": method,
                    "label_group": group_name,
                    **_stats(total[mask]),
                }
            )
        for index, feature in enumerate(feature_names):
            feature_rows.append(
                {
                    "score_name": method,
                    "feature": feature,
                    "ROC-AUC": _safe_auc(labels, values[:, index], "roc"),
                    "PR-AUC": _safe_auc(labels, values[:, index], "pr"),
                }
            )

        if aligned_meta is not None and "fine_anomaly_type" in aligned_meta.columns:
            anomaly_types = aligned_meta["fine_anomaly_type"].astype(str).to_numpy()
            fault_types = [
                name
                for name in sorted(set(anomaly_types), key=anomaly_type_sort_key)
                if name != "normal" and np.any((anomaly_types == name) & (labels == 1))
            ]
            for fault_type in fault_types:
                mask = anomaly_types == fault_type
                masked_labels = labels[mask]
                type_rows.append(
                    {
                        "anomaly_type": fault_type,
                        "score_name": method,
                        "topk": topk,
                        "ROC-AUC": _safe_auc(masked_labels, total[mask], "roc"),
                        "PR-AUC": _safe_auc(masked_labels, total[mask], "pr"),
                        "points": int(mask.sum()),
                        "anomaly_points": int(masked_labels.sum()),
                    }
                )

    return (
        pd.DataFrame(score_rows),
        pd.DataFrame(feature_rows),
        pd.DataFrame(dist_rows),
        pd.DataFrame(type_rows),
    )


def _save_baselines(output_dir, z_base, robust_base):
    rows = []
    for index in range(len(z_base["mean"])):
        rows.append(
            {
                "feature_index": index,
                "mean": z_base["mean"][index],
                "std": z_base["std"][index],
                "median": robust_base["median"][index],
                "mad": robust_base["mad"][index],
            }
        )
    pd.DataFrame(rows).to_csv(
        Path(output_dir) / "a10_train_normal_feature_baseline.csv",
        index=False,
        encoding="utf-8-sig",
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="A10 calibrated feature-score analysis for DTSGAD."
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("dataset/ALFA10vars"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    output_dir = args.output_dir or (args.result_dir / "score_diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_args = _load_experiment_args(
        args.checkpoint_dir,
        use_gpu=args.use_gpu,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    exp = Exp_Anomaly_Detection(exp_args)
    _load_checkpoint(exp, args.checkpoint_dir)
    train_data, _ = exp._get_data(flag="train")

    train_scores = _compute_train_observation_scores(exp, train_data)
    test_scores, labels, indices = _load_test_scores(args.result_dir)
    feature_names = load_feature_names(args.data_root, obs_feature=test_scores)
    aligned_meta = _align_meta(args.data_root, indices)

    z_scores, z_base = zscore_calibrate(test_scores, train_scores)
    robust_scores, robust_base = robust_zscore_calibrate(test_scores, train_scores)
    empirical_scores, empirical_base = empirical_tail_calibrate(
        test_scores,
        train_scores,
    )
    calibrated_scores = {
        "A10_zscore_topk": z_scores,
        "A10_robust_mad_topk": robust_scores,
        "A10_empirical_tail_topk": empirical_scores,
    }

    np.save(output_dir / "a10_train_s_obs_feature.npy", train_scores)
    np.save(output_dir / "a10_zscore_feature.npy", z_scores)
    np.save(output_dir / "a10_robust_mad_feature.npy", robust_scores)
    np.save(output_dir / "a10_empirical_tail_feature.npy", empirical_scores)
    np.save(output_dir / "a10_empirical_tail_probability.npy", empirical_base["tail_probability"])
    for method, values in calibrated_scores.items():
        np.save(output_dir / f"{method}.npy", topk_mean(values, args.topk))

    _save_baselines(output_dir, z_base, robust_base)
    score_auc, feature_auc, dist_stats, type_auc = _score_tables(
        calibrated_scores,
        labels,
        feature_names,
        aligned_meta,
        args.topk,
    )
    score_auc.to_csv(
        output_dir / "a10_calibrated_score_auc.csv",
        index=False,
        encoding="utf-8-sig",
    )
    feature_auc.to_csv(
        output_dir / "a10_calibrated_feature_auc.csv",
        index=False,
        encoding="utf-8-sig",
    )
    dist_stats.to_csv(
        output_dir / "a10_calibrated_distribution_stats.csv",
        index=False,
        encoding="utf-8-sig",
    )
    type_auc.to_csv(
        output_dir / "a10_calibrated_auc_by_anomaly_type.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("A10 calibrated score AUC")
    print(score_auc.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nSaved to:", output_dir)


if __name__ == "__main__":
    main()
