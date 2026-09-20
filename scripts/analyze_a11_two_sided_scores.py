"""Offline A11 diagnosis using the saved A1 No-ReVIN point scores."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.alfa_anomaly_types import anomaly_type_sort_key  # noqa: E402
from utils.score_calibration import (  # noqa: E402
    topk_mean, topk_selection_frequency, two_sided_calibrate,
)
from utils.score_diagnostics import load_feature_names  # noqa: E402


def auc(labels, scores):
    if np.unique(labels).size < 2:
        return np.nan, np.nan
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


def stats(scores):
    return dict(mean=np.mean(scores), median=np.median(scores), std=np.std(scores),
                P95=np.percentile(scores, 95), P99=np.percentile(scores, 99))


def write_csv(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def analyze(result_dir, data_root, output_dir):
    component_dir = result_dir / "score_components"
    diag_dir = result_dir / "score_diagnostics"
    labels = np.load(result_dir / "test_labels.npy").astype(int)
    labels = (labels != 0).astype(int)
    indices = np.load(result_dir / "test_indices.npy").astype(int)
    meta = pd.read_csv(data_root / "test_meta.csv").iloc[indices].reset_index(drop=True)
    kinds = meta["fine_anomaly_type"].astype(str).to_numpy()
    feature_data = {
        "MSE": np.load(component_dir / "s_mse_feature.npy"),
        "NSE": np.load(component_dir / "s_stdres_feature.npy"),
        "logvar": np.load(component_dir / "s_unc_feature.npy"),
        "NLL": np.load(component_dir / "s_nll_feature.npy"),
    }
    train = np.load(diag_dir / "a10_train_s_obs_feature.npy")
    nll = feature_data["NLL"]
    features = load_feature_names(data_root, obs_feature=nll)
    if len(indices) != len(labels) or any(x.shape != nll.shape for x in feature_data.values()):
        raise ValueError("A1 components, test indices and labels are not aligned")
    if train.shape[1] != nll.shape[1] or len(features) != nll.shape[1]:
        raise ValueError("Training and test feature dimensions differ")
    if not np.all(np.isfinite(train)) or any(not np.all(np.isfinite(x)) for x in feature_data.values()):
        raise ValueError("Nonfinite score component encountered")
    if not np.allclose(nll, feature_data["NSE"] / 2 + feature_data["logvar"] / 2 + np.log(2 * np.pi) / 2, atol=1e-4):
        raise ValueError("Saved NLL does not match NSE and logvar")
    fault_types = sorted((set(kinds[labels == 1])), key=anomaly_type_sort_key)
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_rows, feature_rows = [], []
    for feature_index, feature in enumerate(features):
        feature_rows.append({"feature": feature, **{
            f"{part}_{metric}": value
            for part, values in feature_data.items()
            for metric, value in zip(("ROC-AUC", "PR-AUC"), auc(labels, values[:, feature_index]))
        }})
        for fault in fault_types:
            mask = kinds == fault
            row = {"anomaly_type": fault, "feature": feature,
                   "points": int(mask.sum()), "anomaly_points": int(labels[mask].sum())}
            for part, values in feature_data.items():
                part_values = values[:, feature_index]
                roc, pr = auc(labels[mask], part_values[mask])
                row.update({f"{part}_ROC-AUC": roc, f"{part}_PR-AUC": pr,
                            f"{part}_normal_mean": np.mean(part_values[mask & (labels == 0)]),
                            f"{part}_anomaly_mean": np.mean(part_values[mask & (labels == 1)])})
            detail_rows.append(row)
    write_csv(feature_rows, output_dir / "a11_mse_nll_logvar_auc_by_feature.csv")
    write_csv(detail_rows, output_dir / "a11_mse_nll_logvar_auc_by_anomaly_type.csv")

    calibrated, baseline = two_sided_calibrate(nll, train)
    # The A10 method is an already established, one-sided comparison.
    a10_robust = np.load(diag_dir / "a10_robust_mad_feature.npy")
    if a10_robust.shape != nll.shape:
        raise ValueError("A10 comparison scores are not aligned with A1 test scores")
    methods = {"A1_raw_top3": nll, "A8_raw_top2": nll,
               "A10_robust_one_sided": a10_robust, **calibrated}
    score_rows, type_rows, dist_rows, normal_freq, fault_freq = [], [], [], [], []
    for method, matrix in methods.items():
        k = 3 if method == "A1_raw_top3" else 2
        score = topk_mean(matrix, k)
        np.save(output_dir / f"{method}_score.npy", score)
        if method.startswith("A11"):
            np.save(output_dir / f"{method}_feature.npy", matrix)
        roc, pr = auc(labels, score)
        score_rows.append({"method": method, "topk": k, "ROC-AUC": roc, "PR-AUC": pr})
        for label_name, mask in (("normal", labels == 0), ("anomaly", labels == 1)):
            dist_rows.append({"method": method, "group": label_name,
                              "count": int(mask.sum()), **stats(score[mask])})
        for feature, frequency in zip(features, topk_selection_frequency(matrix[labels == 0], k)):
            normal_freq.append({"method": method, "feature": feature, "topk_frequency": frequency})
        for fault in fault_types:
            mask = kinds == fault
            roc, pr = auc(labels[mask], score[mask])
            type_rows.append({"method": method, "anomaly_type": fault, "ROC-AUC": roc,
                              "PR-AUC": pr, "points": int(mask.sum()),
                              "anomaly_points": int(labels[mask].sum())})
            fault_mask = mask & (labels == 1)
            for feature, frequency in zip(features, topk_selection_frequency(matrix[fault_mask], k)):
                fault_freq.append({"method": method, "anomaly_type": fault,
                                   "feature": feature, "topk_frequency": frequency})
    write_csv(score_rows, output_dir / "a11_two_sided_score_auc.csv")
    write_csv(type_rows, output_dir / "a11_two_sided_auc_by_anomaly_type.csv")
    write_csv(dist_rows, output_dir / "a11_two_sided_distribution_stats.csv")
    write_csv(normal_freq, output_dir / "a11_top2_selection_frequency_normal.csv")
    write_csv(fault_freq, output_dir / "a11_top2_selection_frequency_by_anomaly_type.csv")
    write_csv([{"feature": feature, **{key: values[j] for key, values in baseline.items() if isinstance(values, np.ndarray)}}
               for j, feature in enumerate(features)], output_dir / "a11_train_normal_baseline.csv")
    plot_faults(output_dir, meta, labels, kinds, methods)
    print(pd.DataFrame(score_rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return pd.DataFrame(score_rows), pd.DataFrame(type_rows)


def plot_faults(output_dir, meta, labels, kinds, methods):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for fault in ("Right aileron stuck at zero", "Engine full power loss",
                  "Both aileron stuck at zero"):
        matching = np.flatnonzero((kinds == fault) & (labels == 1))
        if not len(matching):
            continue
        first = matching[0]
        segment = meta.iloc[first]["segment_id"]
        take = np.flatnonzero((kinds == fault) & (meta["segment_id"].to_numpy() == segment))
        if not len(take):
            continue
        fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
        for ax, (name, matrix) in zip(axes, (("Raw NLL Top-2", methods["A8_raw_top2"]),
                                               ("A11-R0 two-sided", methods["A11-R0"]))):
            ax.plot(np.arange(len(take)), topk_mean(matrix[take], 2), linewidth=1, label=name)
            ax.set_ylabel(name)
            ax.grid(alpha=0.2)
            shade = ax.twinx()
            shade.fill_between(np.arange(len(take)), 0, labels[take], color="tomato",
                               alpha=0.12, label="GT anomaly")
            shade.set_ylim(0, 1.2)
            shade.set_yticks([])
        axes[-1].set_xlabel("Point within flight segment")
        fig.suptitle(fault)
        fig.tight_layout()
        fig.savefig(output_dir / f"a11_{fault.replace(' ', '_')}_feature_scores.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("dataset/ALFA10vars"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    analyze(args.result_dir, args.data_root,
            args.output_dir or args.result_dir / "score_diagnostics")


if __name__ == "__main__":
    main()
