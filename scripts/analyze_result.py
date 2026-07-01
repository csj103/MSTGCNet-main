import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze saved pointwise results.")
    parser.add_argument("--setting", required=True)
    parser.add_argument("--score-mode", default="causal_last")
    parser.add_argument("--data-root", default="dataset/ALFA10vars")
    return parser.parse_args()


def main():
    args = parse_args()
    result_root = os.path.join("label_results", args.setting, args.score_mode)
    energy = np.load(os.path.join(result_root, "test_energy.npy"))
    threshold = np.load(os.path.join(result_root, "threshold.npy"))
    labels = np.load(os.path.join(result_root, "test_labels.npy")).astype(int)
    predictions = np.load(os.path.join(result_root, "raw_pred.npy")).astype(int)
    indices = np.load(os.path.join(result_root, "test_indices.npy")).astype(int)
    meta = pd.read_csv(os.path.join(args.data_root, "test_meta.csv"))
    segment_ids = meta.iloc[indices]["segment_id"].to_numpy()

    for segment_id in np.unique(segment_ids):
        mask = segment_ids == segment_id
        segment_labels = labels[mask]
        segment_scores = energy[mask]
        segment_thresholds = threshold[mask]
        segment_predictions = predictions[mask]
        has_both_classes = segment_labels.min() != segment_labels.max()
        auc = (
            roc_auc_score(segment_labels, segment_scores)
            if has_both_classes
            else float("nan")
        )
        anomaly_mask = segment_labels == 1
        anomaly_score_median = (
            float(np.median(segment_scores[anomaly_mask]))
            if anomaly_mask.any()
            else float("nan")
        )
        anomaly_threshold_median = (
            float(np.median(segment_thresholds[anomaly_mask]))
            if anomaly_mask.any()
            else float("nan")
        )
        print(
            "segment", int(segment_id),
            "points", len(segment_labels),
            "anomalies", int(anomaly_mask.sum()),
            "detected_anomalies", int(segment_predictions[anomaly_mask].sum()),
            "false_positives", int(segment_predictions[~anomaly_mask].sum()),
            "auc", f"{auc:.4f}",
            "anomaly_score_median", f"{anomaly_score_median:.6f}",
            "anomaly_threshold_median", f"{anomaly_threshold_median:.6f}",
        )


if __name__ == "__main__":
    main()
