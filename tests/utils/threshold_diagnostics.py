from statistics import NormalDist

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def _as_float_array(values):
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _segment_boundaries(total, segment_ids=None):
    if total == 0:
        return np.array([0], dtype=int)
    if segment_ids is None:
        return np.array([0, total], dtype=int)
    segment_ids = np.asarray(segment_ids)
    if segment_ids.shape[0] != total:
        raise ValueError("segment_ids must have the same length as scores")
    return np.flatnonzero(np.r_[True, segment_ids[1:] != segment_ids[:-1], True])


def _normal_threshold(history, alpha):
    if history.size < 2:
        return np.inf
    z_value = NormalDist().inv_cdf(1 - alpha)
    return float(history.mean() + z_value * max(history.std(ddof=0), 1e-10))


def oracle_best_f1(scores, labels):
    scores = _as_float_array(scores)
    labels = np.asarray(labels).astype(int).reshape(-1)
    if scores.shape[0] != labels.shape[0]:
        raise ValueError("scores and labels must have the same length")
    if scores.size == 0:
        return np.array([], dtype=np.int8), 0.0, {}

    candidates = np.unique(scores)
    if candidates.size == 1:
        thresholds = np.array([candidates[0] - 1e-12, candidates[0] + 1e-12])
    else:
        thresholds = (candidates[:-1] + candidates[1:]) / 2.0
        thresholds = np.r_[candidates[0] - 1e-12, thresholds, candidates[-1] + 1e-12]

    best = None
    for threshold in thresholds:
        pred = (scores > threshold).astype(np.int8)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            labels,
            pred,
            average="binary",
            zero_division=0,
        )
        if best is None or f_score > best["F-score"]:
            best = {
                "threshold": float(threshold),
                "Precision": float(precision),
                "Recall": float(recall),
                "F-score": float(f_score),
                "Predicted": int(pred.sum()),
            }

    pred = (scores > best["threshold"]).astype(np.int8)
    return pred, best["threshold"], best


def val_quantile_threshold(test_scores, val_scores, quantile=0.995):
    test_scores = _as_float_array(test_scores)
    val_scores = _as_float_array(val_scores)
    if val_scores.size == 0:
        raise ValueError("val_scores must not be empty")
    threshold = float(np.quantile(val_scores, np.clip(float(quantile), 0.0, 1.0)))
    return (test_scores > threshold).astype(np.int8), threshold


def lagged_atssd(scores, window_size=96, alpha=0.01, segment_ids=None):
    scores = _as_float_array(scores)
    detections = np.zeros(scores.shape[0], dtype=np.int8)
    thresholds = np.full(scores.shape[0], np.inf, dtype=np.float64)
    boundaries = _segment_boundaries(scores.shape[0], segment_ids)
    window_size = max(1, int(window_size))

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        for idx in range(start, end):
            hist_start = max(start, idx - window_size)
            history = scores[hist_start:idx]
            threshold = _normal_threshold(history, alpha)
            thresholds[idx] = threshold
            detections[idx] = 1 if scores[idx] > threshold else 0
    return detections, thresholds


def freeze_atssd(scores, window_size=96, alpha=0.01, segment_ids=None):
    scores = _as_float_array(scores)
    detections = np.zeros(scores.shape[0], dtype=np.int8)
    thresholds = np.full(scores.shape[0], np.inf, dtype=np.float64)
    boundaries = _segment_boundaries(scores.shape[0], segment_ids)
    window_size = max(1, int(window_size))

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        normal_history = []
        for idx in range(start, end):
            history = np.asarray(normal_history[-window_size:], dtype=np.float64)
            threshold = _normal_threshold(history, alpha)
            thresholds[idx] = threshold
            is_anomaly = scores[idx] > threshold
            detections[idx] = 1 if is_anomaly else 0
            if not is_anomaly:
                normal_history.append(scores[idx])
    return detections, thresholds


def median_mad_threshold(scores, window_size=96, mad_k=5.0, segment_ids=None):
    scores = _as_float_array(scores)
    detections = np.zeros(scores.shape[0], dtype=np.int8)
    thresholds = np.full(scores.shape[0], np.inf, dtype=np.float64)
    boundaries = _segment_boundaries(scores.shape[0], segment_ids)
    window_size = max(1, int(window_size))
    mad_k = max(0.0, float(mad_k))

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        for idx in range(start, end):
            hist_start = max(start, idx - window_size)
            history = scores[hist_start:idx]
            if history.size < 2:
                threshold = np.inf
            else:
                median = float(np.median(history))
                mad = float(np.median(np.abs(history - median)))
                robust_std = 1.4826 * max(mad, 1e-10)
                threshold = median + mad_k * robust_std
            thresholds[idx] = threshold
            detections[idx] = 1 if scores[idx] > threshold else 0
    return detections, thresholds
