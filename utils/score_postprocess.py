import numpy as np


def aggregate_feature_errors(errors, strategy="mean", topk=3, feature_scale=None):
    errors = np.asarray(errors, dtype=np.float64)
    if errors.ndim != 2:
        raise ValueError("errors must have shape [time, features]")
    if errors.shape[0] == 0:
        return np.array([], dtype=np.float64)

    topk = max(1, min(int(topk), errors.shape[1]))
    normalized = errors
    if strategy in {"train_feature_topk", "weighted_mean"}:
        if feature_scale is None:
            raise ValueError(f"{strategy} requires feature_scale")
        scale = np.asarray(feature_scale, dtype=np.float64)
        if scale.shape[0] != errors.shape[1]:
            raise ValueError("feature_scale length must match the feature count")
        normalized = errors / np.maximum(scale, 1e-8)

    if strategy == "mean":
        return errors.mean(axis=1)
    if strategy == "max":
        return errors.max(axis=1)
    if strategy == "topk":
        return np.partition(errors, -topk, axis=1)[:, -topk:].mean(axis=1)
    if strategy == "train_feature_topk":
        return np.partition(normalized, -topk, axis=1)[:, -topk:].mean(axis=1)
    if strategy == "weighted_mean":
        return normalized.mean(axis=1)
    raise ValueError(f"Unknown score aggregation strategy: {strategy}")


def _segment_boundaries(length, segment_ids=None):
    if length == 0:
        return np.array([0], dtype=int)
    if segment_ids is None:
        return np.array([0, length], dtype=int)
    segment_ids = np.asarray(segment_ids)
    if len(segment_ids) != length:
        raise ValueError("segment_ids must have the same length as scores")
    return np.flatnonzero(np.r_[True, segment_ids[1:] != segment_ids[:-1], True])


def ewma_by_segment(scores, alpha=0.05, segment_ids=None):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return np.array([], dtype=np.float64)
    alpha = float(np.clip(alpha, 1e-4, 1.0))
    smoothed = np.zeros_like(scores, dtype=np.float64)
    boundaries = _segment_boundaries(len(scores), segment_ids)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if start >= end:
            continue
        smoothed[start] = scores[start]
        for idx in range(start + 1, end):
            smoothed[idx] = alpha * scores[idx] + (1.0 - alpha) * smoothed[idx - 1]
    return smoothed


def _quantile_threshold(scores, quantile):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return 0.0
    quantile = float(np.clip(quantile, 0.0, 1.0))
    return float(np.quantile(scores, quantile))


def _apply_warmup(pred, warmup, segment_ids=None):
    pred = np.asarray(pred, dtype=np.int8).copy()
    warmup = max(0, int(warmup))
    if warmup == 0 or pred.size == 0:
        return pred
    boundaries = _segment_boundaries(len(pred), segment_ids)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        pred[start:min(end, start + warmup)] = 0
    return pred


def dual_time_threshold(
    test_scores,
    validation_scores,
    short_quantile=0.995,
    long_quantile=0.995,
    ewma_alpha=0.05,
    warmup=96,
    segment_ids=None,
):
    test_scores = np.asarray(test_scores, dtype=np.float64)
    validation_scores = np.asarray(validation_scores, dtype=np.float64)
    short_threshold = _quantile_threshold(validation_scores, short_quantile)
    val_long_scores = ewma_by_segment(validation_scores, alpha=ewma_alpha)
    test_long_scores = ewma_by_segment(
        test_scores,
        alpha=ewma_alpha,
        segment_ids=segment_ids,
    )
    long_threshold = _quantile_threshold(val_long_scores, long_quantile)
    short_pred = (test_scores > short_threshold).astype(np.int8)
    long_pred = (test_long_scores > long_threshold).astype(np.int8)
    short_pred = _apply_warmup(short_pred, warmup, segment_ids)
    long_pred = _apply_warmup(long_pred, warmup, segment_ids)
    pred = np.maximum(short_pred, long_pred).astype(np.int8)
    return pred, {
        "short_threshold": short_threshold,
        "long_threshold": long_threshold,
        "short_pred": short_pred,
        "long_pred": long_pred,
        "long_scores": test_long_scores,
        "validation_long_scores": val_long_scores,
    }
