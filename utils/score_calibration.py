import numpy as np


EPS = 1e-8


def _as_feature_matrix(values, name):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"{name} must have shape [time, features]")
    return values


def zscore_calibrate(test_scores, train_normal_scores, eps=EPS):
    test_scores = _as_feature_matrix(test_scores, "test_scores")
    train_normal_scores = _as_feature_matrix(
        train_normal_scores,
        "train_normal_scores",
    )
    if test_scores.shape[1] != train_normal_scores.shape[1]:
        raise ValueError("test and train feature counts must match")
    mean = train_normal_scores.mean(axis=0)
    std = train_normal_scores.std(axis=0)
    scale = np.maximum(std, eps)
    return (test_scores - mean) / scale, {"mean": mean, "std": std}


def robust_zscore_calibrate(test_scores, train_normal_scores, eps=EPS):
    test_scores = _as_feature_matrix(test_scores, "test_scores")
    train_normal_scores = _as_feature_matrix(
        train_normal_scores,
        "train_normal_scores",
    )
    if test_scores.shape[1] != train_normal_scores.shape[1]:
        raise ValueError("test and train feature counts must match")
    median = np.median(train_normal_scores, axis=0)
    mad = np.median(np.abs(train_normal_scores - median), axis=0)
    scale = np.maximum(1.4826 * mad, eps)
    return (test_scores - median) / scale, {"median": median, "mad": mad}


def empirical_tail_calibrate(test_scores, train_normal_scores, eps=EPS):
    test_scores = _as_feature_matrix(test_scores, "test_scores")
    train_normal_scores = _as_feature_matrix(
        train_normal_scores,
        "train_normal_scores",
    )
    if test_scores.shape[1] != train_normal_scores.shape[1]:
        raise ValueError("test and train feature counts must match")

    train_sorted = np.sort(train_normal_scores, axis=0)
    train_count = train_sorted.shape[0]
    tail_probability = np.empty_like(test_scores, dtype=np.float64)
    for feature_index in range(test_scores.shape[1]):
        ranks = np.searchsorted(
            train_sorted[:, feature_index],
            test_scores[:, feature_index],
            side="right",
        )
        tail_probability[:, feature_index] = (
            train_count - ranks + 1.0
        ) / (train_count + 1.0)
    tail_probability = np.clip(tail_probability, eps, 1.0)
    calibrated = -np.log(tail_probability)
    baseline = {
        "train_count": np.full(test_scores.shape[1], train_count, dtype=int),
        "tail_probability": tail_probability,
    }
    return calibrated, baseline


def topk_mean(values, topk):
    values = _as_feature_matrix(values, "values")
    if values.shape[0] == 0:
        return np.array([], dtype=np.float64)
    topk = max(1, min(int(topk), values.shape[1]))
    return np.partition(values, -topk, axis=1)[:, -topk:].mean(axis=1)


def two_sided_calibrate(test_scores, train_normal_scores, mad_floor=0.1, eps=EPS):
    test_scores = _as_feature_matrix(test_scores, "test_scores")
    train = _as_feature_matrix(train_normal_scores, "train_normal_scores")
    if test_scores.shape[1] != train.shape[1] or train.shape[0] == 0:
        raise ValueError("train/test features must match and train must be nonempty")
    mean, std = train.mean(axis=0), train.std(axis=0)
    median = np.median(train, axis=0)
    mad = np.median(np.abs(train - median), axis=0)
    robust_scale = np.maximum(1.4826 * mad, eps)
    floor_scale = np.maximum(robust_scale, mad_floor * std)
    centered = np.abs(test_scores - median)
    n = train.shape[0]
    empirical = np.empty_like(test_scores)
    for feature in range(train.shape[1]):
        sorted_train = np.sort(train[:, feature])
        lower_count = np.searchsorted(sorted_train, test_scores[:, feature], side="right")
        upper_count = n - np.searchsorted(sorted_train, test_scores[:, feature], side="left")
        lower = (lower_count + 1) / (n + 1)
        upper = (upper_count + 1) / (n + 1)
        probability = np.minimum(1.0, 2 * np.minimum(lower, upper))
        empirical[:, feature] = -np.log(np.maximum(probability, eps))
    return {
        "A11-Z": np.abs(test_scores - mean) / np.maximum(std, eps),
        "A11-R0": centered / robust_scale,
        "A11-R1": centered / np.maximum(floor_scale, eps),
        "A11-E": empirical,
    }, {
        "mean": mean, "std": std, "median": median, "mad": mad,
        "robust_floor_scale": floor_scale, "train_count": n,
    }


def topk_selection_frequency(values, topk):
    values = _as_feature_matrix(values, "values")
    if values.shape[0] == 0:
        return np.full(values.shape[1], np.nan)
    topk = max(1, min(int(topk), values.shape[1]))
    selected = np.argsort(-values, axis=1, kind="stable")[:, :topk]
    return np.bincount(selected.ravel(), minlength=values.shape[1]) / len(values)
