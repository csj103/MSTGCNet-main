from statistics import NormalDist

import numpy as np


def atssd(scores, window_size=96, alpha=0.01, segment_ids=None):
    """Adaptive Threshold Strategy for Streaming Data.

    This follows the paper's ATSSD procedure: for each time t, build a sliding
    score window H_t, compute its mean and standard deviation, form a baseline
    normal-quantile threshold, and apply an upward trend correction factor.
    """
    scores = np.asarray(scores, dtype=float)
    total = scores.shape[0]
    if total == 0:
        return np.array([], dtype=np.int8), np.array([], dtype=float)

    if segment_ids is not None:
        segment_ids = np.asarray(segment_ids)
        if segment_ids.shape[0] != total:
            raise ValueError("segment_ids must have the same length as scores")
        detections = np.zeros(total, dtype=np.int8)
        thresholds = np.zeros(total, dtype=float)
        boundaries = np.flatnonzero(
            np.r_[True, segment_ids[1:] != segment_ids[:-1], True]
        )
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            segment_pred, segment_threshold = atssd(
                scores[start:end],
                window_size=window_size,
                alpha=alpha,
            )
            detections[start:end] = segment_pred
            thresholds[start:end] = segment_threshold
        return detections, thresholds

    detections = np.zeros(total, dtype=np.int8)
    thresholds = np.zeros(total, dtype=float)
    z_value = NormalDist().inv_cdf(1 - alpha)

    prev_mean = None
    prev_std = None
    for t in range(total):
        if window_size is None or window_size <= 0 or t + 1 <= window_size:
            start = 0
        else:
            start = t - window_size + 1
        history = scores[start:t + 1]

        mean_t = history.mean()
        std_t = history.std(ddof=0)
        base_threshold = mean_t + z_value * std_t

        if prev_mean is None or prev_std is None or prev_mean == 0 or prev_std == 0:
            trend = 0.0
        else:
            delta_mean = (mean_t - prev_mean) / abs(prev_mean)
            delta_std = (std_t - prev_std) / abs(prev_std)
            trend = max(0.0, delta_mean + delta_std)

        threshold = base_threshold * (1 + trend)
        thresholds[t] = threshold
        detections[t] = 1 if scores[t] > threshold else 0
        prev_mean, prev_std = mean_t, std_t

    return detections, thresholds


def causal_atssd(
    scores,
    window_size=96,
    alpha=0.01,
    segment_ids=None,
    min_history=None,
    confirmation=1,
    latch_alarm=False,
):
    """Causal threshold for pointwise detection of persistent anomalies.

    Only previously accepted normal scores update the reference window. This
    prevents a sustained anomaly from raising its own threshold.
    """
    scores = np.asarray(scores, dtype=float)
    total = scores.shape[0]
    if total == 0:
        return np.array([], dtype=np.int8), np.array([], dtype=float)

    if segment_ids is not None:
        segment_ids = np.asarray(segment_ids)
        if segment_ids.shape[0] != total:
            raise ValueError("segment_ids must have the same length as scores")
        detections = np.zeros(total, dtype=np.int8)
        thresholds = np.zeros(total, dtype=float)
        boundaries = np.flatnonzero(
            np.r_[True, segment_ids[1:] != segment_ids[:-1], True]
        )
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            segment_pred, segment_threshold = causal_atssd(
                scores[start:end],
                window_size=window_size,
                alpha=alpha,
                min_history=min_history,
                confirmation=confirmation,
                latch_alarm=latch_alarm,
            )
            detections[start:end] = segment_pred
            thresholds[start:end] = segment_threshold
        return detections, thresholds

    min_history = min_history or window_size
    min_history = max(2, min(min_history, window_size))
    confirmation = max(1, int(confirmation))
    z_value = NormalDist().inv_cdf(1 - alpha)
    detections = np.zeros(total, dtype=np.int8)
    thresholds = np.zeros(total, dtype=float)
    normal_history = []
    candidate_run = 0
    alarm_active = False

    for t, score in enumerate(scores):
        history = np.asarray(normal_history[-window_size:], dtype=float)
        if history.size < 2:
            threshold = score
        else:
            mean_t = history.mean()
            std_t = history.std(ddof=0)
            threshold = mean_t + z_value * max(std_t, 1e-10)
        thresholds[t] = threshold

        if history.size < min_history:
            normal_history.append(score)
            continue

        if alarm_active and latch_alarm:
            detections[t] = 1
            continue

        is_candidate = score > threshold
        if is_candidate:
            candidate_run += 1
        else:
            candidate_run = 0
            normal_history.append(score)

        if candidate_run >= confirmation:
            detections[t] = 1
            alarm_active = True

    return detections, thresholds
