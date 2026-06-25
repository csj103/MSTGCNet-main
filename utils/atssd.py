from statistics import NormalDist

import numpy as np


def atssd(scores, window_size=100, alpha=0.01):
    """Adaptive Threshold Strategy for Streaming Data.

    This follows the paper's ATSSD procedure: for each time t, build a sliding
    score window H_t, compute its mean and standard deviation, form a baseline
    normal-quantile threshold, and apply an upward trend correction factor.
    """
    scores = np.asarray(scores, dtype=float)
    total = scores.shape[0]
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
