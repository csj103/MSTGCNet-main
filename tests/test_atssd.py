import numpy as np

from utils.atssd import atssd, causal_atssd


def test_causal_atssd_preserves_persistent_anomaly_alarm():
    normal = np.tile(np.array([0.9, 1.0, 1.1, 1.0]), 24)
    anomaly = np.full(96, 5.0)
    scores = np.r_[normal, anomaly]

    paper_pred, _ = atssd(scores, window_size=96, alpha=0.01)
    causal_pred, _ = causal_atssd(scores, window_size=96, alpha=0.01)

    assert causal_pred[-96:].sum() == 96
    assert causal_pred[-96:].sum() > paper_pred[-96:].sum()


def test_causal_atssd_resets_history_between_segments():
    first = np.r_[np.ones(96), np.full(8, 5.0)]
    second = np.ones(104)
    scores = np.r_[first, second]
    segment_ids = np.r_[np.zeros(len(first)), np.ones(len(second))]

    predictions, _ = causal_atssd(
        scores,
        window_size=96,
        alpha=0.01,
        segment_ids=segment_ids,
    )

    assert predictions[len(first):].sum() == 0


def test_confirmation_requires_consecutive_candidates():
    scores = np.r_[np.ones(96), 5.0, 1.0, 5.0, 5.0, 5.0]
    predictions, _ = causal_atssd(
        scores,
        window_size=96,
        alpha=0.01,
        confirmation=3,
    )

    assert predictions[-5:].tolist() == [0, 0, 0, 0, 1]
