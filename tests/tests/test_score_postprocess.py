import numpy as np

from utils.score_postprocess import (
    aggregate_feature_errors,
    dual_time_threshold,
    ewma_by_segment,
)


def test_topk_feature_score_keeps_sparse_channel_anomaly_visible():
    errors = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 20.0, 1.0],
        ]
    )

    mean_score = aggregate_feature_errors(errors, strategy="mean", topk=1)
    topk_score = aggregate_feature_errors(errors, strategy="topk", topk=1)

    assert mean_score.tolist() == [1.0, 5.75]
    assert topk_score.tolist() == [1.0, 20.0]


def test_train_feature_topk_normalizes_high_variance_channels():
    errors = np.array([[10.0, 2.0], [10.0, 8.0]])
    feature_scale = np.array([10.0, 2.0])

    scores = aggregate_feature_errors(
        errors,
        strategy="train_feature_topk",
        topk=1,
        feature_scale=feature_scale,
    )

    assert scores.tolist() == [1.0, 4.0]


def test_ewma_resets_between_flight_segments():
    scores = np.array([0.0, 10.0, 0.0, 0.0])
    segments = np.array([0, 0, 1, 1])

    smoothed = ewma_by_segment(scores, alpha=0.5, segment_ids=segments)

    assert smoothed.tolist() == [0.0, 5.0, 0.0, 0.0]


def test_dual_time_threshold_detects_persistent_mild_drift():
    validation_scores = np.r_[np.ones(110), np.full(10, 1.6)]
    test_scores = np.r_[np.ones(120), np.full(80, 1.4)]
    segment_ids = np.zeros(len(test_scores))

    pred, details = dual_time_threshold(
        test_scores,
        validation_scores,
        short_quantile=0.999,
        long_quantile=0.95,
        ewma_alpha=0.05,
        warmup=20,
        segment_ids=segment_ids,
    )

    assert pred[-40:].sum() > 0
    assert details["short_pred"].sum() == 0
    assert details["long_pred"].sum() > 0
    assert details["long_scores"].shape == test_scores.shape
