from types import SimpleNamespace

import numpy as np
import pandas as pd

from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from run import build_parser, build_setting, normalize_args
from utils.threshold_diagnostics import (
    freeze_atssd,
    lagged_atssd,
    median_mad_threshold,
    oracle_best_f1,
    val_quantile_threshold,
)


def test_oracle_best_f1_uses_labels_to_find_upper_bound():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])

    pred, threshold, info = oracle_best_f1(scores, labels)

    assert pred.tolist() == [0, 0, 1, 1]
    assert 0.2 <= threshold < 0.8
    assert info["F-score"] == 1.0


def test_val_quantile_threshold_uses_validation_scores_only():
    val_scores = np.array([1.0, 2.0, 3.0, 4.0])
    test_scores = np.array([2.5, 3.5, 4.5])

    pred, threshold = val_quantile_threshold(
        test_scores,
        val_scores,
        quantile=0.75,
    )

    assert threshold == 3.25
    assert pred.tolist() == [0, 1, 1]


def test_lagged_atssd_does_not_put_current_score_in_threshold_history():
    scores = np.r_[np.ones(96), 10.0]

    pred, threshold = lagged_atssd(scores, window_size=96, alpha=0.01)

    assert threshold[-1] < 2.0
    assert pred[-1] == 1


def test_freeze_atssd_does_not_let_anomaly_raise_future_threshold():
    scores = np.r_[np.ones(96), np.full(10, 10.0)]

    pred, threshold = freeze_atssd(scores, window_size=96, alpha=0.01)

    assert pred[-10:].sum() == 10
    assert np.all(threshold[-10:] < 2.0)


def test_median_mad_threshold_is_robust_to_a_single_large_history_outlier():
    scores = np.r_[np.ones(95), 100.0, 5.0]

    pred, threshold = median_mad_threshold(scores, window_size=96, mad_k=3.0)

    assert threshold[-1] < 2.0
    assert pred[-1] == 1


def test_threshold_diagnostics_report_global_and_type_metrics():
    args = SimpleNamespace(
        winsize=2,
        alpha=0.01,
        alarm_confirmation=1,
        latch_alarm=0,
        threshold_adaptation_clip=2.0,
        val_quantile=0.5,
        val_quantiles=[0.25, 0.5, 0.75],
        mad_k=3.0,
        mad_ks=[2.0, 3.0],
    )
    exp = object.__new__(Exp_Anomaly_Detection)
    exp.args = args
    scores = np.array([0.1, 0.2, 1.0, 0.3])
    labels = np.array([0, 0, 1, 0])
    segments = np.array([0, 0, 0, 0])
    meta = pd.DataFrame(
        {
            "source_file": [
                "normal.csv",
                "normal.csv",
                "carbonZ_2018-09-11-14-22-07_2_engine_failure.csv",
                "normal.csv",
            ],
            "fault_type": ["normal", "normal", "engine", "normal"],
        }
    )

    rows, type_rows, predictions, thresholds = exp._build_threshold_diagnostics(
        scores,
        labels,
        segments,
        val_energy=np.array([0.1, 0.2, 0.3]),
        test_meta=meta,
        score_indices=np.arange(4),
    )

    method_names = {row["Threshold Method"] for row in rows}
    assert {
        "oracle",
        "val_quantile_0.25",
        "val_quantile_0.5",
        "val_quantile_0.75",
        "lagged_atssd",
        "freeze_atssd",
        "median_mad_k2",
        "median_mad_k3",
    }.issubset(method_names)
    assert any(row["PR-AUC"] == 1.0 for row in rows)
    assert any(row["Event Hits"] == 1 for row in rows)
    assert "oracle" in predictions
    assert "median_mad_k3" in thresholds
    assert any(
        row["Threshold Method"] == "oracle"
        and row["Anomaly Type"] == "Engine full power loss"
        and row["F-score"] == 1.0
        for row in type_rows
    )


def test_threshold_sweep_lists_do_not_change_checkpoint_setting():
    base_args = normalize_args(
        build_parser().parse_args(
            [
                "--model",
                "DTSGAD",
                "--use_gpu",
                "false",
                "--paper_strict",
                "false",
                "--implementation_tag",
                "same_model",
            ]
        )
    )
    sweep_args = normalize_args(
        build_parser().parse_args(
            [
                "--model",
                "DTSGAD",
                "--use_gpu",
                "false",
                "--paper_strict",
                "false",
                "--implementation_tag",
                "same_model",
                "--val_quantiles",
                "0.9",
                "0.95",
                "0.99",
                "--mad_ks",
                "2",
                "4",
                "8",
                "--threshold_diagnostics",
                "false",
            ]
        )
    )

    assert build_setting(base_args, 0) == build_setting(sweep_args, 0)
