import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from utils.score_calibration import (
    empirical_tail_calibrate,
    robust_zscore_calibrate,
    topk_mean,
    zscore_calibrate,
)


def test_a10_zscore_calibration_uses_featurewise_train_baseline():
    train_scores = np.array(
        [
            [1.0, 10.0],
            [2.0, 12.0],
            [3.0, 14.0],
        ]
    )
    test_scores = np.array([[4.0, 16.0]])

    calibrated, baseline = zscore_calibrate(test_scores, train_scores)

    np.testing.assert_allclose(baseline["mean"], [2.0, 12.0])
    np.testing.assert_allclose(baseline["std"], [np.sqrt(2.0 / 3.0), np.sqrt(8.0 / 3.0)])
    np.testing.assert_allclose(calibrated[0], [(4.0 - 2.0) / baseline["std"][0], (16.0 - 12.0) / baseline["std"][1]])


def test_a10_robust_calibration_uses_median_and_scaled_mad():
    train_scores = np.array(
        [
            [1.0, 10.0],
            [2.0, 12.0],
            [100.0, 14.0],
        ]
    )
    test_scores = np.array([[3.0, 16.0]])

    calibrated, baseline = robust_zscore_calibrate(test_scores, train_scores)

    np.testing.assert_allclose(baseline["median"], [2.0, 12.0])
    np.testing.assert_allclose(baseline["mad"], [1.0, 2.0])
    np.testing.assert_allclose(
        calibrated[0],
        [(3.0 - 2.0) / 1.4826, (16.0 - 12.0) / (1.4826 * 2.0)],
    )


def test_a10_empirical_tail_calibration_is_featurewise():
    train_scores = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]
    )
    test_scores = np.array([[3.0, 35.0]])

    calibrated, baseline = empirical_tail_calibrate(test_scores, train_scores)

    np.testing.assert_allclose(baseline["train_count"], [4, 4])
    expected_tail = np.array([[2.0 / 5.0, 2.0 / 5.0]])
    np.testing.assert_allclose(baseline["tail_probability"], expected_tail)
    np.testing.assert_allclose(calibrated, -np.log(expected_tail))


def test_topk_mean_uses_largest_k_values_per_row():
    values = np.array([[1.0, 5.0, 2.0], [10.0, -1.0, 3.0]])

    np.testing.assert_allclose(topk_mean(values, 2), [3.5, 6.5])
