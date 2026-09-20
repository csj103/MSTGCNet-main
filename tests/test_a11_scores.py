import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from utils.score_calibration import two_sided_calibrate, topk_selection_frequency


def test_two_sided_scores_detect_both_tails_and_use_train_only():
    train = np.array([[0., 0.], [1., 1.], [2., 2.], [3., 3.], [4., 4.]])
    test = np.array([[-2., 2.], [6., 2.]])
    scores, baseline = two_sided_calibrate(test, train)
    for name in ("A11-Z", "A11-R0", "A11-R1", "A11-E"):
        assert scores[name][0, 0] > scores[name][0, 1]
        assert scores[name][1, 0] > scores[name][1, 1]
        assert np.isfinite(scores[name]).all()
    np.testing.assert_allclose(scores["A11-Z"][:, 0], [4 / np.sqrt(2)] * 2)
    assert baseline["train_count"] == 5


def test_mad_floor_prevents_tiny_scale_from_dominating():
    train = np.array([[0., 0.], [0., 1.], [0., 2.], [0., 3.], [100., 4.]])
    test = np.array([[1., 2.]])
    scores, baseline = two_sided_calibrate(test, train)
    assert scores["A11-R1"][0, 0] < scores["A11-R0"][0, 0]
    assert baseline["robust_floor_scale"][0] >= 0.1 * train[:, 0].std()


def test_topk_frequency_counts_each_selected_variable_once():
    scores = np.array([[3., 2., 1.], [1., 3., 2.]])
    np.testing.assert_allclose(topk_selection_frequency(scores, 2), [0.5, 1., 0.5])
