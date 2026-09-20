import numpy as np
import pandas as pd

from exp.exp_anomaly_detection import Exp_Anomaly_Detection


def test_alfa_source_files_map_to_table_viii_anomaly_types():
    cases = {
        "carbonZ_2018-09-11-14-22-07_2_engine_failure.csv": "Engine full power loss",
        "carbonZ_2018-09-11-15-05-11_1_elevator_failure.csv": "Elevator stuck at zero",
        "carbonZ_2018-09-11-15-06-34_3_rudder_left_failure.csv": "Rudder stuck to left",
        "carbonZ_2018-09-11-15-06-34_1_rudder_right_failure.csv": "Rudder stuck to right",
        "carbonZ_2018-09-11-17-27-13_2_both_ailerons_failure.csv": "Both aileron stuck at zero",
        "carbonZ_2018-10-05-14-37-22_3_left_aileron_failure.csv": "Left aileron stuck at zero",
        "carbonZ_2018-10-05-14-37-22_2_right_aileron_failure.csv": "Right aileron stuck at zero",
    }

    for source_file, expected in cases.items():
        row = {"source_file": source_file, "fault_type": "aileron"}
        assert Exp_Anomaly_Detection._canonical_anomaly_type(row) == expected


def test_anomaly_type_metrics_align_metadata_with_scored_indices():
    meta = pd.DataFrame(
        {
            "source_file": [
                "normal.csv",
                "carbonZ_2018-09-11-14-22-07_2_engine_failure.csv",
                "carbonZ_2018-09-11-14-22-07_2_engine_failure.csv",
                "unused.csv",
                "carbonZ_2018-09-11-15-06-34_1_rudder_right_failure.csv",
                "carbonZ_2018-09-11-15-06-34_1_rudder_right_failure.csv",
            ],
            "fault_type": [
                "normal",
                "engine",
                "engine",
                "normal",
                "rudder",
                "rudder",
            ],
        }
    )
    score_indices = np.array([1, 2, 4, 5])
    gt = np.array([0, 1, 0, 1])
    pred = np.array([1, 1, 0, 0])

    rows = Exp_Anomaly_Detection._metrics_by_anomaly_type(
        gt, pred, meta, score_indices
    )

    assert [row["Anomaly Type"] for row in rows] == [
        "Engine full power loss",
        "Rudder stuck to right",
    ]
    assert rows[0]["Points"] == 2
    assert rows[0]["Anomalies"] == 1
    assert rows[0]["Accuracy"] == 0.5
    assert rows[0]["Precision"] == 0.5
    assert rows[0]["Recall"] == 1.0
    assert rows[0]["F-score"] == 2 / 3
    assert rows[1]["Points"] == 2
    assert rows[1]["Anomalies"] == 1
    assert rows[1]["F-score"] == 0.0
