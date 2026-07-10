from pathlib import Path

import numpy as np
import pandas as pd

from data_provider.data_loader import fill_features_by_segment
from scripts.preprocess_alfa import load_flight, split_fault_balanced


def test_fault_balanced_split_has_no_leakage_and_balances_faults():
    paths = [
        path
        for path in sorted(Path("alfa_10vars").glob("*.csv"))
        if "no_ground_truth" not in path.name
    ]
    data = pd.concat([load_flight(path) for path in paths], ignore_index=True)
    data["original_index"] = np.arange(len(data))

    train, val, test, all_rows, target = split_fault_balanced(data, seed=2025)
    split_files = {
        split: set(all_rows.loc[all_rows["split"].eq(split), "source_file"])
        for split in ("train", "val", "test")
    }
    anomaly_counts = test.groupby("fault_type")["label"].sum().to_dict()

    assert target == 244
    assert train["label"].sum() == 0
    assert val["label"].sum() == 0
    assert not split_files["train"] & split_files["val"]
    assert not split_files["train"] & split_files["test"]
    assert not split_files["val"] & split_files["test"]
    assert set(anomaly_counts) == {"aileron", "elevator", "engine", "rudder"}
    assert max(anomaly_counts.values()) - min(anomaly_counts.values()) <= 100
    train_fault_types = set(
        all_rows.loc[all_rows["split"].eq("train"), "fault_type"]
    )
    assert {"aileron", "elevator", "engine", "rudder"} <= train_fault_types


def test_missing_values_are_filled_within_segments_only():
    data = pd.DataFrame(
        {
            "time_sec": [0.0, 1.0, 0.0, 1.0],
            "feature": [1.0, np.nan, np.nan, 9.0],
            "label": [0, 0, 0, 0],
        }
    )
    meta = pd.DataFrame({"segment_id": [0, 0, 1, 1]})
    filled = fill_features_by_segment(data, meta)
    assert filled["feature"].tolist() == [1.0, 1.0, 9.0, 9.0]
