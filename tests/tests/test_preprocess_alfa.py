from pathlib import Path

import numpy as np
import pandas as pd

from data_provider.data_loader import Dataset_ALFA, fill_features_by_segment
from scripts.preprocess_alfa import (
    FINE_GRAINED_ANOMALY_TYPES,
    load_flight,
    split_fault_balanced,
    split_fine_grained_test,
)


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


def test_fine_grained_split_puts_every_anomaly_type_in_test():
    paths = [
        path
        for path in sorted(Path("alfa_10vars").glob("*.csv"))
        if "no_ground_truth" not in path.name
    ]
    data = pd.concat([load_flight(path) for path in paths], ignore_index=True)
    data["original_index"] = np.arange(len(data))

    train, val, test, all_rows = split_fine_grained_test(data, seed=2025)
    split_files = {
        split: set(all_rows.loc[all_rows["split"].eq(split), "source_file"])
        for split in ("train", "val", "test")
    }
    test_fine_types = set(
        test.loc[test["label"].eq(1), "fine_anomaly_type"].unique()
    )

    assert train["label"].sum() == 0
    assert val["label"].sum() == 0
    assert not split_files["train"] & split_files["val"]
    assert not split_files["train"] & split_files["test"]
    assert not split_files["val"] & split_files["test"]
    assert set(FINE_GRAINED_ANOMALY_TYPES) <= test_fine_types
    assert test.groupby("fine_anomaly_type")["source_file"].nunique().min() == 1
    left_files = set(
        test.loc[
            test["fine_anomaly_type"].eq("Left aileron stuck at zero"),
            "source_file",
        ]
    )
    assert not any("rudder_zero__" in source_file for source_file in left_files)


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


def test_fine_grained_loader_uses_explicit_validation_split(tmp_path):
    columns = ["time_sec", "f1", "label"]
    pd.DataFrame(
        [[0.0, 1.0, 0], [1.0, 2.0, 0], [2.0, 3.0, 0], [3.0, 4.0, 0]],
        columns=columns,
    ).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame(
        [[10.0, 5.0, 0], [11.0, 6.0, 0]],
        columns=columns,
    ).to_csv(tmp_path / "val.csv", index=False)
    pd.DataFrame(
        [[20.0, 7.0, 0], [21.0, 8.0, 1]],
        columns=columns,
    ).to_csv(tmp_path / "test.csv", index=False)
    pd.DataFrame({"segment_id": [0, 0, 0, 0]}).to_csv(
        tmp_path / "train_meta.csv", index=False
    )
    pd.DataFrame({"segment_id": [1, 1]}).to_csv(
        tmp_path / "val_meta.csv", index=False
    )
    pd.DataFrame({"segment_id": [2, 2]}).to_csv(
        tmp_path / "test_meta.csv", index=False
    )
    (tmp_path / "metadata.json").write_text(
        '{"split_policy": "fine_grained"}',
        encoding="utf-8",
    )

    dataset = Dataset_ALFA(None, str(tmp_path), win_size=2, flag="val")

    assert len(dataset.val) == 2
    assert len(dataset) == 1


def test_alfa_next_step_prediction_windows_exclude_target_point(tmp_path):
    columns = ["time_sec", "f1", "f2", "label"]
    pd.DataFrame(
        [
            [0.0, 0.0, 10.0, 0],
            [1.0, 1.0, 11.0, 0],
            [2.0, 2.0, 12.0, 0],
            [3.0, 3.0, 13.0, 0],
            [4.0, 4.0, 14.0, 0],
            [5.0, 5.0, 15.0, 0],
        ],
        columns=columns,
    ).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame(
        [
            [10.0, 10.0, 20.0, 0],
            [11.0, 11.0, 21.0, 0],
            [12.0, 12.0, 22.0, 0],
            [13.0, 13.0, 23.0, 0],
        ],
        columns=columns,
    ).to_csv(tmp_path / "val.csv", index=False)
    pd.DataFrame(
        [
            [20.0, 20.0, 30.0, 0],
            [21.0, 21.0, 31.0, 0],
            [22.0, 22.0, 32.0, 0],
            [23.0, 23.0, 33.0, 1],
            [24.0, 24.0, 34.0, 1],
        ],
        columns=columns,
    ).to_csv(tmp_path / "test.csv", index=False)
    pd.DataFrame({"segment_id": [0, 0, 0, 0, 0, 0]}).to_csv(
        tmp_path / "train_meta.csv", index=False
    )
    pd.DataFrame({"segment_id": [1, 1, 1, 1]}).to_csv(
        tmp_path / "val_meta.csv", index=False
    )
    pd.DataFrame({"segment_id": [2, 2, 2, 2, 2]}).to_csv(
        tmp_path / "test_meta.csv", index=False
    )
    (tmp_path / "metadata.json").write_text(
        '{"split_policy": "fine_grained"}',
        encoding="utf-8",
    )
    args = type(
        "Args",
        (),
        {"dtsgad_objective": "next_step_prediction", "target_horizon": 1},
    )()

    dataset = Dataset_ALFA(args, str(tmp_path), win_size=3, flag="train")
    input_window, target_point, marks = dataset[0]

    assert len(dataset) == 3
    assert input_window.shape == (3, 2)
    assert target_point.shape == (2,)
    assert marks.tolist() == [0.0, 1.0, 2.0]
    assert np.allclose(input_window, dataset.train[0:3])
    assert np.allclose(target_point, dataset.train[3])
