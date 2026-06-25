import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "alt_error",
    "aspd_error",
    "xtrack_error",
    "wp_dist",
    "nav_roll",
    "nav_pitch",
    "nav_yaw",
    "nav_x",
    "nav_y",
    "nav_z",
]

ID_COLUMNS = ["timestamp_ns", "time_sec", "flight_id", "fault_type"]
LABEL_COLUMN = "label_anomaly"
OUTPUT_COLUMNS = ["time_sec", *FEATURE_COLUMNS, "label"]
META_COLUMNS = [
    "source_file",
    "flight_id",
    "fault_type",
    "original_index",
    "segment_id",
    "time_sec",
    "label",
]
PAPER_TRAIN_VAL_ROWS = 58321
PAPER_TEST_ROWS = 24556
PAPER_TEST_ANOMALY_ROWS = 6139


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare the selected ALFA variables for MSTGCNet."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("alfa_10vars"),
        help="Directory containing per-flight ALFA csv files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset") / "ALFA10vars",
        help="Output directory for train.csv, test.csv, and metadata.",
    )
    parser.add_argument(
        "--include-no-ground-truth",
        action="store_true",
        help="Include files whose names contain no_ground_truth.",
    )
    parser.add_argument(
        "--split-policy",
        choices=["paper", "flight"],
        default="paper",
        help=(
            "paper: match the ALFA counts reported by the paper; "
            "flight: use normal flights for train and fault flights for test."
        ),
    )
    return parser.parse_args()


def load_flight(path):
    df = pd.read_csv(path)
    required = set(ID_COLUMNS + FEATURE_COLUMNS + [LABEL_COLUMN])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = df[ID_COLUMNS + FEATURE_COLUMNS + [LABEL_COLUMN]].copy()
    df["source_file"] = path.name
    df["label"] = df[LABEL_COLUMN].fillna(0).astype(int)
    df.loc[df["label"] != 0, "label"] = 1
    return df


def summarize_flights(df):
    if "split" not in df.columns:
        df = df.copy()
        df["split"] = "unused"

    summary = (
        df.groupby(["source_file", "split"])
        .agg(
            flight_id=("flight_id", "first"),
            fault_type=("fault_type", "first"),
            rows=("label", "size"),
            anomaly_rows=("label", "sum"),
        )
        .reset_index()
    )
    return summary


def select_rows(rows, count, description):
    if len(rows) < count:
        raise ValueError(
            f"Not enough rows for {description}: required {count}, found {len(rows)}"
        )
    return rows.iloc[:count].copy(), rows.iloc[count:].copy()


def assign_segments(df):
    df = df.sort_values("original_index").reset_index(drop=True).copy()
    boundary = (
        df["source_file"].ne(df["source_file"].shift())
        | df["original_index"].diff().fillna(1).ne(1)
    )
    df["segment_id"] = boundary.cumsum().astype(int) - 1
    return df


def find_contiguous_test_window(data):
    labels = data["label"].to_numpy()
    if len(labels) < PAPER_TEST_ROWS:
        raise ValueError("Not enough rows to build the paper-sized ALFA test split.")

    anomaly_count = int(labels[:PAPER_TEST_ROWS].sum())
    for start in range(0, len(labels) - PAPER_TEST_ROWS + 1):
        if start > 0:
            anomaly_count += int(labels[start + PAPER_TEST_ROWS - 1])
            anomaly_count -= int(labels[start - 1])
        if anomaly_count == PAPER_TEST_ANOMALY_ROWS:
            return start, start + PAPER_TEST_ROWS

    raise ValueError(
        "Could not find a contiguous ALFA test window matching the paper counts."
    )


def split_like_paper(data):
    test_start, test_end = find_contiguous_test_window(data)
    test = data.iloc[test_start:test_end].copy()
    outside_test = pd.concat(
        [data.iloc[:test_start], data.iloc[test_end:]], ignore_index=True
    )
    normal_outside_test = outside_test[outside_test["label"] == 0]
    anomaly_outside_test = outside_test[outside_test["label"] == 1]
    train, unused_normal = select_rows(
        outside_test[outside_test["label"] == 0],
        PAPER_TRAIN_VAL_ROWS,
        "paper train+val normal rows",
    )
    train["split"] = "train_val"
    test["split"] = "test"
    unused = pd.concat([unused_normal, anomaly_outside_test], ignore_index=True)
    unused["split"] = "unused"

    train = assign_segments(train)
    test = assign_segments(test)
    unused = assign_segments(unused)
    all_with_split = pd.concat([train, test, unused], ignore_index=True)
    return train, test, all_with_split


def split_by_flight(data):
    normal_mask = (data["fault_type"] == "normal") & (data["label"] == 0)
    train = data.loc[normal_mask].copy()
    test = data.loc[~normal_mask].copy()
    train["split"] = "train_val"
    test["split"] = "test"
    train = assign_segments(train)
    test = assign_segments(test)
    all_with_split = pd.concat([train, test], ignore_index=True)
    return train, test, all_with_split


def main():
    args = parse_args()
    source = args.source
    output = args.output

    if not source.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source}")

    csv_files = sorted(source.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No csv files found under: {source}")

    frames = []
    skipped_files = []
    for path in csv_files:
        skip_no_ground_truth = (
            "no_ground_truth" in path.name
            and not args.include_no_ground_truth
            and args.split_policy != "paper"
        )
        if skip_no_ground_truth:
            skipped_files.append(path.name)
            continue
        frames.append(load_flight(path))

    if not frames:
        raise ValueError("No usable ALFA files remained after filtering.")

    data = pd.concat(frames, ignore_index=True)
    data["original_index"] = np.arange(len(data))

    feature_block = data[FEATURE_COLUMNS]
    if feature_block.isna().any().any():
        data[FEATURE_COLUMNS] = feature_block.bfill().ffill()

    if args.split_policy == "paper":
        train, test, all_with_split = split_like_paper(data)
    else:
        train, test, all_with_split = split_by_flight(data)

    train_output = train[OUTPUT_COLUMNS].reset_index(drop=True)
    test_output = test[OUTPUT_COLUMNS].reset_index(drop=True)
    train_meta = train[META_COLUMNS].reset_index(drop=True)
    test_meta = test[META_COLUMNS].reset_index(drop=True)

    if train_output.empty:
        raise ValueError("Training split is empty; no normal labelled rows found.")
    if test_output.empty:
        raise ValueError("Test split is empty; no fault rows found.")

    output.mkdir(parents=True, exist_ok=True)
    train_output.to_csv(output / "train.csv", index=False)
    test_output.to_csv(output / "test.csv", index=False)
    train_meta.to_csv(output / "train_meta.csv", index=False)
    test_meta.to_csv(output / "test_meta.csv", index=False)

    summary = summarize_flights(all_with_split)
    summary.to_csv(output / "split_summary.csv", index=False)

    metadata = {
        "source": str(source),
        "output": str(output),
        "split_policy": args.split_policy,
        "features": FEATURE_COLUMNS,
        "train_val_rows": int(len(train_output)),
        "loader_train_rows": int(len(train_output) * 0.9),
        "loader_val_rows": int(len(train_output) - int(len(train_output) * 0.9)),
        "test_rows": int(len(test_output)),
        "test_anomaly_rows": int(test_output["label"].sum()),
        "test_anomaly_ratio": float(test_output["label"].mean()),
        "unused_rows": int((all_with_split["split"] == "unused").sum()),
        "test_anomaly_intervals": int(
            (test_output["label"].ne(test_output["label"].shift()).cumsum())
            [test_output["label"].eq(1)]
            .nunique()
        ),
        "included_no_ground_truth": bool(
            args.include_no_ground_truth or args.split_policy == "paper"
        ),
        "skipped_files": skipped_files,
        "output_columns": OUTPUT_COLUMNS,
    }
    with (output / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote {output / 'train.csv'} ({len(train_output)} rows)")
    print(f"Wrote {output / 'test.csv'} ({len(test_output)} rows)")
    print(
        "Loader split: "
        f"{metadata['loader_train_rows']} train / {metadata['loader_val_rows']} val"
    )
    print(f"Test anomaly ratio: {metadata['test_anomaly_ratio']:.4f}")
    if skipped_files:
        print(f"Skipped {len(skipped_files)} no-ground-truth file(s)")


if __name__ == "__main__":
    main()
