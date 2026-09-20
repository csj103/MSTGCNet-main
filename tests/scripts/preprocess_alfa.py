import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.alfa_anomaly_types import (
    FINE_GRAINED_ANOMALY_TYPES,
    add_fine_anomaly_type,
)


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
    "fine_anomaly_type",
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
        choices=["paper", "fine_grained", "flight", "fault_balanced"],
        default="fine_grained",
        help=(
            "paper: match the ALFA counts reported by the paper; "
            "fine_grained: keep every fine-grained ALFA anomaly type in test; "
            "flight: use normal flights for train and fault flights for test; "
            "fault_balanced: split whole flights and balance test anomaly counts; "
            "kept for ablation/diagnostics."
        ),
    )
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def load_flight(path):
    df = pd.read_csv(path)
    required = set(ID_COLUMNS + FEATURE_COLUMNS + [LABEL_COLUMN])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = df[ID_COLUMNS + FEATURE_COLUMNS + [LABEL_COLUMN]].copy()
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].bfill().ffill()
    if df[FEATURE_COLUMNS].isna().any().any():
        raise ValueError(f"{path} contains feature columns that cannot be filled")
    df["source_file"] = path.name
    df["label"] = df[LABEL_COLUMN].fillna(0).astype(int)
    df.loc[df["label"] != 0, "label"] = 1
    df = add_fine_anomaly_type(df)
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
            fine_anomaly_type=("fine_anomaly_type", "first"),
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
    train_count = min(len(normal_outside_test), PAPER_TRAIN_VAL_ROWS)
    train, unused_normal = select_rows(
        normal_outside_test,
        train_count,
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


def _flight_summary(data):
    return (
        data.groupby("source_file", sort=True)
        .agg(
            flight_id=("flight_id", "first"),
            fault_type=("fault_type", "first"),
            fine_anomaly_type=("fine_anomaly_type", "first"),
            rows=("label", "size"),
            anomaly_rows=("label", "sum"),
        )
        .reset_index()
    )


def _select_representative_fine_type_files(flights):
    fault_flights = flights[flights["anomaly_rows"] > 0].copy()
    available_types = set(fault_flights["fine_anomaly_type"].unique())
    missing_types = set(FINE_GRAINED_ANOMALY_TYPES) - available_types
    if missing_types:
        raise ValueError(
            "Missing fine-grained ALFA anomaly type(s): "
            + ", ".join(sorted(missing_types))
        )

    selected = []
    for fine_type in FINE_GRAINED_ANOMALY_TYPES:
        group = fault_flights[fault_flights["fine_anomaly_type"].eq(fine_type)]
        if fine_type == "Left aileron stuck at zero":
            single_fault_group = group[
                ~group["source_file"].str.contains("rudder_zero__", case=False)
            ]
            if not single_fault_group.empty:
                group = single_fault_group
        target = group["anomaly_rows"].median()
        ranked = group.assign(
            distance=(group["anomaly_rows"] - target).abs()
        ).sort_values(["distance", "anomaly_rows", "source_file"])
        selected.append(ranked.iloc[0]["source_file"])
    return set(selected)


def _assign_remaining_validation_files(flights, test_files, val_ratio, seed):
    rng = np.random.default_rng(seed)
    remaining = flights[~flights["source_file"].isin(test_files)]
    val_files = set()
    for _, group in remaining.groupby("fault_type", sort=True):
        files = sorted(group["source_file"].tolist())
        rng.shuffle(files)
        if len(files) < 2 or val_ratio == 0:
            continue
        val_count = max(1, int(round(len(files) * val_ratio)))
        val_count = min(val_count, len(files) - 1)
        val_files.update(files[:val_count])
    return val_files


def _apply_file_assignment(data, assignment):
    all_with_split = data.copy()
    all_with_split["split"] = all_with_split["source_file"].map(assignment)
    discarded_anomaly = (
        all_with_split["split"].isin(["train", "val"])
        & all_with_split["label"].eq(1)
    )
    all_with_split.loc[discarded_anomaly, "split"] = "unused"

    train = assign_segments(all_with_split[all_with_split["split"].eq("train")])
    val = assign_segments(all_with_split[all_with_split["split"].eq("val")])
    test = assign_segments(all_with_split[all_with_split["split"].eq("test")])
    return train, val, test, all_with_split


def split_fine_grained_test(data, val_ratio=0.15, seed=2025):
    if not 0 <= val_ratio < 1:
        raise ValueError("--val-ratio must be in [0, 1).")

    flights = _flight_summary(data)
    test_files = _select_representative_fine_type_files(flights)
    val_files = _assign_remaining_validation_files(
        flights,
        test_files=test_files,
        val_ratio=val_ratio,
        seed=seed,
    )

    assignment = {}
    for source_file in flights["source_file"]:
        if source_file in test_files:
            assignment[source_file] = "test"
        elif source_file in val_files:
            assignment[source_file] = "val"
        else:
            assignment[source_file] = "train"
    return _apply_file_assignment(data, assignment)


def closest_anomaly_subset(group, target, max_files=None):
    records = [
        (row.source_file, int(row.anomaly_rows))
        for row in group.itertuples(index=False)
    ]
    reachable = {0: ()}
    for source_file, anomaly_rows in records:
        additions = {
            total + anomaly_rows: selected + (source_file,)
            for total, selected in list(reachable.items())
        }
        for total, selected in additions.items():
            current = reachable.get(total)
            if current is None or len(selected) < len(current):
                reachable[total] = selected

    candidates = [
        (total, files)
        for total, files in reachable.items()
        if files and (max_files is None or len(files) <= max_files)
    ]
    _, selected = min(
        candidates,
        key=lambda item: (abs(item[0] - target), len(item[1]), item[0]),
    )
    return set(selected)


def split_fault_balanced(data, val_ratio=0.15, seed=2025):
    if not 0 <= val_ratio < 1:
        raise ValueError("--val-ratio must be in [0, 1).")

    flights = _flight_summary(data)
    fault_flights = flights[flights["anomaly_rows"] > 0]
    anomaly_totals = fault_flights.groupby("fault_type")["anomaly_rows"].sum()
    if anomaly_totals.empty:
        raise ValueError("No labelled fault flights are available for testing.")
    reservable_totals = []
    for _, group in fault_flights.groupby("fault_type", sort=True):
        if len(group) < 2:
            raise ValueError(
                "fault_balanced requires at least two flights per fault type"
            )
        reservable_totals.append(
            int(group["anomaly_rows"].sum() - group["anomaly_rows"].min())
        )
    target = min(reservable_totals)

    test_files = set()
    for _, group in fault_flights.groupby("fault_type", sort=True):
        test_files.update(
            closest_anomaly_subset(group, target, max_files=len(group) - 1)
        )

    val_files = _assign_remaining_validation_files(
        flights,
        test_files=test_files,
        val_ratio=val_ratio,
        seed=seed,
    )

    assignment = {}
    for source_file in flights["source_file"]:
        if source_file in test_files:
            assignment[source_file] = "test"
        elif source_file in val_files:
            assignment[source_file] = "val"
        else:
            assignment[source_file] = "train"

    train, val, test, all_with_split = _apply_file_assignment(data, assignment)
    return train, val, test, all_with_split, target


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
        )
        if skip_no_ground_truth:
            skipped_files.append(path.name)
            continue
        frames.append(load_flight(path))

    if not frames:
        raise ValueError("No usable ALFA files remained after filtering.")

    data = pd.concat(frames, ignore_index=True)
    data["original_index"] = np.arange(len(data))

    val = None
    balance_target = None
    if args.split_policy == "fault_balanced":
        train, val, test, all_with_split, balance_target = split_fault_balanced(
            data,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    elif args.split_policy == "fine_grained":
        train, val, test, all_with_split = split_fine_grained_test(
            data,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    elif args.split_policy == "paper":
        train, test, all_with_split = split_like_paper(data)
    else:
        train, test, all_with_split = split_by_flight(data)

    train_output = train[OUTPUT_COLUMNS].reset_index(drop=True)
    test_output = test[OUTPUT_COLUMNS].reset_index(drop=True)
    train_meta = train[META_COLUMNS].reset_index(drop=True)
    test_meta = test[META_COLUMNS].reset_index(drop=True)
    if val is not None:
        val_output = val[OUTPUT_COLUMNS].reset_index(drop=True)
        val_meta = val[META_COLUMNS].reset_index(drop=True)

    if train_output.empty:
        raise ValueError("Training split is empty; no normal labelled rows found.")
    if test_output.empty:
        raise ValueError("Test split is empty; no fault rows found.")

    output.mkdir(parents=True, exist_ok=True)
    train_output.to_csv(output / "train.csv", index=False)
    test_output.to_csv(output / "test.csv", index=False)
    train_meta.to_csv(output / "train_meta.csv", index=False)
    test_meta.to_csv(output / "test_meta.csv", index=False)
    if val is not None:
        val_output.to_csv(output / "val.csv", index=False)
        val_meta.to_csv(output / "val_meta.csv", index=False)
    else:
        for stale_path in (output / "val.csv", output / "val_meta.csv"):
            if stale_path.exists():
                stale_path.unlink()

    summary = summarize_flights(all_with_split)
    summary.to_csv(output / "split_summary.csv", index=False)

    metadata = {
        "source": str(source),
        "output": str(output),
        "split_policy": args.split_policy,
        "split_seed": args.seed,
        "validation_ratio": args.val_ratio,
        "features": FEATURE_COLUMNS,
        "train_rows": int(len(train_output)),
        "val_rows": int(len(val_output)) if val is not None else None,
        "train_val_rows": int(
            len(train_output) + (len(val_output) if val is not None else 0)
        ),
        "paper_train_val_target_rows": int(PAPER_TRAIN_VAL_ROWS),
        "matched_paper_train_val_rows": bool(
            len(train_output) + (len(val_output) if val is not None else 0)
            == PAPER_TRAIN_VAL_ROWS
        ),
        "loader_train_rows": (
            int(len(train_output))
            if val is not None
            else int(len(train_output) * 0.9)
        ),
        "loader_val_rows": (
            int(len(val_output))
            if val is not None
            else int(len(train_output) - int(len(train_output) * 0.9))
        ),
        "test_rows": int(len(test_output)),
        "test_anomaly_rows": int(test_output["label"].sum()),
        "test_anomaly_ratio": float(test_output["label"].mean()),
        "unused_rows": int((all_with_split["split"] == "unused").sum()),
        "test_anomaly_intervals": int(
            (test_output["label"].ne(test_output["label"].shift()).cumsum())
            [test_output["label"].eq(1)]
            .nunique()
        ),
        "fault_balance_target": balance_target,
        "test_anomaly_rows_by_fault": {
            str(name): int(value)
            for name, value in test.groupby("fault_type")["label"].sum().items()
            if int(value) > 0
        },
        "test_anomaly_rows_by_fine_type": {
            str(name): int(value)
            for name, value in test.groupby("fine_anomaly_type")["label"].sum().items()
            if int(value) > 0
        },
        "flight_counts_by_split": {
            str(name): int(value)
            for name, value in all_with_split[
                all_with_split["split"].ne("unused")
            ].groupby("split")["source_file"].nunique().items()
        },
        "included_no_ground_truth": bool(args.include_no_ground_truth),
        "skipped_files": skipped_files,
        "output_columns": OUTPUT_COLUMNS,
    }
    with (output / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote {output / 'train.csv'} ({len(train_output)} rows)")
    print(f"Wrote {output / 'test.csv'} ({len(test_output)} rows)")
    if val is not None:
        print(f"Wrote {output / 'val.csv'} ({len(val_output)} rows)")
    print(
        "Loader split: "
        f"{metadata['loader_train_rows']} train / {metadata['loader_val_rows']} val"
    )
    if (
        args.split_policy == "paper"
        and not metadata["matched_paper_train_val_rows"]
    ):
        print(
            "Warning: train+val normal rows do not match the paper target "
            f"({metadata['train_val_rows']} vs {metadata['paper_train_val_target_rows']}) "
            "after excluding no-ground-truth files."
        )
    print(f"Test anomaly ratio: {metadata['test_anomaly_ratio']:.4f}")
    if skipped_files:
        print(f"Skipped {len(skipped_files)} no-ground-truth file(s)")


if __name__ == "__main__":
    main()
