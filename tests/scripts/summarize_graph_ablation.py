import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.score_diagnostics import run_score_diagnostics


def _safe_metric(func, labels, scores):
    labels = np.asarray(labels).astype(int)
    if np.unique(labels).size < 2:
        return float("nan")
    return float(func(labels, scores))


def _load_post_graph_cosine(result_dir, data_root):
    result_dir = Path(result_dir)
    table_path = result_dir / "score_diagnostics" / "g1_pre_post_graph_cosine.csv"
    if not table_path.exists():
        run_score_diagnostics(result_dir=result_dir, data_root=data_root)
    if not table_path.exists():
        return {}
    table = pd.read_csv(table_path)
    table = table[
        (table["position"] == "post_graph")
        & (table["label_group"] == "all")
    ]
    return {
        int(block): float(group["mean"].mean())
        for block, group in table.groupby("block")
    }


def summarize_result(model_name, result_dir, data_root):
    result_dir = Path(result_dir)
    scores = np.load(result_dir / "test_energy.npy")
    labels = np.load(result_dir / "test_labels.npy").astype(int)
    labels[labels != 0] = 1
    pred = np.load(result_dir / "raw_pred.npy").astype(int)
    cosine_by_block = _load_post_graph_cosine(result_dir, data_root)
    return {
        "Model": model_name,
        "ROC-AUC": _safe_metric(roc_auc_score, labels, scores),
        "PR-AUC": _safe_metric(average_precision_score, labels, scores),
        "Raw F1": float(f1_score(labels, pred)),
        "B1 cosine": cosine_by_block.get(1, float("nan")),
        "B2 cosine": cosine_by_block.get(2, float("nan")),
        "B3 cosine": cosine_by_block.get(3, float("nan")),
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Summarize DTSGAD graph ablation experiments."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("dataset/ALFA10vars"),
    )
    parser.add_argument(
        "--experiment",
        nargs=2,
        action="append",
        metavar=("NAME", "RESULT_DIR"),
        required=True,
        help="Model name and its causal_last result directory.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    rows = [
        summarize_result(name, Path(result_dir), args.data_root)
        for name, result_dir in args.experiment
    ]
    table = pd.DataFrame(rows)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
