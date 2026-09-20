import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.score_diagnostics import DEFAULT_FAULT_TYPES, run_score_diagnostics


def latest_result_dir(label_root):
    label_root = Path(label_root)
    candidates = [
        path
        for path in label_root.rglob("*")
        if path.is_dir()
        and (path / "score_components" / "s_obs_topk.npy").exists()
        and (path / "test_labels.npy").exists()
        and (path / "test_indices.npy").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            "No result directory with saved DTSGAD score components was found."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_parser():
    parser = argparse.ArgumentParser(description="Analyze saved DTSGAD scores.")
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--label-root", type=Path, default=Path("label_results"))
    parser.add_argument("--data-root", type=Path, default=Path("dataset/ALFA10vars"))
    parser.add_argument("--test-meta", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--fault-types",
        nargs="+",
        default=DEFAULT_FAULT_TYPES,
    )
    parser.add_argument(
        "--obs-topk",
        type=int,
        default=3,
        help="Top-k feature errors used by M2 train-standardized diagnostics.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    result_dir = args.result_dir or latest_result_dir(args.label_root)
    output_dir = run_score_diagnostics(
        result_dir=result_dir,
        data_root=args.data_root,
        test_meta_path=args.test_meta,
        output_dir=args.output_dir,
        fault_types=args.fault_types,
        obs_topk=args.obs_topk,
    )
    def print_artifact(label, path):
        status = "exists" if path.exists() else "missing"
        print(f"{label} [{status}]:", path)

    print("Result directory:", result_dir)
    print("Diagnostics saved to:", output_dir)
    print_artifact("S1", output_dir / "s1_component_auc.csv")
    print_artifact("S1", output_dir / "s1_component_distribution_stats.csv")
    print_artifact("S1", output_dir / "s1_score_auc_by_anomaly_type.csv")
    print_artifact("S1", output_dir / "s1_score_auc_by_anomaly_type.png")
    print_artifact("S1", output_dir / "s1_obs_feature_auc.csv")
    print_artifact("S1", output_dir / "s1_obs_feature_distribution_stats.csv")
    print_artifact("M2", output_dir / "m2_prediction_feature_auc.csv")
    print_artifact("M2", output_dir / "m2_train_standardized_topk_auc.csv")
    print_artifact("S4", output_dir / "s4_probabilistic_score_auc.csv")
    print_artifact("S4", output_dir / "s4_probabilistic_score_distribution_stats.csv")
    print_artifact("S4", output_dir / "s4_probabilistic_feature_auc.csv")
    print_artifact("S5", output_dir / "s5_logvar_stats.csv")
    print_artifact("S5", output_dir / "s5_logvar_distribution.png")
    print_artifact("S3", output_dir / "s3_score_distribution.png")
    print_artifact("R1", output_dir / "r1_router_gate_stats.csv")
    print_artifact("R1", output_dir / "r1_router_entropy_stats.csv")
    print_artifact("R1", output_dir / "r1_router_summary.csv")
    print_artifact("R1", output_dir / "r1_router_gate_distribution.png")
    print_artifact("R2", output_dir / "r2_expert_cosine_stats.csv")
    print_artifact("G1", output_dir / "g1_pre_post_graph_cosine.csv")
    print_artifact("G4", output_dir / "g4_graph_trace_cosine.csv")
    print_artifact("G4", output_dir / "g4_graph_trace_summary.csv")
    print_artifact("R3", output_dir / "r3_single_expert_auc.csv")
    print_artifact("R5", output_dir / "r5_expert_score_fusion_auc.csv")
    print_artifact("R6", output_dir / "r6_single_expert_auc_by_anomaly_type.csv")
    print_artifact("R4", output_dir / "r4_adjacency_stats.csv")
    for fault_type in args.fault_types:
        print_artifact(
            "S2",
            output_dir
            / (
                "s2_timeseries_"
                + "".join(ch if ch.isalnum() else "_" for ch in fault_type).strip("_")
                + ".png"
            ),
        )
        print_artifact(
            "S6",
            output_dir
            / (
                "s6_mse_vs_nll_"
                + "".join(ch if ch.isalnum() else "_" for ch in fault_type).strip("_")
                + ".png"
            ),
        )


if __name__ == "__main__":
    main()
